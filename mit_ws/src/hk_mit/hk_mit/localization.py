"""MIT 위치 추정: 바퀴·IMU 예측 + GNSS 보정, 50 Hz odom/TF.

(x, y, yaw, 자이로 편향)을 추정하는 EKF. GNSS 음영에서도 이동을 예측하고,
GNSS의 잡음·이상값과 IMU 편향을 보정한다. 정지 중에는 자이로 영점을 보정한다.
경로에 위치를 강제로 맞추지 않으므로 회피·후진에도 사용할 수 있다.
이 파일만 교체하면 된다. numpy는 대회 Docker 이미지에 포함되어 있다.
조정값은 DEFAULTS 및 같은 이름의 ROS 파라미터에 있다.
"""
import heapq
import math
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from tf2_ros import TransformBroadcaster
from .course import declare_file_params, load_course
from .sim_events import SimEvents
from .utm import ll_to_map

RATE_HZ = 50.0
DEFAULTS = {
    'gps_sigma': 0.70,              # GNSS 위치 표준편차 가정 (m)
    'gps_update_period': 0.15,      # 상관된 GPS를 과신하지 않도록 보정 간격 제한 (s)
    'gps_gate': 25.0,               # 정규화한 2차원 위치 잔차 제곱의 상한
    'max_gps_step': 0.25,           # GNSS 보정 한 번의 최대 위치 이동 (m)
    'position_noise': 0.035,        # 이동 모델 위치 불확실성 (m/sqrt(s))
    'gyro_noise': 0.004,            # 방향 예측 불확실성 (rad/sqrt(s))
    'bias_walk': 0.00012,           # IMU 편향 변화 불확실성 (rad/s/sqrt(s))
    'stationary_gyro_sigma': 0.008, # 정지 중 자이로 영점 측정 표준편차 (rad/s)
    'sensor_timeout': 0.30,         # 오래된 속도·각속도는 계속 적분하지 않음 (s)
    'fusion_delay': 0.04,          # 토픽 사이 수신 순서를 정렬할 여유 (s)
}


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PoseFilter:
    """ROS와 독립적인 계산부. 기록 재계산에서도 동일한 코드를 사용한다."""
    def __init__(self, pose, timestamp, parameters=None):
        self.cfg = dict(DEFAULTS)
        if parameters:
            self.cfg.update(parameters)
        self.state = np.array([*pose, 0.0], dtype=float)
        self.cov = np.diag([0.08**2, 0.08**2, math.radians(0.5)**2, 0.005**2])
        self.time = float(timestamp)
        self.speed = self.rate = 0.0
        self.wheel_time = self.imu_time = None
        self.still_since = self.last_gps_update = None
        self.counts = Counter()

    def motion(self, timestamp):
        timeout = self.cfg['sensor_timeout']
        speed = self.speed if self.wheel_time is not None and 0 <= timestamp - self.wheel_time <= timeout else 0.0
        rate = self.rate if self.imu_time is not None and 0 <= timestamp - self.imu_time <= timeout else None
        return speed, rate

    def _predict(self, state, cov, begin, end):
        remaining = end - begin
        if remaining <= 0:
            return state, cov
        if remaining > 1.0:
            # 센서 전체의 긴 공백을 오래된 입력으로 채우지 않는다.
            return state, cov + np.diag([4.0, 4.0, 0.03, 0.00001]) * min(remaining, 30.0)
        while remaining > 1e-9:
            dt = min(remaining, 0.05)
            speed, measured_rate = self.motion(begin + dt * 0.5)
            moving = abs(speed) >= 0.04
            omega = measured_rate - state[3] if measured_rate is not None and moving else 0.0
            mid_yaw = state[2] + omega * dt * 0.5
            distance = speed * dt
            s, c = math.sin(mid_yaw), math.cos(mid_yaw)
            transition = np.eye(4)
            transition[0, 2], transition[1, 2] = -distance * s, distance * c
            if measured_rate is not None and moving:
                transition[0, 3] = 0.5 * distance * dt * s
                transition[1, 3] = -0.5 * distance * dt * c
                transition[2, 3] = -dt
            state = state.copy()
            state[0] += distance * c
            state[1] += distance * s
            state[2] = wrap(state[2] + omega * dt)
            noise = np.diag([self.cfg['position_noise']**2 * dt,
                             self.cfg['position_noise']**2 * dt,
                             self.cfg['gyro_noise']**2 * dt,
                             self.cfg['bias_walk']**2 * dt])
            cov = transition @ cov @ transition.T + noise
            begin += dt
            remaining = end - begin
        return state, (cov + cov.T) * 0.5

    def advance(self, timestamp):
        if timestamp < self.time - 1e-6:
            self.counts['old_measurement'] += 1
            return False
        self.state, self.cov = self._predict(self.state, self.cov, self.time, timestamp)
        self.time = max(self.time, timestamp)
        return True

    def wheel(self, timestamp, speed):
        if not math.isfinite(speed) or not self.advance(timestamp):
            return
        self.speed, self.wheel_time = float(speed), timestamp
        if abs(speed) < 0.04:
            if self.still_since is None:
                self.still_since = timestamp
        else:
            self.still_since = None

    def imu(self, timestamp, rate):
        if not math.isfinite(rate) or not self.advance(timestamp):
            return
        self.rate, self.imu_time = float(rate), timestamp
        if (self.still_since is not None and timestamp - self.still_since >= 0.15
                and self.wheel_time is not None and timestamp - self.wheel_time < 0.1
                and abs(rate) < 0.1):
            variance = self.cfg['stationary_gyro_sigma']**2
            gain = self.cov[:, 3] / (self.cov[3, 3] + variance)
            self.state += gain * (rate - self.state[3])
            residual = np.eye(4)
            residual[:, 3] -= gain
            self.cov = residual @ self.cov @ residual.T + np.outer(gain, gain) * variance
            self._normalize()
            self.counts['stationary_bias_update'] += 1

    def gps(self, timestamp, x, y):
        if not all(math.isfinite(value) for value in (x, y)) or not self.advance(timestamp):
            return
        if self.last_gps_update is not None and timestamp - self.last_gps_update < self.cfg['gps_update_period'] - 1e-6:
            return
        innovation = np.array([x, y]) - self.state[:2]
        measurement_cov = np.eye(2) * self.cfg['gps_sigma']**2
        innovation_cov = self.cov[:2, :2] + measurement_cov
        score = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        if score > self.cfg['gps_gate']:
            self.counts['gps_rejected'] += 1
            return
        if score > 9.0:
            measurement_cov *= score / 9.0
            innovation_cov = self.cov[:2, :2] + measurement_cov
        gain = np.linalg.solve(innovation_cov, self.cov[:2, :]).T
        speed, _ = self.motion(timestamp)
        if abs(speed) < 0.3:
            gain[2:, :] = 0.0  # 정지 중 위치 잡음으로 방향·편향을 바꾸지 않음
        delta = gain @ innovation
        scale = min(1.0, self.cfg['max_gps_step'] / max(np.linalg.norm(delta[:2]), 1e-12),
                    math.radians(1.0) / max(abs(delta[2]), 1e-12))
        gain *= scale
        self.state += gain @ innovation
        residual = np.eye(4)
        residual[:, :2] -= gain
        self.cov = residual @ self.cov @ residual.T + gain @ measurement_cov @ gain.T
        self.last_gps_update = timestamp
        self._normalize()
        self.counts['gps_update'] += 1

    def _normalize(self):
        self.state[2] = wrap(self.state[2])
        self.state[3] = float(np.clip(self.state[3], -0.05, 0.05))
        self.cov = (self.cov + self.cov.T) * 0.5

    def estimate(self, timestamp):
        # 정렬 여유로 생긴 지연을 현재 시각까지 예측해서 출력. 내부 상태는 유지한다.
        return self._predict(self.state.copy(), self.cov.copy(), self.time, timestamp)


class Localization(Node):
    def __init__(self):
        super().__init__('hk_mit_localization')
        declare_file_params(self)
        self.cfg = {}
        for name, default in DEFAULTS.items():
            self.declare_parameter(name, default)
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            self.cfg[name] = value
        self.course = None
        self.reset(None)
        self.create_subscription(NavSatFix, '/hk/sensors/gps/fix', self.on_gps, qos_profile_sensor_data)
        self.create_subscription(Imu, '/hk/sensors/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, '/hk/sensors/wheel_encoder/speed', self.on_wheel, qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, '/hk/local/odom', 10)
        self.tf = TransformBroadcaster(self)
        SimEvents(self, self.on_reset, self.on_end)
        self.create_timer(1.0 / RATE_HZ, self.publish)

    def on_reset(self, checkpoint, reload):
        if reload or self.course is None:
            self.course = load_course(self)
        self.reset(self.course.pose(checkpoint))

    def on_end(self):
        if self.ready:
            self.get_logger().info(f'위치 추정 통계: {dict(self.filter.counts)}, 수신: {dict(self.received)}')
        self.ready = False
        self.queue.clear()

    def reset(self, pose):
        self.ready = pose is not None
        self.reset_time = self.get_clock().now().nanoseconds * 1e-9
        self.last_clock = self.reset_time
        self.filter = PoseFilter(pose or (0.0, 0.0, 0.0), self.reset_time, self.cfg)
        self.queue, self.last_stamp = [], {}
        self.sequence = 0
        self.received = Counter()
        self.x, self.y, self.yaw = pose or (0.0, 0.0, 0.0)
        self.speed = 0.0

    def enqueue(self, kind, message, values):
        if not self.ready:
            return
        timestamp = stamp(message)
        now = self.get_clock().now().nanoseconds * 1e-9
        if (timestamp < self.reset_time - 1e-6 or timestamp > now + 2.0
                or timestamp <= self.last_stamp.get(kind, -math.inf)):
            self.received['ignored'] += 1
            return
        if not all(math.isfinite(value) for value in values):
            self.received['invalid'] += 1
            return
        self.last_stamp[kind] = timestamp
        self.sequence += 1
        heapq.heappush(self.queue, (timestamp, self.sequence, kind, values))
        self.received[kind] += 1

    def on_wheel(self, message):
        self.enqueue('wheel', message, (message.twist.linear.x,))

    def on_imu(self, message):
        self.enqueue('imu', message, (message.angular_velocity.z,))

    def on_gps(self, message):
        if not self.ready or message.status.status < 0:
            return
        if (not math.isfinite(message.latitude) or not math.isfinite(message.longitude)
                or not -90 <= message.latitude <= 90 or not -180 <= message.longitude <= 180):
            self.received['invalid'] += 1
            return
        x, y = ll_to_map(message.latitude, message.longitude, self.course.origin)
        self.enqueue('gps', message, (x, y))

    def publish(self):
        if not self.ready:
            return
        clock = self.get_clock().now()
        now = clock.nanoseconds * 1e-9
        if now < self.last_clock - 1e-6:
            self.ready = False
            self.queue.clear()
            self.get_logger().warning('시뮬레이션 시각이 되돌아갔습니다. START/RESPAWN을 기다립니다.')
            return
        self.last_clock = now
        cutoff = max(self.reset_time, now - self.cfg['fusion_delay'])
        while self.queue and self.queue[0][0] <= cutoff:
            timestamp, _, kind, values = heapq.heappop(self.queue)
            getattr(self.filter, kind)(timestamp, *values)
        self.filter.advance(cutoff)
        state, covariance = self.filter.estimate(now)
        self.x, self.y, self.yaw = map(float, state[:3])
        self.speed, measured_rate = self.filter.motion(now)
        angular_rate = measured_rate - state[3] if measured_rate is not None and abs(self.speed) >= 0.04 else 0.0
        message = Odometry()
        message.header.stamp = clock.to_msg()
        message.header.frame_id = 'map'
        message.child_frame_id = 'base_link'
        message.pose.pose.position.x, message.pose.pose.position.y = self.x, self.y
        message.pose.pose.orientation.z = math.sin(self.yaw / 2)
        message.pose.pose.orientation.w = math.cos(self.yaw / 2)
        message.twist.twist.linear.x = float(self.speed)
        message.twist.twist.angular.z = float(angular_rate)
        axes = (0, 1, 5)
        for i, row in enumerate(axes):
            for j, column in enumerate(axes):
                message.pose.covariance[row * 6 + column] = float(covariance[i, j])
        for axis in (2, 3, 4):
            message.pose.covariance[axis * 6 + axis] = 1e6
        message.twist.covariance[0] = 0.15**2
        message.twist.covariance[35] = float(covariance[3, 3] + 0.01**2)
        self.pub.publish(message)
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.rotation = message.pose.pose.orientation
        self.tf.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
