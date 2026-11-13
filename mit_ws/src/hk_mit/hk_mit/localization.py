"""MIT 위치 추정: 바퀴·IMU 예측 + GNSS 보정, 50 Hz odom/TF.

(x, y, yaw, 자이로 편향, GNSS의 느린 x/y 오차)를 추정하는 EKF.
속도·회전·가감속에 따라 위치 예측 신뢰도를 조절하고 GNSS의 빠른 잡음과 느린 치우침을 구분한다.
GNSS 음영에서는 바퀴·IMU로 이동을 예측한다. 정지 중 자이로 영점, 완만한 구간의 궤적 방향을 보정한다.
경로에 위치를 강제로 맞추지 않으므로 회피·후진에도 사용할 수 있다.
이 파일만 교체하면 된다. numpy는 대회 Docker 이미지에 포함되어 있다.
조정값은 DEFAULTS 및 같은 이름의 ROS 파라미터에 있다.
"""
import heapq
import math
from collections import Counter, deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan, NavSatFix
from tf2_ros import TransformBroadcaster
from .course import COURSE_DIR, declare_file_params, load_course, load_route, load_runtime_config, option_on
from .sim_events import SimEvents
from .utm import ll_to_map

RATE_HZ = 50.0
DEFAULTS = {
    'gps_sigma': 0.60,              # GNSS의 빠른 위치 잡음 표준편차 가정 (m)
    'gps_bias_sigma': 0.20,         # 천천히 변하는 GNSS 공통 오차의 표준편차 가정 (m)
    'gps_bias_tau': 10.0,           # GNSS 공통 오차의 시간 상관 길이 (s)
    'gps_update_period': 0.15,      # 상관된 GPS를 과신하지 않도록 보정 간격 제한 (s)
    'gps_gate': 25.0,               # 정규화한 2차원 위치 잔차 제곱의 상한
    'max_gps_step': 0.25,           # GNSS 보정 한 번의 최대 위치 이동 (m)
    'position_noise': 0.035,        # 이동 모델 위치 불확실성 (m/sqrt(s))
    'gyro_noise': 0.004,            # 방향 예측 불확실성 (rad/sqrt(s))
    'bias_walk': 0.00012,           # IMU 편향 변화 불확실성 (rad/s/sqrt(s))
    'stationary_gyro_sigma': 0.008, # 정지 중 자이로 영점 측정 표준편차 (rad/s)
    'sensor_timeout': 0.30,         # 오래된 속도·각속도는 계속 적분하지 않음 (s)
    'fusion_delay': 0.04,          # 토픽 사이 수신 순서를 정렬할 여유 (s)
    'speed_noise_gain': 0.025,     # 속도가 커질 때 위치 예측 불확실성 증가
    'turn_noise_gain': 0.15,       # 회전 중 횡방향 모델 오차를 더 허용
    'accel_noise_gain': 0.02,      # 가감속 중 바퀴 기반 예측 불확실성 증가
    'heading_window': 5.0,        # GNSS 궤적과 추측항법을 비교할 시간 (s)
    'heading_distance': 8.0,      # 방향 비교에 필요한 최소 이동 거리 (m)
    'heading_period': 2.5,        # 겹치는 GPS 궤적을 연속으로 과신하지 않도록 간격 제한 (s)
    'heading_sigma': 0.04,        # 궤적 방향 측정의 최소 표준편차 (rad)
    'heading_max_step': 0.003,    # 방향 보정 한 번의 상한 (rad)
    'heading_max_turn': 0.35,     # 방향 맞춤은 완만한 구간에서만 사용 (rad)
    # --- LiDAR 통로 벽 보정: 양쪽 벽 중앙 = 전역 경로(통로 중심선)로 옆 방향 위치를 맞춤 ---
    'wall_sigma': 0.10,           # 벽으로 잰 옆 방향 위치 측정 표준편차 (m). 기록 재생에서 오차 p90 0.05 m
    'wall_gate': 1.2,             # 이보다 큰 옆 방향 어긋남은 오검출로 보고 버림 (m)
    'wall_width_tol': 0.30,       # 잰 통로 폭이 설정 통로 폭과 이만큼 이내일 때만 사용 (m)
    'wall_max_step': 0.20,        # 한 번 보정의 최대 위치 이동 (m)
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
        # 차량 상태 4개 + GNSS의 천천히 변하는 x/y 오차 2개.
        self.state = np.array([*pose, 0.0, 0.0, 0.0], dtype=float)
        self.cov = np.diag([0.08**2, 0.08**2, math.radians(0.5)**2, 0.005**2,
                            self.cfg['gps_bias_sigma']**2, self.cfg['gps_bias_sigma']**2])
        self.time = float(timestamp)
        self.speed = self.rate = 0.0
        self.wheel_time = self.imu_time = None
        self.still_since = self.last_gps_update = None
        self.acceleration = 0.0
        self.dead_pose = np.array([0.0, 0.0, pose[2]], dtype=float)
        self.gps_history = deque()
        self.last_heading_update = None
        self.diagnostics = {}
        self.counts = Counter()

    def motion(self, timestamp):
        timeout = self.cfg['sensor_timeout']
        speed = self.speed if self.wheel_time is not None and 0 <= timestamp - self.wheel_time <= timeout else 0.0
        rate = self.rate if self.imu_time is not None and 0 <= timestamp - self.imu_time <= timeout else None
        return speed, rate

    def _predict(self, state, cov, begin, end, track_motion=False):
        remaining = end - begin
        if remaining <= 0:
            return state, cov
        if remaining > 1.0:
            # 센서 전체의 긴 공백을 오래된 입력으로 채우지 않는다.
            state = state.copy()
            decay = math.exp(-remaining / self.cfg['gps_bias_tau'])
            transition = np.diag([1., 1., 1., 1., decay, decay])
            state[4:] *= decay
            bias_noise = self.cfg['gps_bias_sigma']**2 * (1.0-decay**2)
            noise = np.diag([4.0, 4.0, 0.03, 0.00001, 0., 0.]) * min(remaining, 30.0)
            noise[4, 4] = noise[5, 5] = bias_noise
            if track_motion:
                self.gps_history.clear()
                self.dead_pose = np.array([0., 0., state[2]])
            return state, transition @ cov @ transition.T + noise
        while remaining > 1e-9:
            dt = min(remaining, 0.05)
            speed, measured_rate = self.motion(begin + dt * 0.5)
            moving = abs(speed) >= 0.04
            omega = measured_rate - state[3] if measured_rate is not None and moving else 0.0
            mid_yaw = state[2] + omega * dt * 0.5
            distance = speed * dt
            s, c = math.sin(mid_yaw), math.cos(mid_yaw)
            transition = np.eye(6)
            gps_bias_decay = math.exp(-dt / self.cfg['gps_bias_tau'])
            transition[4, 4] = transition[5, 5] = gps_bias_decay
            transition[0, 2], transition[1, 2] = -distance * s, distance * c
            if measured_rate is not None and moving:
                transition[0, 3] = 0.5 * distance * dt * s
                transition[1, 3] = -0.5 * distance * dt * c
                transition[2, 3] = -dt
            state = state.copy()
            state[0] += distance * c
            state[1] += distance * s
            state[2] = wrap(state[2] + omega * dt)
            state[4:] *= gps_bias_decay
            gps_bias_noise = self.cfg['gps_bias_sigma']**2 * (1.0 - gps_bias_decay**2)
            noise = np.diag([self.cfg['position_noise']**2 * dt,
                             self.cfg['position_noise']**2 * dt,
                             self.cfg['gyro_noise']**2 * dt,
                             self.cfg['bias_walk']**2 * dt, gps_bias_noise, gps_bias_noise])
            # 가속·회전 때 바퀴 적분을 과신하지 않는다. 전진·후진 모두 동일한 규칙.
            forward_sigma = self.cfg['speed_noise_gain'] * abs(speed) + self.cfg['accel_noise_gain'] * abs(self.acceleration)
            lateral_sigma = self.cfg['turn_noise_gain'] * abs(speed * omega)
            forward_axis, lateral_axis = np.array([c, s]), np.array([-s, c])
            noise[:2, :2] += dt * (forward_sigma**2 * np.outer(forward_axis, forward_axis)
                                   + lateral_sigma**2 * np.outer(lateral_axis, lateral_axis))
            cov = transition @ cov @ transition.T + noise
            if track_motion:
                self.diagnostics['forward_model_sigma'] = forward_sigma
                self.diagnostics['lateral_model_sigma'] = lateral_sigma
                dead_mid = self.dead_pose[2] + omega * dt * 0.5
                self.dead_pose[:2] += distance * np.array([math.cos(dead_mid), math.sin(dead_mid)])
                self.dead_pose[2] = wrap(self.dead_pose[2] + omega * dt)
            begin += dt
            remaining = end - begin
        return state, (cov + cov.T) * 0.5

    def advance(self, timestamp):
        if timestamp < self.time - 1e-6:
            self.counts['old_measurement'] += 1
            return False
        self.state, self.cov = self._predict(self.state, self.cov, self.time, timestamp, track_motion=True)
        self.time = max(self.time, timestamp)
        return True

    def wheel(self, timestamp, speed):
        if not math.isfinite(speed) or not self.advance(timestamp):
            return
        if self.wheel_time is not None:
            dt = timestamp - self.wheel_time
            if 1e-4 < dt <= self.cfg['sensor_timeout']:
                raw_acceleration = float(np.clip((speed - self.speed) / dt, -8.0, 8.0))
                self.acceleration += (1.0 - math.exp(-dt / 0.25)) * (raw_acceleration - self.acceleration)
            elif dt > self.cfg['sensor_timeout']:
                self.acceleration = 0.0
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
            residual = np.eye(6)
            residual[:, 3] -= gain
            self.cov = residual @ self.cov @ residual.T + np.outer(gain, gain) * variance
            self._normalize()
            self.counts['stationary_bias_update'] += 1

    def gps(self, timestamp, x, y):
        if not all(math.isfinite(value) for value in (x, y)) or not self.advance(timestamp):
            return
        measurement = np.array([[1., 0., 0., 0., 1., 0.], [0., 1., 0., 0., 0., 1.]])
        innovation = np.array([x, y]) - measurement @ self.state
        measurement_cov = np.eye(2) * self.cfg['gps_sigma']**2
        innovation_cov = measurement @ self.cov @ measurement.T + measurement_cov
        score = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        if score > self.cfg['gps_gate']:
            self.counts['gps_rejected'] += 1
            return
        self.diagnostics['gps_residual_m'] = float(np.linalg.norm(innovation))
        self.diagnostics['gps_nis'] = score
        self._trajectory_heading(timestamp, x, y)
        if self.last_gps_update is not None and timestamp - self.last_gps_update < self.cfg['gps_update_period'] - 1e-6:
            return
        # 궤적 방향 보정으로 공분산이 바뀔 수 있으므로 위치 이득은 다시 계산한다.
        innovation = np.array([x, y]) - measurement @ self.state
        innovation_cov = measurement @ self.cov @ measurement.T + measurement_cov
        if score > 9.0:
            measurement_cov *= score / 9.0
            innovation_cov = measurement @ self.cov @ measurement.T + measurement_cov
        gain = np.linalg.solve(innovation_cov, measurement @ self.cov).T
        speed, _ = self.motion(timestamp)
        if abs(speed) < 0.3:
            gain[2:4, :] = 0.0  # 정지 중 위치 잡음으로 방향·자이로 편향을 바꾸지 않음
        if (self.still_since is not None and timestamp - self.still_since >= 0.25
                and self.wheel_time is not None and timestamp-self.wheel_time < 0.1
                and abs(speed) < 0.04):
            # A stopped car must not wander in the parking slot as GPS jitters.
            # Keep map pose fixed; the slow GNSS bias can still be estimated.
            gain[:4, :] = 0.0
        delta = gain @ innovation
        scale = min(1.0, self.cfg['max_gps_step'] / max(np.linalg.norm(delta[:2]), 1e-12),
                    math.radians(1.0) / max(abs(delta[2]), 1e-12))
        gain *= scale
        self.state += gain @ innovation
        residual = np.eye(6) - gain @ measurement
        self.cov = residual @ self.cov @ residual.T + gain @ measurement_cov @ gain.T
        self.last_gps_update = timestamp
        self._normalize()
        self.counts['gps_update'] += 1

    def wall(self, timestamp, nx, ny, offset):
        """통로 양쪽 벽으로 잰 옆 방향 위치 오차 offset = n·(추정 위치 - 실제 위치) (n = 경로 법선). GNSS 편향과 무관.
        LiDAR는 지연되어 도착하므로 필터 시각을 되돌리지 않고 현재 상태에 적용한다(오차는 천천히 변함)."""
        if timestamp > self.time:                # 더 새 측정이면 그 시각까지 예측, 오래된 측정이면 현재 상태 그대로
            self.advance(timestamp)
        measurement = np.array([[nx, ny, 0., 0., 0., 0.]])
        innovation = np.array([-offset])
        if abs(innovation[0]) > self.cfg['wall_gate']:
            self.counts['wall_rejected'] += 1
            return
        measurement_cov = np.array([[self.cfg['wall_sigma']**2]])
        innovation_cov = measurement @ self.cov @ measurement.T + measurement_cov
        gain = np.linalg.solve(innovation_cov, measurement @ self.cov).T
        gain[2:4, :] = 0.0                       # 옆 방향 위치만 고침 (방향·자이로 편향은 건드리지 않음)
        delta = gain @ innovation
        scale = min(1.0, self.cfg['wall_max_step'] / max(np.linalg.norm(delta[:2]), 1e-12))
        gain *= scale
        self.state += gain @ innovation
        residual = np.eye(6) - gain @ measurement
        self.cov = residual @ self.cov @ residual.T + gain @ measurement_cov @ gain.T
        self._normalize()
        self.diagnostics['wall_innovation_m'] = float(innovation[0])
        self.counts['wall_update'] += 1

    def _trajectory_heading(self, timestamp, x, y):
        """한 점의 GPS 방향 대신 여러 점의 궤적과 독립적인 추측항법 궤적을 비교한다."""
        self.gps_history.append((timestamp, x, y, *self.dead_pose))
        while self.gps_history and timestamp - self.gps_history[0][0] > self.cfg['heading_window']:
            self.gps_history.popleft()
        if (len(self.gps_history) < 12 or abs(self.speed) < 0.5
                or (self.last_heading_update is not None
                    and timestamp - self.last_heading_update < self.cfg['heading_period'])):
            return
        rows = np.asarray(self.gps_history)
        if np.ptp(np.unwrap(rows[:, 5])) > self.cfg['heading_max_turn']:
            return
        dead, gps = rows[:, 3:5], rows[:, 1:3]
        distance = float(np.linalg.norm(dead[-1] - dead[0]))
        if distance < self.cfg['heading_distance']:
            return
        # 가중 2차원 회전 맞춤. 큰 잔차를 가진 점의 영향을 줄인다.
        weights = np.ones(len(rows))
        for _ in range(3):
            mean_dead = np.average(dead, axis=0, weights=weights)
            mean_gps = np.average(gps, axis=0, weights=weights)
            a, b = dead - mean_dead, gps - mean_gps
            dot = float(np.sum(weights * np.sum(a * b, axis=1)))
            cross = float(np.sum(weights * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))
            angle = math.atan2(cross, dot)
            c, s = math.cos(angle), math.sin(angle)
            rotated = a @ np.array([[c, s], [-s, c]])
            residuals = np.linalg.norm(b - rotated, axis=1)
            weights = np.minimum(1.0, 1.5 / np.maximum(residuals, 1e-9))
        rms = float(np.sqrt(np.average(residuals**2, weights=weights)))
        if rms > 1.5:
            self.counts['heading_rejected'] += 1
            return
        innovation = wrap(self.dead_pose[2] + angle - self.state[2])
        variance = max(self.cfg['heading_sigma']**2, (self.cfg['gps_sigma'] / distance)**2)
        if abs(innovation) > max(0.14, 3.0 * math.sqrt(variance)):
            self.counts['heading_rejected'] += 1
            return
        gain = self.cov[:, 2] / (self.cov[2, 2] + variance)
        gain[:2] = 0.0
        gain[4:] = 0.0
        delta = gain * innovation
        scale = min(1.0, self.cfg['heading_max_step'] / max(abs(delta[2]), 1e-12),
                    0.0002 / max(abs(delta[3]), 1e-12))
        gain *= scale
        self.state += gain * innovation
        residual = np.eye(6)
        residual[:, 2] -= gain
        self.cov = residual @ self.cov @ residual.T + np.outer(gain, gain) * variance
        self._normalize()
        self.last_heading_update = timestamp
        self.diagnostics['heading_residual_deg'] = math.degrees(innovation)
        self.diagnostics['heading_fit_rms_m'] = rms
        self.counts['trajectory_heading_update'] += 1

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
        self.base_cfg = dict(self.cfg)
        self.declare_parameter('auto_sensor_noise', True)
        self.runtime_ready = False
        self.config_retry = -math.inf
        self.corridor_width = 4.392
        self.pose_history = deque(maxlen=60)          # (t, x, y, yaw) of published poses, for LiDAR scan time
        self._load_wall_geometry()
        self.reset(None)
        self.create_subscription(LaserScan, '/hk/sensors/lidar/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, '/hk/sensors/gps/fix', self.on_gps, qos_profile_sensor_data)
        self.create_subscription(Imu, '/hk/sensors/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, '/hk/sensors/wheel_encoder/speed', self.on_wheel, qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, '/hk/local/odom', 10)
        self.tf = TransformBroadcaster(self)
        SimEvents(self, self.on_reset, self.on_end)
        self.create_timer(1.0 / RATE_HZ, self.publish)

    def on_reset(self, checkpoint, reload):
        self.checkpoint = checkpoint
        if self.ready:
            self.log_diagnostics('RESPAWN/START')
        if reload or self.course is None:
            self.course = load_course(self)
            self.configure_noise()
        self.reset(self.course.pose(checkpoint))

    def configure_noise(self):
        self.config_retry = self.get_clock().now().nanoseconds * 1e-9
        try:
            runtime = load_runtime_config(self.get_parameter('course_json').value or None)
            self.cfg = dict(self.base_cfg)
            if self.get_parameter('auto_sensor_noise').value:
                options, sensors = runtime['options'], runtime.get('course', {}).get('sensors', {})
                if not option_on(options, 'gnss_noise'):
                    self.cfg.update(gps_sigma=0.04, gps_bias_sigma=0.02)
                else:
                    self.cfg['gps_bias_sigma'] = max(self.cfg['gps_bias_sigma'], float(sensors.get('gnss_multipath_sigma', 0.3)))
                if not option_on(options, 'imu_noise'):
                    self.cfg.update(gyro_noise=0.0002, bias_walk=0.00002)
            self.corridor_width = float(runtime.get('corridor_width_m', self.corridor_width))
            self.zone_widths = []
            for mission in runtime.get('course', {}).get('missions', []):
                zone = mission.get('zone') if str(mission.get('type', '')).startswith('obstacle') else None
                if zone:
                    xs, ys = [q[0] for q in zone], [q[1] for q in zone]
                    self.zone_widths.append(round(min(max(xs) - min(xs), max(ys) - min(ys)), 2))
            self.runtime_ready = True
            self.get_logger().info(f'위치 추정 벽 보정: 통로 폭 {self.corridor_width:.2f} m')
            self.get_logger().info(f'위치 추정 잡음 설정: gps_sigma={self.cfg["gps_sigma"]:.3f}, gps_bias_sigma={self.cfg["gps_bias_sigma"]:.3f}')
        except ValueError as error:
            self.runtime_ready = False
            self.get_logger().error(f'위치 추정 구성 확인 대기: {error}')

    def on_end(self):
        if self.ready:
            self.log_diagnostics('END')
        self.ready = False
        self.queue.clear()

    def log_diagnostics(self, reason):
        f = self.filter
        detail = dict(f.diagnostics)
        detail['gyro_bias_rad_s'] = float(f.state[3])
        detail['gps_bias_xy_m'] = f.state[4:].tolist()
        detail['position_sigma_m'] = np.sqrt(np.maximum(np.diag(f.cov)[:2], 0.0)).tolist()
        self.get_logger().info(f'위치 추정 {reason}: 처리={dict(f.counts)}, 수신={dict(self.received)}, 추정={detail}')

    def reset(self, pose):
        self.ready = pose is not None
        self.reset_time = self.get_clock().now().nanoseconds * 1e-9
        self.last_clock = self.reset_time
        self.filter = PoseFilter(pose or (0.0, 0.0, 0.0), self.reset_time, self.cfg)
        self.queue, self.last_stamp = [], {}
        self.pose_history.clear()
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

    def _load_wall_geometry(self):
        pts = np.asarray(load_route(self).points, dtype=float)
        seg = np.diff(pts, axis=0)
        self.route_pts = pts
        self.route_s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        tan = np.vstack([seg, seg[-1:]])
        tan[1:-1] = seg[1:] + seg[:-1]
        tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
        self.route_tan = tan
        self.route_nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
        self.lidar_x = 1.15
        try:
            import yaml
            with open(COURSE_DIR.parent / 'config' / 'sensor_mounts.yaml', encoding='utf-8') as f:
                self.lidar_x = float(yaml.safe_load(f)['lidar']['x'])
        except Exception:
            pass

    def wall_widths(self):
        return [self.corridor_width] + sorted(set(getattr(self, 'zone_widths', [])))

    def _pose_at(self, t):
        h = self.pose_history
        if not h or t < h[0][0] - 0.05 or t > h[-1][0] + 0.15:
            return None
        previous = h[0]
        for row in h:
            if row[0] >= t:
                span = row[0] - previous[0]
                k = 0.0 if span <= 1e-9 else (t - previous[0]) / span
                dh = math.atan2(math.sin(row[3] - previous[3]), math.cos(row[3] - previous[3]))
                return (previous[1] + k * (row[1] - previous[1]), previous[2] + k * (row[2] - previous[2]), previous[3] + k * dh)
            previous = row
        return h[-1][1:]

    def on_scan(self, message):
        """양쪽 통로 벽의 중앙 = 경로 중심선. 벽 중앙이 경로에서 e만큼 벗어나 보이면 내 위치가 e만큼 틀린 것."""
        if not self.ready or not self.runtime_ready:
            return
        t = stamp(message)
        pose = self._pose_at(t)
        if pose is None:
            return
        x, y, yaw = pose
        pts = self.route_pts
        win = slice(0, len(pts))
        ei = int(np.argmin(np.hypot(pts[win, 0] - x, pts[win, 1] - y)))
        r = np.asarray(message.ranges, dtype=float)
        ang = message.angle_min + np.arange(len(r)) * message.angle_increment
        ok = np.isfinite(r) & (r > max(message.range_min, 0.3)) & (r < 12.0)
        if ok.sum() < 20:
            return
        ox, oy = x + self.lidar_x * math.cos(yaw), y + self.lidar_x * math.sin(yaw)
        a = ang[ok] + yaw
        xy = np.stack([ox + r[ok] * np.cos(a), oy + r[ok] * np.sin(a)], axis=1)
        i0, i1 = max(0, ei - 20), min(len(pts), ei + 40)
        cand = pts[i0:i1]
        gi = ((xy[:, None, :] - cand[None, :, :]) ** 2).sum(-1).argmin(1) + i0
        rel = xy - pts[gi]
        s = self.route_s[gi] + (rel * self.route_tan[gi]).sum(1)
        d = (rel * self.route_nor[gi]).sum(1)
        s0 = self.route_s[ei]
        near = (s > s0 - 2.0) & (s < s0 + 8.0)
        med = None
        for width_expected in self.wall_widths():       # 좁은 통로 폭, 그다음 장애물 구간 폭(7.5 m)
            half = width_expected / 2.0
            found = []
            for side in (1, -1):
                m = near & (side * d > half - 0.9) & (side * d < half + 0.9)
                if m.sum() < 8 or np.ptp(s[m]) < 3.0:
                    break
                md = float(np.median(d[m]))
                if np.std(d[m] - md) > 0.15:
                    break
                found.append(md)
            if len(found) == 2 and abs(found[0] - found[1] - width_expected) <= self.cfg['wall_width_tol']:
                med = found
                break
        if med is None:
            return
        offset = 0.5 * (med[0] + med[1])             # = n·(내 위치 - 실제 위치)
        nx, ny = map(float, self.route_nor[ei])
        self.enqueue('wall', message, (nx, ny, offset))

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
        if not self.runtime_ready:
            if now - self.config_retry >= 1.0:
                self.configure_noise()
                if self.runtime_ready:
                    self.reset(self.course.pose(self.checkpoint))
            return
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
        self.pose_history.append((now, self.x, self.y, self.yaw))
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
        if any(t is None or now-t > self.cfg['sensor_timeout']
               for t in (self.filter.wheel_time, self.filter.imu_time)):
            # A fresh ROS header is not proof of fresh motion sensors.
            message.twist.covariance[0] = 1e6
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
