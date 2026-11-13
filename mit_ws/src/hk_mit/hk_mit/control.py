"""HEVEN 2026 MIT 기본 코드 3/3: 제어 (control). 담당: 기본 주행·제어.

localization.py의 /hk/local/odom(위치·방향·속력)과 planner.py의 /hk/plan/local_path(따라갈 경로)를 받아
50 Hz로 /hk/control/command에 구동·제동·조향을 냅니다. 고칠 곳은 아래 `drive()`입니다.

  drive()에서 쓸 수 있는 값                       돌려줄 값
  self.x, self.y   지도 위치 (m, 뒤차축)            accel  0~1   (구동)
  self.yaw         진행 방향 (rad)                 brake  0~1   (제동)
  self.speed       속력 (m/s)                      steer  -1~1  (+ 왼쪽, - 오른쪽, 1 = 35.4°)
  self.path        따라갈 경로 [(x, y, 경로 방향), ...] (비었으면 경로 끝)
  self.nearest(x, y)  경로에서 (x, y)와 가장 가까운 점 (x, y, 경로 방향)

조향은 Stanley, 속력은 PI 제어입니다. 일부러 아주 단순하게 두었습니다: 굽은 길에서 속도를 줄이지 않고,
노면·지연·경사도 따로 생각하지 않습니다. 속도를 올리거나 옵션을 켜면서 직접 고쳐 보세요.

속도 프로파일 (profile_enabled: true, 기본)
  local_path 점들의 곡률로 구간별 속도 한계 v = sqrt(a_lat_max / |곡률|)을 구하고(v_min ~ v_max로 제한),
  뒤에서 앞으로 a_decel 한계의 감속을 전파해 "코너 진입 전에 미리 감속"하는 목표 속도 프로파일을 만듭니다.
  직선은 v_max까지 가속합니다. 조향은 지금까지처럼 Stanley입니다.
  코너 앞에서는 프로파일이 요구하는 감속을 brake_ff_decel로 환산한 앞먹임 제동을 PI에 더합니다.

지연 보상 (steer_delay_comp, speed_lead_s)
  조향: 이미 보낸 조향 명령(steer_hist)과 Bicycle Kinematic Model(축거 3.0 m)로 steer_delay_comp초 뒤의 차 위치·방향을
  예측하고, 그 자세를 기준으로 Stanley 조향각을 계산합니다. 시뮬레이터의 조향 지연 옵션을 켜면 0.25, 끄면 0.0으로 두세요
  (끈 상태에서 0.25를 쓰면 오히려 흔들립니다).
  구동·제동: speed_lead_s초 앞의 목표 속도를 쓰고, a_decel을 낮추고 brake_ff_lookahead_s를 늘려 감속을 일찍 시작합니다.
  주의: bringup.launch.py의 target_speed 인자(기본 5.0)는 프로파일을 쓰는 동안 무시됩니다. 최고 속도는 yaml의 v_max입니다.
  profile_enabled: false면 예전처럼 target_speed 하나로 달립니다.

미션 직접 제어 (override): planner.py의 미션 스위치가 /hk/mission/override_cmd(CtrlCmd)를 내는 동안에는
drive() 대신 그 명령(구동·제동·조향·기어)을 그대로 보냅니다. override_timeout_s(기본 0.2 s) 동안 새 명령이 없으면
평소 제어(Stanley + PI)로 돌아가며, 이때 PI 적분값을 비웁니다. 미션이 아무것도 내지 않으면 원래와 똑같이 동작합니다.
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry, Path
from hk_interfaces.msg import CtrlCmd

from .sim_events import SimEvents

# 조정할 값은 config/bringup.yaml의 파라미터입니다(공식 주행 직전에는 코드가 아니라 이 파일만 바꿀 수 있음).
# 아래는 yaml에 값이 없을 때 쓰는 기본값입니다.
TARGET_SPEED_DEFAULT = 5.0
PARAMS = {
    'target_speed': TARGET_SPEED_DEFAULT,   # 목표 속력 (m/s). 실행할 때 target_speed:=… 로도 바꿀 수 있습니다.
    'k_stanley': 1.0,      # 경로에서 옆으로 벗어난 거리에 대한 조향 세기
    'k_soft': 1.0,         # 저속에서 조향이 너무 세지지 않게 하는 값 (m/s)
    'kp': 0.8,             # 속력 P 이득
    'ki': 0.08,            # 속력 I 이득
    'brake_gain': 0.6,     # 너무 빠를 때 제동 세기
    'profile_enabled': 1.0,   # 1: 곡률 기반 속도 프로파일, 0: target_speed 하나로 (예전 방식)
    'v_max': 12.0,         # 직선 최고 속도 (m/s)
    'v_min': 4.5,          # 급커브에서도 이 속도 밑으로는 낮추지 않음 (m/s)
    'a_lat_max': 3.2,      # 코너에서 허용할 횡가속도 (m/s²)
    'a_decel': 2.6,        # 코너 진입 전에 미리 감속하는 세기 (m/s², 클수록 늦게 감속)
    'curv_window_m': 3.0,  # 곡률을 잴 때 앞뒤로 보는 길이 (m). 노이즈를 줄임
    'brake_ff_decel': 4.0,   # 제동 앞먹임: brake=1일 때의 감속(m/s²) 추정. 크게 잡으면 앞먹임 제동이 약해짐 (0이면 끔)
    'brake_ff_lookahead_s': 0.8,  # 앞먹임 제동이 내다보는 시간 (s)
    'steer_delay_comp': 0.25,   # 조향 지연 보상 시간 (s). 시뮬레이터 조향 지연 옵션이 켜지면 0.25, 꺼지면 0.0
    'speed_lead_s': 0.25,  # 구동 지연 보상: 이 시간만큼 앞의 목표 속도를 씀 (s)
    'override_timeout_s': 0.2,   # 미션 직접 제어 명령이 이 시간 넘게 안 오면 평소 제어로 복귀 (s)
}
OVERRIDE_CMD_TOPIC = '/hk/mission/override_cmd'   # planner.py의 미션 스위치가 냄
WHEELBASE = 3.0        # 축간 거리 (m)
MAX_STEER = math.radians(35.4)   # steer = 1 일 때 앞바퀴 각도
RATE_HZ = 50.0
PREDICT_DT = 0.02      # 지연 보상 예측의 계산 간격 (s)
SETTLE_S = 0.5         # 시작·복귀 뒤 이 시간 동안은 제동만 (시뮬레이터도 복귀 뒤 0.5초 명령을 무시함)


class Control(Node):
    def __init__(self):
        super().__init__('hk_mit_control')
        for name, default in PARAMS.items():           # self.target_speed, self.k_stanley, ... 로 씁니다
            self.declare_parameter(name, default)
            setattr(self, name, float(self.get_parameter(name).value))
        self.dt = 1.0 / RATE_HZ
        self.odom = None
        self.path = None
        self.active = False
        self.reset_time = None
        self.override = None                         # (CtrlCmd, 받은 시각)
        self.override_active = False
        self._curv_src = self._curv = self._s = self._prof = None
        self.steer_hist = deque()                    # (보낸 시각, 앞바퀴 각도 rad): 아직 차에 닿지 않은 조향 명령 포함
        self.cmd_pub = self.create_publisher(CtrlCmd, '/hk/control/command', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Odometry, '/hk/local/odom', self.on_odom, 10)
        self.create_subscription(Path, '/hk/plan/local_path', self.on_path, 10)
        self.create_subscription(CtrlCmd, OVERRIDE_CMD_TOPIC, self.on_override,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        SimEvents(self, self.on_reset, self.on_end)
        self.create_timer(self.dt, self.tick)

    # ---- 시작·복귀 -----------------------------------------------------------------------------------
    def on_reset(self, checkpoint, reload):
        self.reset_control()
        self.odom = self.path = None
        self.override = None
        self.override_active = False
        self.steer_hist.clear()
        self.reset_time = self.now()
        self.active = True

    def on_end(self):
        self.active = False
        self.send(0.0, 1.0, 0.0)

    def reset_control(self):
        """시작·복귀 때 불립니다. 제어기 상태를 처음으로 돌립니다."""
        self.speed_error_sum = 0.0

    # ---- 받은 값 -------------------------------------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, m):
        self.odom = m
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.speed = m.twist.twist.linear.x

    def on_path(self, m):
        self.path = [(ps.pose.position.x, ps.pose.position.y,
                      2 * math.atan2(ps.pose.orientation.z, ps.pose.orientation.w)) for ps in m.poses]

    def on_override(self, m):
        self.override = (m, self.now())

    def nearest(self, x, y):
        return min(self.path, key=lambda p: math.hypot(p[0] - x, p[1] - y))

    # ---- 50 Hz ------------------------------------------------------------------------------------------
    def tick(self):
        if not self.active:
            return
        if self.odom is None or self.path is None or self.now() - self.reset_time < SETTLE_S:
            self.send(0.0, 1.0, 0.0)                 # 시작·복귀 직후, 위치·경로가 아직 없으면 정지
            return
        if self.override is not None and self.now() - self.override[1] <= self.override_timeout_s:
            m = self.override[0]                     # 미션 직접 제어가 우선
            self.override_active = True
            self.send(m.accel, m.brake, m.steering, m.gear)
            return
        if self.override_active:                     # 직접 제어가 끝남: PI를 새로 시작
            self.override_active = False
            self.reset_control()
        self.send(*self.drive())

    def drive(self):
        if not self.path:                            # 경로 끝을 지남: 곧게 서서 멈춤
            steer, target = 0.0, 0.0
        else:
            steer = self.steering()
            target = self.speed_target()

        # 3) 속력: PI 제어
        error = target - self.speed
        brake_ff = self.brake_feedforward() if self.path and error < 0.3 else 0.0
        self.speed_error_sum = clamp(self.speed_error_sum + error * self.dt, -5.0, 5.0)
        u = self.kp * error + self.ki * self.speed_error_sum
        if brake_ff > 0.0:                          # 곧 속도를 줄여야 하는 구간: 앞먹임 제동 + 오차 보정
            return 0.0, min(brake_ff + max(-u, 0.0) * self.brake_gain, 1.0), steer
        if u >= 0:
            return min(u, 1.0), 0.0, steer          # accel, brake, steer
        return 0.0, min(-u * self.brake_gain, 1.0), steer

    def steering(self):
        """Stanley(방향 오차 + 옆 오차). 반환: steer -1~1.
        steer_delay_comp초 뒤의 예측 자세(predict_pose)를 기준으로 계산합니다. 지금 내는 조향은 그때 차에 닿기 때문입니다.
        (앞바퀴 축 기준 Stanley는 굽은 길의 곡률 앞먹임을 이미 포함하므로 따로 더하지 않습니다.)"""
        x, y, yaw = self.predict_pose()
        fx = x + WHEELBASE * math.cos(yaw)                           # 앞바퀴 축 위치
        fy = y + WHEELBASE * math.sin(yaw)
        i = self.nearest_index(fx, fy)
        px, py, path_yaw = self.path[i]
        heading_error = angle_diff(path_yaw, yaw)
        cross_error = -math.sin(path_yaw) * (px - fx) + math.cos(path_yaw) * (py - fy)   # 경로가 왼쪽이면 +
        wheel_angle = heading_error + math.atan2(self.k_stanley * cross_error, self.k_soft + self.speed)
        return clamp(wheel_angle / MAX_STEER, -1.0, 1.0)

    def predict_pose(self):
        """Bicycle Kinematic Model로 steer_delay_comp초 뒤의 뒤차축 위치·방향 (x, y, yaw)를 예측합니다.
        속력은 지금 값으로 일정하다고 보고, 그 사이 차에 닿는 조향은 이미 보낸 명령(steer_hist)에서 가져옵니다
        (시각 t의 조향각 = t - steer_delay_comp에 보낸 명령). 보상 시간이 0이면 지금 자세 그대로입니다."""
        T = self.steer_delay_comp
        x, y, yaw = self.x, self.y, self.yaw
        if T <= 0.0:
            return x, y, yaw
        steps = max(1, int(round(T / PREDICT_DT)))
        dt = T / steps
        t0 = self.now() - T                          # 예측 구간의 첫 순간에 차에 닿는 명령을 보낸 시각
        hist = self.steer_hist
        j, delta = 0, 0.0
        for k in range(steps):
            tk = t0 + k * dt
            while j < len(hist) and hist[j][0] <= tk:
                delta = hist[j][1]
                j += 1
            x += self.speed * math.cos(yaw) * dt
            y += self.speed * math.sin(yaw) * dt
            yaw += self.speed / WHEELBASE * math.tan(delta) * dt
        return x, y, yaw

    def note_steer(self, steer):
        """보낸 조향 명령을 기록합니다 (예측에 씀)."""
        t = self.now()
        self.steer_hist.append((t, clamp(steer, -1.0, 1.0) * MAX_STEER))
        while self.steer_hist and self.steer_hist[0][0] < t - self.steer_delay_comp - 0.5:
            self.steer_hist.popleft()

    def nearest_index(self, x, y):
        return min(range(len(self.path)), key=lambda i: math.hypot(self.path[i][0] - x, self.path[i][1] - y))

    def curvature(self):
        """경로 점마다의 부호 있는 곡률 (1/m, + 왼쪽). 앞뒤 curv_window_m의 방향 변화로 구해 노이즈에 강합니다."""
        path = self.path
        if self._curv_src is path:
            return self._curv
        n = len(path)
        s = [0.0]
        for a, b in zip(path, path[1:]):
            s.append(s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        k = [0.0] * n
        j0 = j1 = 0
        for i in range(n):
            while j0 < i and s[i] - s[j0] > self.curv_window_m:
                j0 += 1
            while j1 < n - 1 and s[j1 + 1] - s[i] <= self.curv_window_m:
                j1 += 1
            if s[j1] - s[j0] > 1e-6:
                k[i] = angle_diff(path[j1][2], path[j0][2]) / (s[j1] - s[j0])
        self._curv_src, self._curv, self._s = path, k, s
        return k

    def brake_feedforward(self):
        """앞으로 brake_ff_lookahead_s 동안 프로파일이 요구하는 감속을 brake(0~1)로 바꾼 앞먹임. 줄일 필요가 없으면 0."""
        if not self.profile_enabled or self.brake_ff_decel <= 0.0 or self._prof is None:
            return 0.0
        v, s = self._prof, self._s
        i = self.nearest_index(self.x, self.y)
        j = min(len(v) - 1, i + max(1, int(self.speed * self.brake_ff_lookahead_s / 0.5)))
        if j <= i or s[j] - s[i] < 1e-6:
            return 0.0
        a_req = (v[i] ** 2 - v[j] ** 2) / (2.0 * (s[j] - s[i]))   # 프로파일을 그대로 따라가는 데 필요한 감속
        return clamp(a_req / self.brake_ff_decel, 0.0, 1.0)

    def speed_target(self):
        """목표 속력 (m/s). 곡률 한계 속도에 코너 앞 감속(뒤에서 앞으로 전파)을 더한 프로파일에서 지금 위치의 값."""
        if not self.profile_enabled:
            return self.target_speed
        k = self.curvature()
        s = self._s
        n = len(k)
        v = [clamp(math.sqrt(self.a_lat_max / max(abs(c), 1e-6)), self.v_min, self.v_max) for c in k]
        for i in range(n - 2, -1, -1):               # 앞쪽 점의 속도에서 a_decel로 줄일 수 있는 속도까지만 허용
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * self.a_decel * (s[i + 1] - s[i])))
        self._prof = v
        i = self.nearest_index(self.x, self.y)
        last = min(n - 1, i + 1 + int(self.speed * self.speed_lead_s / 0.5))
        return min(v[i:last + 1])

    def send(self, accel, brake, steer, gear=CtrlCmd.GEAR_DRIVE):
        self.note_steer(float(steer))
        c = CtrlCmd()
        c.header.stamp = self.get_clock().now().to_msg()
        c.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        c.gear = int(gear)
        c.long_cmd_type = CtrlCmd.LONG_CMD_THROTTLE
        c.accel, c.brake, c.steering = float(accel), float(brake), float(steer)
        self.cmd_pub.publish(c)


def angle_diff(a, b):
    """a - b 를 -π~π 로."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main(args=None):
    rclpy.init(args=args)
    node = Control()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
