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

저마찰·경사·좁은 도로 대응 (course.json의 low_friction·slope zone 다각형을 읽어 그 안에서만 적용)
  저마찰 구간(μ=0.2 기준): 횡가속도 한계(a_lat_lowmu)로 급커브 속도를 2.3~2.7 m/s로 낮추고, 구역 앞 lowmu_lead_m부터
  a_decel_lowmu로 감속 계획을 세워 진입할 때 이미 느리게 합니다. 구동·제동 상한(accel_max_lowmu, brake_max_lowmu)으로
  바퀴가 미끄러지거나 잠기지 않게 합니다. 조향은 변화율 제한을 사실상 풀었고(steer_rate_*), 방향이 바뀔 때는 제한 없이
  즉시 따라가며, Stanley 이득을 저마찰용(k_stanley_lowmu, k_soft_lowmu, k_heading_lowmu)으로 따로 둡니다.
  경사 구간: 최고 속도 v_max_slope, 적분 이득 배수 slope_ki_mult. 속력 PI는 적분 누적 방지(anti-windup)와 구동/제동 전환
  히스테리시스(brake_on/brake_off)를 씁니다.
  좁은 도로: Stanley가 경로에 맞추는 점을 앞바퀴 축(3.0 m)보다 약간 뒤(track_point_m)로 잡아, 코너에서 차 앞 모서리는 바깥으로
  덜 나가고 뒤 모서리는 안쪽으로 덜 파고들게(두 값의 큰 쪽이 작아지게) 합니다.

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
import json
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry, Path
from hk_interfaces.msg import CtrlCmd

from .course import COURSE_DIR
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
    # --- 저마찰 구간 (course.json의 low_friction zone 안): 보수적 속도·부드러운 조작 ---
    'a_lat_lowmu': 0.95,   # 횡가속도 한계 (m/s²). μ=0.2 (한계 약 1.96)의 절반 이하. R=7 m 코너 -> 약 2.6 m/s
    'v_min_lowmu': 2.3,    # 이 구간의 최저 속도 (m/s)
    'v_max_lowmu': 5.0,    # 이 구간의 최고 속도 (m/s)
    'a_decel_lowmu': 1.0,  # 이 구간의 감속 계획 (m/s², μ=0.2의 제동 한계 ~1.8보다 작게)
    'accel_max_lowmu': 0.3,   # 이 구간의 구동 상한 (0~1)
    'brake_max_lowmu': 0.35,  # 이 구간의 제동 상한 (0~1). 바퀴 잠김 방지
    'steer_rate_lowmu': 6.0,  # 이 구간의 조향 변화율 한계 (1/s, 1 = 풀 조향). 사실상 제한 없음. 방향이 바뀔 때는 항상 즉시 추종
    # --- 경사 구간 (course.json의 slope zone 안) ---
    'v_max_slope': 8.0,    # 이 구간의 최고 속도 (m/s). 내리막 과속 방지
    'slope_ki_mult': 2.0,  # 이 구간에서 적분 이득 배수 (오르막 처짐·내리막 과속을 빨리 보정)
    # --- 공통 ---
    'steer_rate_max': 8.0,    # 조향 변화율 한계 (1/s). 사실상 제한 없음. 방향이 바뀔 때는 항상 즉시 추종
    'brake_on': 0.10,      # 구동 -> 제동으로 바꾸는 문턱 (u가 -brake_on보다 작을 때)
    'brake_off': 0.03,     # 제동 -> 구동으로 되돌리는 문턱 (u가 -brake_off보다 클 때). 떨림 방지
    'lowmu_enabled': 1.0,  # 1: 저마찰 구역 대응 켬, 0: 끔 (시뮬레이터 저마찰 옵션을 끄고 달릴 때 0으로 두면 원래 속도로 달림)
    'lowmu_lead_m': 12.0,  # 저마찰 구역 경계가 어긋나도 안전하도록, 구역 앞 이만큼(m)부터 저마찰 속도 한계를 적용
    'k_stanley_lowmu': 2.0,   # 저마찰 구간 Stanley 옆오차 이득 (저속에서 조향을 끝까지 쓰도록)
    'k_soft_lowmu': 0.3,   # 저마찰 구간 Stanley 저속 완화값 (m/s). 작을수록 저속에서 더 적극적
    'k_heading_lowmu': 1.0,   # 저마찰 구간 방향 오차(전방 주시) 가중치
    'track_point_m': 2.6,  # Stanley가 경로에 맞추는 점: 뒤차축에서 앞으로 이만큼 (3.0 = 앞바퀴 축). 줄이면 뒤쪽이 코너 안쪽으로 덜 파고듦
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
        self._low = self._slope = self._low_plan = []
        self._steer_last = 0.0                       # 마지막으로 보낸 조향 (조향 변화율 제한용)
        self._braking = False                        # 구동/제동 전환 히스테리시스 상태
        self._u_sat = 0.0
        self.declare_parameter('course_json', '')    # 저마찰·경사 구간 좌표 파일. 비우면 패키지의 course/course.json
        self.zones = self.load_zones()
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

    def load_zones(self):
        """course.json의 저마찰(low_friction)·경사(slope) 구간 다각형. 못 읽으면 빈 dict (구간 대응만 꺼짐)."""
        path = self.get_parameter('course_json').value or str(COURSE_DIR / 'course.json')
        try:
            with open(path, encoding='utf-8') as f:
                missions = json.load(f)['course']['missions']
            kinds = ('low_friction', 'slope') if self.lowmu_enabled else ('slope',)
            return {m['type']: [tuple(p) for p in m['zone']] for m in missions
                    if m.get('type') in kinds and m.get('zone')}
        except Exception as e:
            self.get_logger().warn(f'저마찰·경사 구간을 못 읽어 일반 주행으로만 합니다: {e}')
            return {}

    def in_zone(self, kind, x, y):
        poly = self.zones.get(kind)
        return bool(poly) and point_in_polygon(poly, x, y)

    def on_end(self):
        self.active = False
        self.send(0.0, 1.0, 0.0)

    def reset_control(self):
        """시작·복귀 때 불립니다. 제어기 상태를 처음으로 돌립니다."""
        self.speed_error_sum = 0.0
        self._steer_last = 0.0
        self._braking = False
        self._u_sat = 0.0

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
        low = self.in_zone('low_friction', self.x, self.y)
        slope = self.in_zone('slope', self.x, self.y)
        if not self.path:                            # 경로 끝을 지남: 곧게 서서 멈춤
            steer, target = 0.0, 0.0
        else:
            rate = self.steer_rate_lowmu if low else self.steer_rate_max
            step = rate * self.dt                    # 조향 변화율 제한: 한 번에 이만큼만 바뀜
            want = self.steering(low)
            if want * self._steer_last < 0.0:        # 조향 방향이 바뀔 때는 제한 없이 바로 따라감 (S자 전환)
                steer = want
            else:
                steer = clamp(want, self._steer_last - step, self._steer_last + step)
            target = self.speed_target()
        accel, brake = self.longitudinal(target, low, slope)
        return accel, brake, steer

    def longitudinal(self, target, low, slope):
        """속력 PI + 앞먹임 제동. 반환: (accel, brake).
        - 적분항: 출력이 한계에 걸려 더 밀고 있을 때는 쌓지 않습니다(anti-windup). 경사 구간에서는 이득을 slope_ki_mult배.
        - 구동 <-> 제동 전환에 히스테리시스(brake_on/brake_off)를 둬서 경계에서 떨지 않게 합니다.
        - 저마찰 구간에서는 구동·제동 상한을 둬서 바퀴가 미끄러지지 않게 합니다."""
        error = target - self.speed
        brake_ff = self.brake_feedforward() if self.path and error < 0.3 else 0.0
        mult = self.slope_ki_mult if slope else 1.0
        ki = self.ki * mult
        if not ((self._u_sat > 0 and error > 0) or (self._u_sat < 0 and error < 0)):
            limit = 5.0 / mult                       # 적분 누적 한계 (ki x 누적 <= 0.4)
            self.speed_error_sum = clamp(self.speed_error_sum + error * self.dt, -limit, limit)
        u = self.kp * error + ki * self.speed_error_sum
        self._u_sat = 1.0 if u > 1.0 else (-1.0 if -u * self.brake_gain > 1.0 else 0.0)
        if brake_ff > 0.0 or u < -self.brake_on:
            self._braking = True
        elif u > -self.brake_off:
            self._braking = False
        if self._braking:
            accel, brake = 0.0, min(brake_ff + max(-u, 0.0) * self.brake_gain, 1.0)
        else:
            accel, brake = clamp(u, 0.0, 1.0), 0.0
        if low:
            accel, brake = min(accel, self.accel_max_lowmu), min(brake, self.brake_max_lowmu)
        return accel, brake

    def steering(self, low=False):
        """Stanley(방향 오차 + 옆 오차). 반환: steer -1~1.
        steer_delay_comp초 뒤의 예측 자세(predict_pose)를 기준으로 계산합니다. 지금 내는 조향은 그때 차에 닿기 때문입니다.
        (앞바퀴 축 기준 Stanley는 굽은 길의 곡률 앞먹임을 이미 포함하므로 따로 더하지 않습니다.)"""
        x, y, yaw = self.predict_pose()
        fx = x + self.track_point_m * math.cos(yaw)                  # 경로에 맞출 점 (기본: 앞바퀴 축)
        fy = y + self.track_point_m * math.sin(yaw)
        i = self.nearest_index(fx, fy)
        px, py, path_yaw = self.path[i]
        heading_error = angle_diff(path_yaw, yaw)
        cross_error = -math.sin(path_yaw) * (px - fx) + math.cos(path_yaw) * (py - fy)   # 경로가 왼쪽이면 +
        if low:                                      # 저마찰 구간: 저속에서도 조향을 끝까지 쓰도록 이득 보강
            wheel_angle = (self.k_heading_lowmu * heading_error
                           + math.atan2(self.k_stanley_lowmu * cross_error, self.k_soft_lowmu + abs(self.speed)))
        else:
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
        """보낸 조향 명령을 기록합니다 (예측과 조향 변화율 제한에 씀)."""
        t = self.now()
        self._steer_last = steer
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
        self._low = [self.in_zone('low_friction', p[0], p[1]) for p in path]
        self._slope = [self.in_zone('slope', p[0], p[1]) for p in path]
        plan, nxt = [False] * n, None               # 계획용: 저마찰 구역 앞 lowmu_lead_m 구간도 저마찰로 봄
        for i in range(n - 1, -1, -1):
            if self._low[i]:
                nxt = s[i]
            plan[i] = nxt is not None and nxt - s[i] <= self.lowmu_lead_m
        self._low_plan = plan
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
        low, slope = self._low_plan, self._slope
        v = []
        for i, c in enumerate(k):                    # 점마다 한계: 저마찰이면 더 보수적, 경사면 최고 속도 제한
            if low[i]:
                a_lat, lo, hi = self.a_lat_lowmu, self.v_min_lowmu, min(self.v_max, self.v_max_lowmu)
            else:
                a_lat, lo, hi = self.a_lat_max, self.v_min, min(self.v_max, self.v_max_slope) if slope[i] else self.v_max
            v.append(clamp(math.sqrt(a_lat / max(abs(c), 1e-6)), lo, hi))
        for i in range(n - 2, -1, -1):               # 앞쪽 점의 속도에서 감속으로 줄일 수 있는 속도까지만 허용
            a_dec = self.a_decel_lowmu if low[i] else self.a_decel
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_dec * (s[i + 1] - s[i])))
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


def point_in_polygon(poly, x, y):
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


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
