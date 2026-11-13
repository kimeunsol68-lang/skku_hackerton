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
from std_msgs.msg import String

from .course import fast_mode, COURSE_DIR, load_runtime_config, option_on
from .sensing import fresh, stamp, predict_motion
from .sim_events import SimEvents

# 조정할 값은 config/bringup.yaml의 파라미터입니다(공식 주행 직전에는 코드가 아니라 이 파일만 바꿀 수 있음).
# 아래는 yaml에 값이 없을 때 쓰는 기본값입니다.
TARGET_SPEED_DEFAULT = 5.0
# 공격 모드 값 (2차 시도 전용). 안전 모드 값은 bringup.yaml (a_lat_max 3.0, a_lat_lowmu 1.3 ...)
FAST_PARAMS = {
    'a_lat_max': 3.5,       # 마른 노면 커브 (안전 3.0). 3.8 + v_max 15에서 cp2 뒤·cp5 뒤 커브 충돌
    'a_decel': 3.0,         # 커브 앞 감속 계획
    'a_lat_lowmu': 1.55,    # 저마찰 커브 (안전 1.3, mu=0.2 한계 1.96)
    'a_lat_slope': 3.3,     # 경사 헤어핀 (안전 3.0)
    'v_max_obstacle': 6.0,  # 장애물 구간 (안전 5.0)
}

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
    'v_max_obstacle': 3.5,   # 장애물 구간(obstacle1/obstacle3 zone) 안 최고 속도 (m/s). 미션이 장애물을 찾기 전에도 미리 낮춰 둠
    'obstacle_lead_m': 12.0,  # 장애물 구역 입구 이만큼(m) 앞에서부터 이미 v_max_obstacle로 달리게 함
    'lowmu_enabled': 1.0,  # 1: 저마찰 구역 대응 켬, 0: 끔 (시뮬레이터 저마찰 옵션을 끄고 달릴 때 0으로 두면 원래 속도로 달림)
    'lowmu_lead_m': 12.0,  # 저마찰 구역 경계가 어긋나도 안전하도록, 구역 앞 이만큼(m)부터 저마찰 속도 한계를 적용
    'lowmu_brake_lead_m': 4.0,  # 구역 앞 이만큼(m)부터는 저마찰 감속(a_decel_lowmu)으로 계획. 그보다 앞(마른 노면)에서는 a_decel로 제동
    'k_stanley_lowmu': 2.0,   # 저마찰 구간 Stanley 옆오차 이득 (저속에서 조향을 끝까지 쓰도록)
    'k_soft_lowmu': 0.3,   # 저마찰 구간 Stanley 저속 완화값 (m/s). 작을수록 저속에서 더 적극적
    'k_heading_lowmu': 1.0,   # 저마찰 구간 방향 오차(전방 주시) 가중치
    'track_point_m': 2.6,  # Stanley가 경로에 맞추는 점: 뒤차축에서 앞으로 이만큼 (3.0 = 앞바퀴 축). 줄이면 뒤쪽이 코너 안쪽으로 덜 파고듦
    'a_lat_slope': 2.8,
    'corner_hold_m': 8.0,           # 급커브 앞 이 거리(m) 안에서는 그 커브 속도 + margin까지만 허용 (0이면 끔)
    'corner_hold_margin': 0.6,      # 위 허용 여유 (m/s)
    'corner_hold_vmax': 6.0,        # 이 속도보다 느린 커브만 급커브로 봄 (m/s)
    'odom_timeout_s': 0.30,
    'path_timeout_s': 0.35,
    'diagnostic_period_s': 1.0,
    'auto_latency': 1.0,
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
        self.path_vlim = []
        self._curv_src = self._curv = self._s = self._prof = None
        self._low = self._slope = self._low_plan = self._obst = []
        self._steer_last = 0.0                       # 마지막으로 보낸 조향 (조향 변화율 제한용)
        self._braking = False                        # 구동/제동 전환 히스테리시스 상태
        self._u_sat = 0.0
        self.declare_parameter('course_json', '')    # 저마찰·경사 구간 좌표 파일. 비우면 패키지의 course/course.json
        self.zones = {}
        self.runtime_config = None
        self.config_ready = False
        self.config_retry = self.last_diagnostic = -math.inf
        self.last_command_gear = CtrlCmd.GEAR_DRIVE
        self.measured_acceleration = 0.0
        self.last_speed_sample = None
        self.target = 0.0
        self.target_reason = 'waiting'
        self.cross_error = self.heading_error = 0.0
        self.steer_hist = deque()                    # (보낸 시각, 앞바퀴 각도 rad): 아직 차에 닿지 않은 조향 명령 포함
        self.cmd_pub = self.create_publisher(CtrlCmd, '/hk/control/command', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.diag_pub = self.create_publisher(String, '/hk/control/diagnostics', 1)
        self.create_subscription(Odometry, '/hk/local/odom', self.on_odom, 1)
        self.create_subscription(Path, '/hk/plan/local_path', self.on_path, 1)
        self.create_subscription(CtrlCmd, OVERRIDE_CMD_TOPIC, self.on_override,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        SimEvents(self, self.on_reset, self.on_end)
        self.create_timer(self.dt, self.tick)

    # ---- 시작·복귀 -----------------------------------------------------------------------------------
    def on_reset(self, checkpoint, reload):
        if reload:                                   # 새 시도 시작: 이번에 켜진 옵션에 맞게 구역 제한을 다시 정함
            self.configure_attempt()
        self.reset_control()
        self.odom = self.path = None
        self.override = None
        self.override_active = False
        self.steer_hist.clear()
        self.path_vlim = []
        self._curv_src = self._prof = None
        self.last_speed_sample = None
        self.measured_acceleration = 0.0
        self.last_command_gear = CtrlCmd.GEAR_DRIVE
        self.reset_time = self.now()
        self.active = True

    def configure_attempt(self):
        self.config_retry = self.now()
        try:
            self.runtime_config = load_runtime_config(self.get_parameter('course_json').value or None)
            options = self.runtime_config['options']
            c = self.runtime_config.get('course', {})
            if self.auto_latency:
                self.steer_delay_comp = float(c.get('steer_latency_ms', 250)) / 1000.0 if option_on(options, 'steer_latency') else 0.0
                self.speed_lead_s = (float(c.get('drive_latency_ms', 75)) / 1000.0 if option_on(options, 'drive_latency') else 0.0) + 0.04
            # 공격 모드(2차 시도): LiDAR 노이즈만 끈 구성에서 커브·저마찰·경사·장애물 속도 한계를 올림
            if not hasattr(self, '_safe_params'):
                self._safe_params = {k: getattr(self, k) for k in FAST_PARAMS}
            fast = fast_mode(options)
            for k, v in (FAST_PARAMS if fast else self._safe_params).items():
                setattr(self, k, v)
            if fast:
                self.get_logger().warn(f'공격 모드 (LiDAR 노이즈 꺼짐): {FAST_PARAMS}')
            self.zones = self.load_zones()
            self._curv_src = None
            self.config_ready = True
            self.get_logger().info(f'제어 지연 보상: 조향 {self.steer_delay_comp:.3f}s, 속도 앞보기 {self.speed_lead_s:.3f}s; v_max={self.v_max:.1f}')
        except (ValueError, OSError) as error:
            self.config_ready = False
            self.get_logger().error(f'구성 확인 대기 (정지): {error}')

    def load_zones(self):
        """course.json의 저마찰(low_friction)·경사(slope)·장애물 구간 다각형. 못 읽으면 빈 dict (구간 대응만 꺼짐).
        시뮬레이터가 이번 시도에서 켠 옵션(/data/current/course.json)에 든 구역만 돌려줍니다: 꺼진 옵션의 구역은 평소 속도로 달립니다.
        옵션을 못 읽으면 모든 구역 대응을 켭니다."""
        path = self.get_parameter('course_json').value or str(COURSE_DIR / 'course.json')
        try:
            with open(path, encoding='utf-8') as f:
                missions = json.load(f)['course']['missions']
            options = self.runtime_config['options']
            kinds = tuple(k for k in ('low_friction', 'slope', 'obstacle1', 'obstacle3')
                          if option_on(options, k) and (k != 'low_friction' or self.lowmu_enabled))
            if getattr(self, '_zone_kinds', None) != kinds:
                self._zone_kinds = kinds
                self.get_logger().info(f'속도 제한을 적용하는 구역: {", ".join(kinds) or "없음"}'
                                       + ('' if options is not None else ' (옵션 파일을 못 읽어 전부 켬)'))
            return {m['type']: [tuple(p) for p in m['zone']] for m in missions
                    if m.get('type') in kinds and m.get('zone')}
        except Exception as e:
            raise ValueError(f'노면 구간을 읽지 못해 주행을 대기합니다: {e}') from e

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
        if not self.active or not fresh(m, self.now(), self.odom_timeout_s, self.reset_time):
            return
        if m.twist.covariance[0] >= 1e5:
            self.odom = None
            return
        t = stamp(m)
        if self.last_speed_sample is not None:
            previous_t, previous_v = self.last_speed_sample
            dt = t - previous_t
            if dt <= 0.0:
                return
            a = clamp((m.twist.twist.linear.x - previous_v) / dt, -4.0, 3.0)
            self.measured_acceleration += (1.0 - math.exp(-dt / 0.25)) * (a - self.measured_acceleration)
        self.last_speed_sample = (t, m.twist.twist.linear.x)
        self.odom = m
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.speed = m.twist.twist.linear.x

    def on_path(self, m):
        if not self.active or not fresh(m, self.now(), self.path_timeout_s, self.reset_time):
            return
        self.path_stamp = stamp(m)
        self.path = [(ps.pose.position.x, ps.pose.position.y,
                      2 * math.atan2(ps.pose.orientation.z, ps.pose.orientation.w)) for ps in m.poses]
        self.path_vlim = [ps.pose.position.z for ps in m.poses]      # 미션이 붙인 점별 속도 한계 (m/s, 0 = 없음)

    def on_override(self, m):
        if self.active and fresh(m, self.now(), self.override_timeout_s, self.reset_time):
            self.override = (m, stamp(m))

    def nearest(self, x, y):
        return min(self.path, key=lambda p: math.hypot(p[0] - x, p[1] - y))

    # ---- 50 Hz ------------------------------------------------------------------------------------------
    def tick(self):
        if not self.active:
            return
        if not self.config_ready:
            if self.now() - self.config_retry >= 1.0:
                self.configure_attempt()
            self.target_reason = 'waiting_for_configuration'
            self.send(0.0, 1.0, 0.0, self.last_command_gear)
            return
        if not fresh(self.odom, self.now(), self.odom_timeout_s, self.reset_time) or self.now() - self.reset_time < SETTLE_S:
            self.target_reason = 'waiting_for_odometry'
            self.send(0.0, 1.0, 0.0, self.last_command_gear)
            return
        if self.override is not None and self.now() - self.override[1] <= self.override_timeout_s:
            m = self.override[0]                     # 미션 직접 제어가 우선
            self.override_active = True
            self.target_reason = 'mission_override'
            self.send(m.accel, m.brake, m.steering, m.gear)
            return
        if self.path is None or self.now() - self.path_stamp > self.path_timeout_s:
            self.target_reason = 'waiting_for_path'
            self.send(0.0, 1.0, 0.0, self.last_command_gear)
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
            steer = clamp(want, self._steer_last - step, self._steer_last + step)
            target = self.speed_target()
        self.target = target
        if not self.path:
            self.target_reason = 'end_of_path'
        accel, brake = self.longitudinal(target, low, slope)
        return accel, brake, steer

    def longitudinal(self, target, low, slope):
        """속력 PI + 앞먹임 제동. 반환: (accel, brake).
        - 적분항: 출력이 한계에 걸려 더 밀고 있을 때는 쌓지 않습니다(anti-windup). 경사 구간에서는 이득을 slope_ki_mult배.
        - 구동 <-> 제동 전환에 히스테리시스(brake_on/brake_off)를 둬서 경계에서 떨지 않게 합니다.
        - 저마찰 구간에서는 구동·제동 상한을 둬서 바퀴가 미끄러지지 않게 합니다."""
        error = target - self.speed
        brake_ff = self.brake_feedforward() if self.path and error < 0.1 else 0.0
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
        self.cross_error, self.heading_error = cross_error, heading_error
        if low:                                      # 저마찰 구간: 저속에서도 조향을 끝까지 쓰도록 이득 보강
            wheel_angle = (self.k_heading_lowmu * heading_error
                           + math.atan2(self.k_stanley_lowmu * cross_error, self.k_soft_lowmu + abs(self.speed)))
        else:
            wheel_angle = heading_error + math.atan2(self.k_stanley * cross_error, self.k_soft + self.speed)
        return clamp(wheel_angle / MAX_STEER, -1.0, 1.0)

    def predict_pose(self):
        """Bicycle Kinematic Model로 steer_delay_comp초 뒤의 뒤차축 위치·방향 (x, y, yaw)를 예측합니다.
        속력은 최근 가감속을 반영하고, 그 사이 차에 닿는 조향은 이미 보낸 명령(steer_hist)에서 가져옵니다
        (시각 t의 조향각 = t - steer_delay_comp에 보낸 명령). 보상 시간이 0이면 지금 자세 그대로입니다."""
        return predict_motion(self.x, self.y, self.yaw, self.speed, self.now(),
                              self.steer_delay_comp, self.steer_hist, self.measured_acceleration)

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
        obst = [self.in_zone('obstacle1', p[0], p[1]) or self.in_zone('obstacle3', p[0], p[1]) for p in path]
        self._obst, nxt = [False] * n, None         # 장애물 구역 입구 obstacle_lead_m 앞부터 구역 끝까지
        for i in range(n - 1, -1, -1):
            if obst[i]:
                nxt = s[i]
            self._obst[i] = nxt is not None and nxt - s[i] <= self.obstacle_lead_m
        plan, brake, nxt = [False] * n, [False] * n, None   # 계획용: 저마찰 구역 앞 lowmu_lead_m 구간도 저마찰로 봄
        for i in range(n - 1, -1, -1):
            if self._low[i]:
                nxt = s[i]
            plan[i] = nxt is not None and nxt - s[i] <= self.lowmu_lead_m
            brake[i] = nxt is not None and nxt - s[i] <= self.lowmu_brake_lead_m   # 감속 계획만 저마찰 값으로 (속도 상한은 plan)
        self._low_plan, self._low_brake = plan, brake
        return k

    def brake_feedforward(self):
        """앞으로 brake_ff_lookahead_s 동안 프로파일이 요구하는 감속을 brake(0~1)로 바꾼 앞먹임. 줄일 필요가 없으면 0."""
        if not self.profile_enabled or self.brake_ff_decel <= 0.0 or self._prof is None:
            return 0.0
        v, s = self._prof, self._s
        i = self.nearest_index(self.x, self.y)
        j = i
        while j + 1 < len(v) and s[j + 1] - s[i] <= max(0.5, abs(self.speed) * self.brake_ff_lookahead_s):
            j += 1
        if j <= i or s[j] - s[i] < 1e-6:
            return 0.0
        a_req = (self.speed ** 2 - v[j] ** 2) / (2.0 * (s[j] - s[i]))
        return clamp(a_req / self.brake_ff_decel, 0.0, 1.0)

    def speed_target(self):
        """목표 속력 (m/s). 곡률 한계 속도에 코너 앞 감속(뒤에서 앞으로 전파)을 더한 프로파일에서 지금 위치의 값."""
        if not self.profile_enabled:
            # keep mission speed limits (roundabout stop line, parking settle) even without the profile
            self.target_reason = 'constant_speed'
            lims = [v for v in self.path_vlim if v > 0.0] if self.path is not None else []
            if lims and min(lims) < self.target_speed:
                self.target_reason = 'mission_path'
                return min(lims)
            return self.target_speed
        k = self.curvature()
        s = self._s
        n = len(k)
        low, slope = self._low_plan, self._slope
        v, reasons = [], []
        for i, c in enumerate(k):                    # 점마다 한계: 저마찰이면 더 보수적, 경사면 최고 속도 제한
            if low[i]:
                a_lat, lo, hi = self.a_lat_lowmu, self.v_min_lowmu, min(self.v_max, self.v_max_lowmu)
                reason = 'low_friction'
            else:
                a_lat = self.a_lat_slope if slope[i] else self.a_lat_max
                lo, hi = self.v_min, min(self.v_max, self.v_max_slope) if slope[i] else self.v_max
                reason = 'slope' if slope[i] else 'straight'
            if self._obst[i]:                        # 장애물 구간: 어디서 장애물이 나와도 대응할 수 있게 미리 속도를 낮춤
                hi = min(hi, self.v_max_obstacle)
                lo = min(lo, hi)                     # v_min(4.5)이 구간 최고 속도보다 크면 구간 속도가 무시되므로 같이 낮춤
                reason = 'obstacle_zone'
            curve_limit = math.sqrt(a_lat / max(abs(c), 1e-6))
            vi = min(hi, curve_limit)                # a speed floor must never defeat a corner limit
            if curve_limit < hi:
                reason = reason + '_curvature'
            if len(self.path_vlim) == n and self.path_vlim[i] > 0.0:   # 미션이 붙인 속도 한계는 v_min보다도 우선
                if self.path_vlim[i] < vi:
                    vi, reason = self.path_vlim[i], 'mission_path'
            v.append(vi)
            reasons.append(reason)
        if self.corner_hold_m > 0.0:                 # 급커브 직전에는 가속하지 않음: 짧은 직선에서 올린 속도를 급조향 중 제동으로 못 줄여 이탈함
            hold = list(v)
            # only curvature limits count as corners: a mission stop-line limit (roundabout V_FLOOR 1 m/s) held the
            # car at 1.6 m/s for the last 8 m before the stop line (~5 s per lap)
            vc = [vi if r not in ('mission_path',) else math.inf for vi, r in zip(v, reasons)]
            j = 0
            for i in range(n):
                j = max(j, i)
                while j + 1 < n and s[j + 1] - s[i] <= self.corner_hold_m:
                    j += 1
                m = min(vc[i:j + 1])
                if m < self.corner_hold_vmax and m + self.corner_hold_margin < hold[i]:
                    hold[i] = m + self.corner_hold_margin
                    reasons[i] = 'corner_hold'
            v = hold
        for i in range(n - 2, -1, -1):               # 앞쪽 점의 속도에서 감속으로 줄일 수 있는 속도까지만 허용
            a_dec = self.a_decel_lowmu if self._low_brake[i] else self.a_decel   # 구역 앞 마른 노면에서는 평소 감속으로 늦게 제동
            limit = math.sqrt(v[i + 1] ** 2 + 2.0 * a_dec * (s[i + 1] - s[i]))
            if limit < v[i]:
                v[i], reasons[i] = limit, 'braking_ahead'
        self._prof = v
        i = self.nearest_index(self.x, self.y)
        last = i
        while last + 1 < n and s[last + 1] - s[i] <= max(0.0, self.speed * self.speed_lead_s):
            last += 1
        limiting = min(range(i, last + 1), key=lambda j: v[j])
        self.target_reason = reasons[limiting]
        return min(v[i:last + 1])

    def send(self, accel, brake, steer, gear=CtrlCmd.GEAR_DRIVE):
        self.last_command_gear = gear
        self.note_steer(float(steer))
        c = CtrlCmd()
        c.header.stamp = self.get_clock().now().to_msg()
        c.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        c.gear = int(gear)
        c.long_cmd_type = CtrlCmd.LONG_CMD_THROTTLE
        c.accel, c.brake, c.steering = float(accel), float(brake), float(steer)
        self.cmd_pub.publish(c)
        if self.now() - self.last_diagnostic >= self.diagnostic_period_s:
            self.last_diagnostic = self.now()
            data = {'t': self.now(), 'attempt_id': getattr(self, '_sim_attempt_id', ''),
                    'speed': self.speed if self.odom else 0.0,
                    'target': None if self.target_reason == 'mission_override' else self.target,
                    'reason': self.target_reason, 'cross_error': self.cross_error,
                    'heading_error_rad': self.heading_error, 'accel': accel, 'brake': brake,
                    'steer': steer, 'gear': gear,
                    'odom_age_s': self.now() - stamp(self.odom) if self.odom else None}
            self.diag_pub.publish(String(data=json.dumps(data)))


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
