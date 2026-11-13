"""HEVEN 2026 MIT 기본 코드 2/3: 경로 계획 (planner). 담당: 기본 주행·제어.

전역 경로(0.5 m 간격)에서 차 앞쪽 30 m를 잘라 20 Hz로 /hk/plan/local_path (nav_msgs/Path, frame map)에 냅니다.
각 점의 방향(orientation)은 그 점의 경로 방향입니다. 경로 끝을 지나면 빈 경로를 내서 control.py가 멈추게 합니다.

미션 스위치 (mission_manager.py)
    매 주기 기본 경로를 만든 뒤, 지금 CP 구간의 미션 모듈(mission_parking / mission_obstacle / mission_rotary)에
    물어봅니다.
      override_path가 있으면 → 기본 경로 대신 그 경로를 /hk/plan/local_path로 냄
                                (점에 속도 한계 v_limit가 있으면 Path.pose.position.z로 같이 보냄. control.py가 목표 속도에 반영)
      override_cmd가 있으면  → /hk/mission/override_cmd (hk_interfaces/CtrlCmd)로 내고, control.py가 그대로 보냄
      둘 다 None이면         → 원래와 똑같이 기본 경로만 냄 (override_cmd는 아예 내지 않음)
    미션이 쓰는 전처리 LiDAR는 /hk/perception/obstacles (geometry_msgs/PoseArray, map 좌표)로 받습니다(팀원 1이 냄).
    config/bringup.yaml의 missions_enabled: false로 미션 스위치를 통째로 끌 수 있습니다.
"""
import json
import math
from std_msgs.msg import String

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from hk_interfaces.msg import CtrlCmd

from .course import COURSE_DIR, declare_file_params, load_runtime_config, load_route, option_on, Route
from .mission_parking import PARAMS as PARK_PARAMS, RPARAMS
from .sensing import PoseHistory, fresh, stamp
from .cp_manager import CPManager
from .mission_manager import MissionManager
from .sim_events import SimEvents

RATE_HZ = 20.0
PARAMS = {              # config/bringup.yaml에서 바꿉니다. 아래는 기본값
    'ahead_m': 45.0,    # minimum preview; grows with stopping distance
    'behind_m': 2.0,    # 차 뒤로 남길 길이 (m)
    'odom_timeout_s': 0.30,
    'scan_timeout_s': 0.35,
    'diagnostic_period_s': 1.0,
}
SETTLE_S = 0.5          # 시작·복귀 뒤 이 시간 동안은 경로 전체에서 위치를 다시 찾음 (늦게 온 예전 위치 대비)
OVERRIDE_CMD_TOPIC = '/hk/mission/override_cmd'
OBSTACLES_TOPIC = '/hk/perception/obstacles'
GEAR = {'D': CtrlCmd.GEAR_DRIVE, 'R': CtrlCmd.GEAR_REVERSE, 'N': CtrlCmd.GEAR_NEUTRAL, 'P': CtrlCmd.GEAR_PARK}


class Planner(Node):
    def __init__(self):
        super().__init__('hk_mit_planner')
        declare_file_params(self)
        for name, default in PARAMS.items():           # self.ahead_m, self.behind_m
            self.declare_parameter(name, default)
            setattr(self, name, float(self.get_parameter(name).value))
        self.declare_parameter('obstacle3_speed', 0.0)       # team profile excludes obstacle3
        self.declare_parameter('obstacle3_lead_m', 12.0)     # 구역 입구 이만큼(m) 앞부터 적용
        self.slow_zone = None                                # obstacle3 구역 다각형 (course.json)
        self.declare_parameter('missions_enabled', True)
        self.missions_enabled = bool(self.get_parameter('missions_enabled').value)
        self.route = Route()
        self.missions = None
        self.checkpoint = 0
        self.odom = None
        self.scan = None
        self.obstacles = None
        self.active = False
        self.reset_time = None
        self.history = PoseHistory()
        self.runtime_config = None
        self.config_ready = False
        self.config_retry = self.last_diagnostic = -math.inf
        self.obstacles_stamp = -math.inf
        self.mission_parameters = {}
        for name, default in dict(PARK_PARAMS, **RPARAMS).items():
            self.declare_parameter(name, default)
            self.mission_parameters[name] = self.get_parameter(name).value
        self.create_subscription(Odometry, '/hk/local/odom', self.on_odom, 1)
        self.create_subscription(LaserScan, '/hk/sensors/lidar/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseArray, OBSTACLES_TOPIC, self.on_obstacles, 10)
        self.pub = self.create_publisher(Path, '/hk/plan/local_path', 1)
        self.diag_pub = self.create_publisher(String, '/hk/plan/diagnostics', 1)
        self.cmd_pub = self.create_publisher(CtrlCmd, OVERRIDE_CMD_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        SimEvents(self, self.on_reset, self.on_end)
        self.create_timer(1.0 / RATE_HZ, self.plan)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_reset(self, checkpoint, reload):
        if reload or not self.route.points:
            self.route = load_route(self)
            self.get_logger().info(f'전역 경로 {len(self.route.points)}점')
            self.configure_attempt()
        self.route.reset()
        self.checkpoint = checkpoint
        if self.missions:
            self.missions.reset(checkpoint)
        self.odom = self.scan = self.obstacles = None
        self.history.clear()
        self.obstacles_stamp = -math.inf
        self.reset_time = self.now()
        self.active = True

    def configure_attempt(self):
        self.config_retry = self.now()
        try:
            self.runtime_config = load_runtime_config(self.get_parameter('course_json').value or None)
            opts = self.runtime_config['options']
            course = self.runtime_config.get('course', {})
            delay = float(course.get('steer_latency_ms', 250.0)) / 1000.0 if option_on(opts, 'steer_latency') else 0.0
            self.mission_parameters.update(park_delay=delay, rpark_delay=delay)
            self.missions = self.load_missions() if self.missions_enabled else None
            if self.missions_enabled and self.missions is None:
                raise ValueError('미션 초기화 실패')
            self.slow_zone = self.load_slow_zone()
            self.config_ready = True
            self.get_logger().info('주행 구성: ' + json.dumps({'source': self.runtime_config['_source'],
                'options': opts, 'parking_steer_delay_s': delay}, ensure_ascii=False))
        except (ValueError, OSError) as error:
            self.config_ready = False
            self.get_logger().error(f'구성 확인 대기 (정지): {error}')

    def load_missions(self):
        course_json = self.get_parameter('course_json').value or str(COURSE_DIR / 'course.json')
        route_csv = self.get_parameter('route_csv').value or str(COURSE_DIR / 'route.csv')
        try:
            return MissionManager(CPManager(course_json, route_csv), self.get_logger(),
                                  self.runtime_config['options'], self.mission_parameters)
        except Exception as e:                         # 미션 스위치가 안 떠도 기본 주행은 계속
            self.get_logger().error(f'미션 스위치를 끕니다 (CPManager 로드 실패): {e}')
            return None

    def load_slow_zone(self):
        """course.json의 obstacle3 구역 다각형. 못 읽으면 None (속도 제한만 꺼짐)."""
        if float(self.get_parameter('obstacle3_speed').value) <= 0.0:
            return None
        if not option_on(self.runtime_config['options'], 'obstacle3'):
            return None                                    # 이번 시도에서 장애물 3개 옵션이 꺼져 있으면 속도를 낮출 필요 없음
        path = self.get_parameter('course_json').value or str(COURSE_DIR / 'course.json')
        try:
            with open(path, encoding='utf-8') as f:
                for m in json.load(f)['course']['missions']:
                    if m.get('type') == 'obstacle3' and m.get('zone'):
                        return [tuple(p) for p in m['zone']]
        except Exception as e:
            self.get_logger().warn(f'obstacle3 구역을 못 읽어 속도 제한을 끕니다: {e}')
        return None

    def with_speed_zone(self, points):
        """obstacle3 구역과 그 입구 obstacle3_lead_m 앞의 경로 점에 속도 한계(4번째 값)를 붙입니다.
        미션 경로가 이미 속도 한계를 갖고 있으면 둘 중 작은 값. control.py가 Path.pose.position.z로 읽어 목표 속도에 반영합니다."""
        if not self.slow_zone or not points:
            return points
        v_zone = float(self.get_parameter('obstacle3_speed').value)
        lead = float(self.get_parameter('obstacle3_lead_m').value)
        n = len(points)
        s = [0.0]
        for a, b in zip(points, points[1:]):
            s.append(s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        flag, nxt = [False] * n, None
        for i in range(n - 1, -1, -1):                       # 구역 안 점, 그리고 구역 입구 lead m 앞의 점
            if in_polygon(self.slow_zone, points[i][0], points[i][1]):
                nxt = s[i]
            flag[i] = nxt is not None and nxt - s[i] <= lead
        out = []
        for p, f in zip(points, flag):
            v = float(p[3]) if len(p) > 3 else 0.0
            if f:
                v = min(v, v_zone) if v > 0.0 else v_zone
            out.append((p[0], p[1], p[2], v))
        return out

    def on_end(self):
        self.active = False

    def on_odom(self, m):
        if self.active and fresh(m, self.now(), self.odom_timeout_s, self.reset_time):
            if m.twist.covariance[0] >= 1e5:
                self.odom = None
                return
            self.odom = m
            self.history.add(m)

    def on_scan(self, m):
        if self.active and fresh(m, self.now(), self.scan_timeout_s, self.reset_time):
            self.scan = m

    def on_obstacles(self, m):
        if self.active and fresh(m, self.now(), self.scan_timeout_s, self.reset_time):
            self.obstacles = [(p.position.x, p.position.y) for p in m.poses]
            self.obstacles_stamp = stamp(m)

    def plan(self):
        if not self.active:
            return
        if not self.config_ready:
            if self.now() - self.config_retry >= 1.0:
                self.configure_attempt()
            self.publish_path([])
            self.publish_cmd(0.0, 1.0, 0.0, 'D')
            return
        if not fresh(self.odom, self.now(), self.odom_timeout_s, self.reset_time) or len(self.route.points) < 2:
            self.publish_path([])
            return
        settling = self.now() - self.reset_time < SETTLE_S
        if settling:
            self.route.reset()
            if self.missions:
                self.missions.reset(self.checkpoint)
        p = self.odom.pose.pose.position
        points = self.default_path(p.x, p.y)

        name, state = None, None
        if self.missions and not settling:
            q = self.odom.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            scan = self.scan if fresh(self.scan, self.now(), self.scan_timeout_s, self.reset_time) else None
            scan_pose = self.history.at(stamp(scan)) if scan is not None else None
            if scan_pose is None:
                scan = None
            sigma = math.sqrt(max(0.0, self.odom.pose.covariance[0], self.odom.pose.covariance[7]))
            obstacles = self.obstacles if self.now() - self.obstacles_stamp <= self.scan_timeout_s else None
            out, name, state = self.missions.step(p.x, p.y, yaw, self.odom.twist.twist.linear.x, points,
                                           scan, obstacles, self.now(), scan_pose, sigma)
            if out.override_path is not None:
                points = with_yaw(out.override_path)
            if out.override_cmd is not None:
                self.publish_cmd(*out.override_cmd)
        self.publish_path(self.with_speed_zone(points))
        if self.now() - self.last_diagnostic >= self.diagnostic_period_s:
            self.last_diagnostic = self.now()
            payload = {'t': self.now(), 'attempt_id': getattr(self, '_sim_attempt_id', ''),
                       'section': state.section if state else None, 'mission': name,
                       'speed': self.odom.twist.twist.linear.x,
                       'scan_age_s': self.now() - stamp(self.scan) if self.scan else None,
                       'path_points': len(points)}
            self.diag_pub.publish(String(data=json.dumps(payload)))

    def default_path(self, x, y):
        """기본 경로: 전역 경로에서 차 뒤 behind_m ~ 앞 ahead_m. [(x, y, yaw), ...], 경로 끝을 지났으면 []."""
        i = self.route.nearest(x, y)
        if self.route.past_end(x, y):
            return []
        first = max(0, i - int(self.behind_m / 0.5))
        speed = abs(self.odom.twist.twist.linear.x) if self.odom else 0.0
        preview = min(100.0, max(self.ahead_m, speed * 0.35 + max(0., speed**2 - 2.5**2) / 3.6 + 8.0))
        last = min(len(self.route.points), i + int(preview / 0.5) + 1)
        result = []
        opts = self.runtime_config['options'] if self.runtime_config else {}
        gates = self.missions.cp.cp_idx if self.missions else []
        for k in range(first, last):
            cap = 0.0
            # Plan braking before the direct parking controller takes over.
            for cp_index, option, limit in ((7, 'parallel_parking', 6.0), (8, 'rear_parking', 5.0)):
                if len(gates) > cp_index and option_on(opts, option) and gates[cp_index]-10 <= k <= gates[cp_index]+12:
                    cap = limit if cap == 0.0 else min(cap, limit)
            result.append((*self.route.points[k], self.route.yaw_at(k), cap))
        return result

    def publish_path(self, points):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for p in points:
            x, y, yaw = p[:3]
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.position.z = float(p[3]) if len(p) > 3 else 0.0      # 속도 한계 (m/s, 0 = 제한 없음)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            path.poses.append(ps)
        self.pub.publish(path)

    def publish_cmd(self, throttle, brake, steer, gear):
        c = CtrlCmd()
        c.header.stamp = self.get_clock().now().to_msg()
        c.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        c.gear = GEAR[gear]
        c.long_cmd_type = CtrlCmd.LONG_CMD_THROTTLE
        c.accel, c.brake, c.steering = float(throttle), float(brake), float(steer)
        self.cmd_pub.publish(c)


def in_polygon(poly, x, y):
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def with_yaw(points):
    """[(x, y), ...] / [(x, y, yaw), ...] / [(x, y, yaw, v_limit), ...] -> yaw를 채운 목록. yaw가 없으면 다음 점 방향.
    4번째 값(속도 한계, m/s)은 있으면 그대로 둡니다."""
    if all(len(p) >= 3 for p in points):
        return [tuple(p) for p in points]
    out = []
    for k, p in enumerate(points):
        a, b = (points[k], points[k + 1]) if k + 1 < len(points) else (points[k - 1], points[k])
        out.append((p[0], p[1], math.atan2(b[1] - a[1], b[0] - a[0])))
    return out


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
