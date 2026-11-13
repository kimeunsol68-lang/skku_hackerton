"""HEVEN 2026 MIT 기본 코드 2/3: 경로 계획 (planner). 담당: 기본 주행·제어.

전역 경로(0.5 m 간격)에서 차 앞쪽 30 m를 잘라 20 Hz로 /hk/plan/local_path (nav_msgs/Path, frame map)에 냅니다.
각 점의 방향(orientation)은 그 점의 경로 방향입니다. 경로 끝을 지나면 빈 경로를 내서 control.py가 멈추게 합니다.

미션 스위치 (mission_manager.py)
    매 주기 기본 경로를 만든 뒤, 지금 CP 구간의 미션 모듈(mission_parking / mission_obstacle / mission_rotary)에
    물어봅니다.
      override_path가 있으면 → 기본 경로 대신 그 경로를 /hk/plan/local_path로 냄
      override_cmd가 있으면  → /hk/mission/override_cmd (hk_interfaces/CtrlCmd)로 내고, control.py가 그대로 보냄
      둘 다 None이면         → 원래와 똑같이 기본 경로만 냄 (override_cmd는 아예 내지 않음)
    미션이 쓰는 전처리 LiDAR는 /hk/perception/obstacles (geometry_msgs/PoseArray, map 좌표)로 받습니다(팀원 1이 냄).
    config/bringup.yaml의 missions_enabled: false로 미션 스위치를 통째로 끌 수 있습니다.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from hk_interfaces.msg import CtrlCmd

from .course import COURSE_DIR, declare_file_params, load_route, Route
from .cp_manager import CPManager
from .mission_manager import MissionManager
from .sim_events import SimEvents

RATE_HZ = 20.0
PARAMS = {              # config/bringup.yaml에서 바꿉니다. 아래는 기본값
    'ahead_m': 30.0,    # 차 앞으로 내보낼 길이 (m)
    'behind_m': 2.0,    # 차 뒤로 남길 길이 (m)
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
        self.create_subscription(Odometry, '/hk/local/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/hk/sensors/lidar/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseArray, OBSTACLES_TOPIC, self.on_obstacles, 10)
        self.pub = self.create_publisher(Path, '/hk/plan/local_path', 10)
        self.cmd_pub = self.create_publisher(CtrlCmd, OVERRIDE_CMD_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        SimEvents(self, self.on_reset, self.on_end)
        self.create_timer(1.0 / RATE_HZ, self.plan)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_reset(self, checkpoint, reload):
        if reload or not self.route.points:
            self.route = load_route(self)
            self.get_logger().info(f'전역 경로 {len(self.route.points)}점')
            self.missions = self.load_missions() if self.missions_enabled else None
        self.route.reset()
        self.checkpoint = checkpoint
        if self.missions:
            self.missions.reset(checkpoint)
        self.odom = None
        self.reset_time = self.now()
        self.active = True

    def load_missions(self):
        course_json = self.get_parameter('course_json').value or str(COURSE_DIR / 'course.json')
        route_csv = self.get_parameter('route_csv').value or str(COURSE_DIR / 'route.csv')
        try:
            return MissionManager(CPManager(course_json, route_csv), self.get_logger())
        except Exception as e:                         # 미션 스위치가 안 떠도 기본 주행은 계속
            self.get_logger().error(f'미션 스위치를 끕니다 (CPManager 로드 실패): {e}')
            return None

    def on_end(self):
        self.active = False

    def on_odom(self, m):
        self.odom = m

    def on_scan(self, m):
        self.scan = m

    def on_obstacles(self, m):
        self.obstacles = [(p.position.x, p.position.y) for p in m.poses]

    def plan(self):
        if not self.active or self.odom is None or len(self.route.points) < 2:
            return
        settling = self.now() - self.reset_time < SETTLE_S
        if settling:
            self.route.reset()
            if self.missions:
                self.missions.reset(self.checkpoint)
        p = self.odom.pose.pose.position
        points = self.default_path(p.x, p.y)

        if self.missions and not settling:
            q = self.odom.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            out, _, _ = self.missions.step(p.x, p.y, yaw, self.odom.twist.twist.linear.x, points,
                                           self.scan, self.obstacles, self.now())
            if out.override_path is not None:
                points = with_yaw(out.override_path)
            if out.override_cmd is not None:
                self.publish_cmd(*out.override_cmd)
        self.publish_path(points)

    def default_path(self, x, y):
        """기본 경로: 전역 경로에서 차 뒤 behind_m ~ 앞 ahead_m. [(x, y, yaw), ...], 경로 끝을 지났으면 []."""
        i = self.route.nearest(x, y)
        if self.route.past_end(x, y):
            return []
        first = max(0, i - int(self.behind_m / 0.5))
        last = min(len(self.route.points), i + int(self.ahead_m / 0.5) + 1)
        return [(*self.route.points[k], self.route.yaw_at(k)) for k in range(first, last)]

    def publish_path(self, points):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for x, y, yaw in points:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
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


def with_yaw(points):
    """[(x, y), ...] 또는 [(x, y, yaw), ...] -> [(x, y, yaw), ...]. yaw가 없으면 다음 점 방향."""
    if all(len(p) == 3 for p in points):
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
