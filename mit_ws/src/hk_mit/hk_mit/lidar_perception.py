"""HEVEN 2026 MIT LiDAR 인지 (lidar_perception).

/hk/sensors/lidar/scan (sensor_msgs/LaserScan, 10 Hz, 360°, 720빔, frame lidar_link)을 받아 self.scan에 둡니다.
기본 주행(bringup)에서는 띄우지 않고, perception:=true일 때만 뜹니다.

공용 함수 density_mask(): "LiDAR 노이즈" 옵션이 켜지면 허공에 흩어진 가짜 점이 생깁니다. 반경 안에 이웃 점이 충분하지 않은 고립된 점을
노이즈로 보고 지웁니다. mission_obstacle.py, mission_rotary.py가 LiDAR 점을 쓰기 전에 이 함수를 거칩니다.
노드는 같은 필터를 거친 스캔(고립된 점의 range를 +Inf로)을 /hk/perception/scan_filtered로도 냅니다.

쓰는 방법 예:
    import numpy as np
    r = np.asarray(m.ranges)                                   # 미검출 +Inf, 최소 거리 안 -Inf
    a = m.angle_min + np.arange(len(r)) * m.angle_increment    # 각 빔의 각도 (rad, lidar_link 기준)
    keep = density_mask(r, a, m.angle_increment)               # 고립된 점은 False
    x, y = r[keep] * np.cos(a[keep]), r[keep] * np.sin(a[keep])
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan

ISOLATION_RADIUS = 0.35     # 이 반경(m) 안에 이웃 점이 있어야 함
ISOLATION_NEIGHBORS = 3     # 이웃 점 최소 개수. 이보다 적으면 고립된 점(노이즈)
BEAM_FACTOR = 2.3           # 멀리서는 빔 사이가 벌어져서(20 m에서 약 0.17 m) 반경 0.35 m 안에 이웃 3개가 들어갈 수 없으므로,
                            # 반경을 max(0.35 m, 2.3 x 빔 간격)으로 넓힘. 고립된 노이즈는 멀리서도 이웃이 없어 그대로 지워짐


def density_mask(ranges, angles, angle_increment, valid=None,
                 radius=ISOLATION_RADIUS, min_neighbors=ISOLATION_NEIGHBORS, beam_factor=BEAM_FACTOR):
    """각 빔이 고립된 점이 아니면 True. valid(선택)는 처음부터 쓸 수 있는 빔(유한하고 거리 범위 안). 반환: bool 배열(빔 수)."""
    r = np.asarray(ranges, dtype=float)
    a = np.asarray(angles, dtype=float)
    ok = np.isfinite(r) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(r)
    keep = np.zeros(len(r), dtype=bool)
    idx = np.nonzero(ok)[0]
    if len(idx) <= min_neighbors:
        return keep
    x, y = r[idx] * np.cos(a[idx]), r[idx] * np.sin(a[idx])
    reach = np.maximum(radius, beam_factor * r[idx] * abs(angle_increment))
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    neighbors = (d2 <= (reach ** 2)[:, None]).sum(axis=1) - 1          # 자기 자신 제외
    keep[idx] = neighbors >= min_neighbors
    return keep


class LidarPerception(Node):
    def __init__(self):
        super().__init__('hk_mit_lidar_perception')
        self.scan = None
        self.pub = self.create_publisher(LaserScan, '/hk/perception/scan_filtered', qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/hk/sensors/lidar/scan', self.on_scan, qos_profile_sensor_data)

    def on_scan(self, m):
        self.scan = m
        r = np.asarray(m.ranges, dtype=float)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r >= m.range_min) & (r <= m.range_max)
        keep = density_mask(r, a, m.angle_increment, ok)
        out = LaserScan()
        out.header = m.header
        out.angle_min, out.angle_max, out.angle_increment = m.angle_min, m.angle_max, m.angle_increment
        out.time_increment, out.scan_time = m.time_increment, m.scan_time
        out.range_min, out.range_max = m.range_min, m.range_max
        out.ranges = [float(v) if k else float('inf') for v, k in zip(r, keep)]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LidarPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
