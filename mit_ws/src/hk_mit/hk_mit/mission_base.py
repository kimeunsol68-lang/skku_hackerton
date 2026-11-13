"""미션 모듈 표준 인터페이스 (공용 모듈, 담당: 기본 주행·제어 — 바꿀 때는 팀 전체와 상의).

mission_parking.py / mission_obstacle.py / mission_rotary.py는 모두 MissionBase를 상속하고 update()만 채웁니다.
planner.py 안의 MissionManager가 20 Hz로 "지금 CP 구간이 SECTIONS에 들어 있는 미션"의 update()를 부릅니다.

입력 MissionInput (모두 map 좌표계, SI 단위)
    x, y, yaw, speed   차의 위치(m, 뒤차축)·방향(rad)·속력(m/s)  ← /hk/local/odom
    local_path         planner가 만든 기본 경로 [(x, y, yaw), ...] (차 뒤 2 m ~ 앞 30 m, 0.5 m 간격, 비면 경로 끝)
    scan               원본 LiDAR sensor_msgs/LaserScan (lidar_link 좌표, 아직 안 왔으면 None)
    obstacles          전처리된 LiDAR 점/장애물 [(x, y), ...] map 좌표 (팀원 1의 /hk/perception/obstacles,
                       geometry_msgs/PoseArray. 아직 안 왔으면 None)
    t                  시뮬레이션 시간 (s, /clock)
    cp                 cp_manager.CPState (section, dist_to_next_m, missions 등)
    info               이 구간의 course.json missions 항목 (dict: zone, slot, center ... / 없으면 {})

출력 MissionOutput
    override_path      None = 기본 경로 그대로 / [(x, y), ...] 또는 [(x, y, yaw), ...] = 이 경로를 대신 냄
                       (control.py의 Stanley + PI가 이 경로를 따라감. 점 2개 이상, 0.5 m 간격 권장)
                       점을 (x, y, yaw, v_limit)로 내면 그 점의 속도 한계(m/s, 0 = 제한 없음)도 control에 전달됩니다.
    override_cmd       None = 평소 제어 / (throttle, brake, steer, gear) = control.py가 이 명령을 그대로 보냄
                         throttle 0~1, brake 0~1, steer -1~1 (+ 왼쪽, 1 = 35.4°), gear 'D' | 'R' | 'N' | 'P'
    is_done            True면 이 구간에서 미션 완료 → 이 구간이 끝날 때까지 다시 부르지 않음

규칙
    * update()는 오래 걸리면 안 됩니다(20 Hz). 예외가 나면 MissionManager가 로그만 남기고 None으로 처리합니다.
    * override_cmd를 낸 동안에는 control.py의 Stanley/PI가 멈춥니다. 내지 않으면 0.2 s 안에 평소 제어로 돌아갑니다.
    * 상태(단계, 타이머 등)는 self에 두고, reset()에서 처음으로 돌리세요.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

GEARS = ('D', 'R', 'N', 'P')


@dataclass
class MissionInput:
    x: float
    y: float
    yaw: float
    speed: float
    local_path: List[Tuple[float, float, float]]
    scan: Any = None                                        # sensor_msgs/LaserScan 또는 None
    obstacles: Optional[List[Tuple[float, float]]] = None   # map 좌표 점 또는 None
    t: float = 0.0
    cp: Any = None                                          # cp_manager.CPState
    info: dict = field(default_factory=dict)


@dataclass
class MissionOutput:
    override_path: Optional[Sequence[Sequence[float]]] = None
    override_cmd: Optional[Tuple[float, float, float, str]] = None
    is_done: bool = False


class MissionBase:
    NAME = 'base'
    SECTIONS = ()          # 이 미션이 불리는 CP 구간들. 예: (('cp8', 'cp9'), ('cp9', 'cp10'))

    def __init__(self, logger=None):
        self.log = logger  # rclpy 로거 (self.log.info('...')), 단독 시험 때는 None일 수 있음
        self.reset()

    def reset(self):
        """시작·체크포인트 복귀 때 불립니다. 상태를 처음으로 돌리세요."""

    def on_enter(self, section):
        """SECTIONS 중 한 구간에 들어올 때 한 번 불립니다. section = ('cp8', 'cp9') 등."""

    def update(self, inp: MissionInput) -> MissionOutput:
        """20 Hz. 기본값(아무것도 바꾸지 않음)을 돌려줍니다."""
        return MissionOutput()
