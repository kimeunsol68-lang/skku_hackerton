"""주차 미션 (담당: 팀원 2). 이 파일만 고치세요.

담당 구간
    CP8 -> CP9   평행주차 (course.json mission 'm_ppark', type 'parallel_parking')
                 slot: 중심 (-18.11, 120.78), 방향 -90°(남쪽), 길이 7.4 m, 폭 3.6 m
                 slot은 axis (0, 1) 방향으로 ±3.0 m 안에서 시도마다 바뀝니다(slide) → LiDAR로 실제 위치를 찾아야 함.
    CP9 -> CP10  후면주차 (course.json mission 'm_rpark', type 'rear_parking')
                 slot: 중심 (0.34, 82.98), 방향 -90°, 길이 5.4 m, 폭 3.6 m, 역시 ±3.0 m slide.
    정확한 값은 inp.info['slot'], inp.info['zone']으로 받습니다(지금 구간의 course.json 항목).

입력 inp: mission_base.MissionInput
    inp.x, inp.y, inp.yaw, inp.speed   차 자세·속력 (map, 뒤차축)
    inp.local_path                     기본 경로 [(x, y, yaw), ...]
    inp.scan / inp.obstacles           원본 LaserScan / 전처리된 장애물 점 [(x, y), ...] (map)
    inp.t                              시뮬레이션 시간 (s)
    inp.cp.section                     ('cp8', 'cp9') 또는 ('cp9', 'cp10')
    inp.cp.dist_to_next_m              다음 게이트까지 남은 거리 (m)

출력 MissionOutput(override_path=None, override_cmd=None, is_done=False)
    override_path   주차 공간 근처까지 가는 경로를 바꾸고 싶을 때 [(x, y), ...]
    override_cmd    직접 조작할 때 (throttle 0~1, brake 0~1, steer -1~1, gear 'D'|'R'|'N'|'P')
                    예: 후진 = (0.2, 0.0, -0.5, 'R'), 정지 = (0.0, 1.0, 0.0, 'D')
    is_done         주차 후 빠져나와 기본 경로로 돌아갈 준비가 되면 True
"""
from .mission_base import MissionBase, MissionInput, MissionOutput


class ParkingMission(MissionBase):
    NAME = 'parking'
    SECTIONS = (('cp8', 'cp9'), ('cp9', 'cp10'))

    def reset(self):
        self.phase = 'idle'          # 예: 'search' -> 'align' -> 'reverse' -> 'stop' -> 'exit' -> 'done'

    def on_enter(self, section):
        self.phase = 'idle'

    def update(self, inp: MissionInput) -> MissionOutput:
        if inp.cp.section == ('cp8', 'cp9'):
            return self.parallel_parking(inp)
        return self.rear_parking(inp)

    def parallel_parking(self, inp: MissionInput) -> MissionOutput:
        # TODO(팀원 2): CP8~CP9 평행주차
        return MissionOutput()

    def rear_parking(self, inp: MissionInput) -> MissionOutput:
        # TODO(팀원 2): CP9~CP10 후면주차
        return MissionOutput()
