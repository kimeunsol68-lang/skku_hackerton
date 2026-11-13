"""장애물 회피 미션 (담당: 추후 선착순). 이 파일만 고치세요.

담당 구간
    CP1 -> CP2   장애물 1개 (course.json mission 'm_obs1', type 'obstacle1')
                 zone: x 15 ~ 45, y -3.75 ~ 3.75 (직선 구간, 동쪽으로 진행). 위치는 시도마다 바뀜(seed).
    CP6 -> CP7   장애물 3개 (course.json mission 'm_obs3', type 'obstacle3')
                 zone: x 12.8 ~ 58.8, y 150.0 ~ 157.5 (직선 구간, 서쪽으로 진행).
    정확한 값은 inp.info['zone'] (다각형 꼭짓점 [[x, y], ...])으로 받습니다.

입력 inp: mission_base.MissionInput
    inp.x, inp.y, inp.yaw, inp.speed   차 자세·속력 (map, 뒤차축)
    inp.local_path                     기본 경로 [(x, y, yaw), ...] ← 이것을 옆으로 밀어 회피 경로를 만들면 됨
    inp.scan / inp.obstacles           원본 LaserScan / 전처리된 장애물 점 [(x, y), ...] (map)
    inp.t                              시뮬레이션 시간 (s)
    inp.cp.section                     ('cp1', 'cp2') 또는 ('cp6', 'cp7')

출력 MissionOutput(override_path=None, override_cmd=None, is_done=False)
    override_path   회피 경로 [(x, y), ...] 또는 [(x, y, yaw), ...] (보통 이것만 쓰면 됨)
    override_cmd    급정지 등 직접 조작이 필요할 때 (throttle, brake, steer, gear)
    is_done         zone을 지나 회피가 끝나면 True
"""
from .mission_base import MissionBase, MissionInput, MissionOutput


class ObstacleMission(MissionBase):
    NAME = 'obstacle'
    SECTIONS = (('cp1', 'cp2'), ('cp6', 'cp7'))

    def reset(self):
        pass

    def on_enter(self, section):
        pass

    def update(self, inp: MissionInput) -> MissionOutput:
        # TODO: inp.obstacles(또는 inp.scan)로 장애물을 찾고 inp.local_path를 옆으로 밀어 override_path로 내기
        return MissionOutput()
