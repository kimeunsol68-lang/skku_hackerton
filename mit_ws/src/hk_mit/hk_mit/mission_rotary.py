"""로터리 미션 (담당: 추후 선착순). 이 파일만 고치세요.

담당 구간
    CP5 -> CP6   로터리 (course.json mission 'm_round', type 'roundabout')
                 로터리 중심 (108.84, 118.78), NPC 차량이 반지름 11.0 m로 돕니다.
                 NPC와 부딪히지 않게 진입 전에 멈췄다가 틈이 나면 들어가는 타이밍 제어가 목표.
    정확한 값은 inp.info['center'], inp.info['npc_radius']로 받습니다.

입력 inp: mission_base.MissionInput
    inp.x, inp.y, inp.yaw, inp.speed   차 자세·속력 (map, 뒤차축)
    inp.local_path                     기본 경로 [(x, y, yaw), ...]
    inp.scan / inp.obstacles           원본 LaserScan / 전처리된 장애물 점 [(x, y), ...] (map) ← NPC 감지
    inp.t                              시뮬레이션 시간 (s) ← NPC 속도 추정에 사용
    inp.cp.dist_to_next_m              CP6까지 남은 거리 (m)

출력 MissionOutput(override_path=None, override_cmd=None, is_done=False)
    override_path   보통 None (전역 경로가 로터리를 따라감)
    override_cmd    진입 대기 정지 = (0.0, 1.0, 0.0, 'D'). 출발해도 되면 None을 돌려 평소 제어로 복귀
    is_done         로터리에 안전하게 진입했으면 True
"""
from .mission_base import MissionBase, MissionInput, MissionOutput


class RotaryMission(MissionBase):
    NAME = 'rotary'
    SECTIONS = (('cp5', 'cp6'),)

    def reset(self):
        pass

    def on_enter(self, section):
        pass

    def update(self, inp: MissionInput) -> MissionOutput:
        # TODO: 진입 지점 전에서 NPC가 가까우면 정지 명령, 틈이 나면 None(평소 주행)
        return MissionOutput()
