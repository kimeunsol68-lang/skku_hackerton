"""미션 스위치 (공용 모듈, 담당: 기본 주행·제어).

CPManager로 지금 CP 구간을 구하고, 그 구간을 SECTIONS에 가진 미션의 update()를 불러 결과를 돌려줍니다.
미션이 없거나, 모두 None을 돌려주거나, 예외가 나면 빈 MissionOutput(= 기본 주행)을 돌려줍니다.
여러 미션이 같은 구간을 맡으면 MISSIONS 순서가 앞인 것이 우선합니다.
"""
import math
import traceback

from .mission_base import GEARS, MissionInput, MissionOutput
from .mission_obstacle import ObstacleMission
from .mission_parking import ParkingMission
from .mission_rotary import RotaryMission

MISSIONS = (ParkingMission, ObstacleMission, RotaryMission)
ERROR_LOG_PERIOD_S = 1.0


class MissionManager:
    def __init__(self, cp_manager, logger=None):
        self.cp = cp_manager
        self.log = logger
        self.missions = [cls(logger) for cls in MISSIONS]
        self.checkpoint = 0
        self.reset(0)

    def reset(self, checkpoint=0):
        """시작·복귀: CP 판별과 모든 미션을 처음으로."""
        self.checkpoint = checkpoint
        self.cp.reset(checkpoint)
        self.section = None
        self.done = set()
        self.last_error = {}
        for m in self.missions:
            self._safe(m, 'reset', 0.0, m.reset)

    def step(self, x, y, yaw, speed, local_path, scan, obstacles, t):
        """(MissionOutput, 실행한 미션 이름 또는 None, CPState)."""
        state = self.cp.update(x, y, yaw)
        sec = state.section
        if sec != self.section:
            self.section = sec
            self.done = set()                      # 새 구간에 들어오면 완료 기록을 지움
            if self.log:
                self.log.info(f'CP 구간 {sec[0]} -> {sec[1]}')
            for m in self.missions:
                if sec in m.SECTIONS:
                    self._safe(m, 'on_enter', t, m.on_enter, sec)
        info = state.missions[0] if state.missions else {}
        inp = None
        for m in self.missions:
            if sec not in m.SECTIONS or (m.NAME, sec) in self.done:
                continue
            if inp is None:
                inp = MissionInput(x, y, yaw, speed, local_path, scan, obstacles, t, state, info)
            out = self._safe(m, 'update', t, m.update, inp)
            if not isinstance(out, MissionOutput):
                continue
            out = MissionOutput(self._check_path(m, out.override_path, t),
                                self._check_cmd(m, out.override_cmd, t), bool(out.is_done))
            if out.is_done:
                self.done.add((m.NAME, sec))
                if self.log:
                    self.log.info(f'미션 {m.NAME} 완료 ({sec[0]} -> {sec[1]})')
            if out.override_path is not None or out.override_cmd is not None:
                return out, m.NAME, state
        return MissionOutput(), None, state

    # ---- 안전장치 --------------------------------------------------------------------------------------
    def _safe(self, m, what, t, fn, *args):
        try:
            return fn(*args)
        except Exception:                          # 팀원 코드의 예외로 주행이 멈추지 않게
            self._error(m, t, f'{what}() 예외\n{traceback.format_exc()}')
            return None

    def _error(self, m, t, msg):
        if t - self.last_error.get(m.NAME, -1e9) >= ERROR_LOG_PERIOD_S:
            self.last_error[m.NAME] = t
            if self.log:
                self.log.error(f'[{m.NAME}] {msg}')

    def _check_path(self, m, path, t):
        if path is None:
            return None
        try:
            pts = [tuple(float(v) for v in p) for p in path]
            if len(pts) >= 2 and all(len(p) in (2, 3) and all(math.isfinite(v) for v in p) for p in pts):
                return pts
        except (TypeError, ValueError):
            pass
        self._error(m, t, 'override_path 형식 오류: [(x, y), ...] 또는 [(x, y, yaw), ...], 점 2개 이상')
        return None

    def _check_cmd(self, m, cmd, t):
        if cmd is None:
            return None
        try:
            throttle, brake, steer, gear = cmd
            throttle, brake, steer = float(throttle), float(brake), float(steer)
            if gear in GEARS and all(math.isfinite(v) for v in (throttle, brake, steer)):
                return (min(max(throttle, 0.0), 1.0), min(max(brake, 0.0), 1.0), min(max(steer, -1.0), 1.0), gear)
        except (TypeError, ValueError):
            pass
        self._error(m, t, "override_cmd 형식 오류: (throttle, brake, steer, gear), gear는 'D'|'R'|'N'|'P'")
        return None
