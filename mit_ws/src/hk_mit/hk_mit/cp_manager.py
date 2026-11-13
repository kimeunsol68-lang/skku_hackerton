"""체크포인트(CP) 구간 판별 도우미 (공용 모듈, 담당: 기본 주행·제어).

course.json의 체크포인트 게이트(a-b 선분)와 route.csv(전역 경로)를 읽어, 차의 지금 위치(x, y, yaw)가
어느 CP 구간에 있는지 알려 줍니다. ROS에 의존하지 않으므로 노트북에서 python으로 바로 시험해 볼 수 있습니다.

    cp = CPManager('course/course.json', 'course/route.csv')
    cp.reset(0)                         # 시작(0) 또는 복귀한 체크포인트 번호 (SimEvents.on_reset의 checkpoint)
    s = cp.update(x, y, yaw)            # 매 주기 호출
    s.section          # ('cp8', 'cp9')  ← 미션 모듈의 SECTIONS와 비교하는 값
    s.prev_cp, s.next_cp   # 'start' ~ 'cp11', 'cp1' ~ 'finish'
    s.passed           # 지나온 체크포인트 수 (0 = 아직 cp1 전)
    s.dist_to_next_m   # 다음 체크포인트 게이트까지 경로를 따라 남은 거리 (m)
    s.missions         # 이 구간에 해당하는 course.json의 missions 항목들 (zone, slot 등)

판별 방법: 각 체크포인트 게이트의 가운데 점을 전역 경로 위의 점 번호로 바꿔 두고, 차와 가장 가까운 경로 점
번호가 그 번호를 넘으면 그 게이트를 지난 것으로 봅니다. 차 위치는 localization의 뒤차축 위치이므로
실제 게이트 통과 판정과 ~1 m 정도 차이가 날 수 있습니다. 경로 점은 시작·복귀 직후에만 경로 전체에서
(진행 방향이 반대인 점은 제외) 찾고, 그 뒤로는 앞쪽 40 m에서만 찾아 구간이 뒤로 돌아가지 않습니다
(주차처럼 후진해도 구간이 바뀌지 않음).
"""
import csv
import json
import math
from dataclasses import dataclass, field

SEARCH_AHEAD_PTS = 80          # 0.5 m 간격 경로에서 앞쪽 40 m
HEADING_GATE_RAD = math.pi / 2 # 전체 탐색 때 진행 방향이 이보다 많이 다른 경로 점은 후보에서 뺌


@dataclass
class CPState:
    index: int                 # 차와 가장 가까운 전역 경로 점 번호
    passed: int                # 지나온 체크포인트 수 (0 ~ 체크포인트 수)
    prev_cp: str               # 마지막으로 지난 체크포인트 id ('start' = 아직 cp1 전)
    next_cp: str               # 다음 체크포인트 id ('finish' = 마지막 체크포인트 뒤)
    dist_to_next_m: float      # 다음 게이트까지 경로를 따라 남은 거리 (m)
    progress_m: float          # 경로 시작부터 지금까지의 거리 (m)
    missions: list = field(default_factory=list)   # 이 구간의 course.json missions 항목

    @property
    def section(self):
        """('cp8', 'cp9')처럼 (지난 CP, 다음 CP). 미션 모듈의 SECTIONS와 같은 형식."""
        return (self.prev_cp, self.next_cp)


class CPManager:
    def __init__(self, course_json, route_csv):
        with open(course_json, encoding='utf-8') as f:
            course = json.load(f)['course']
        with open(route_csv, encoding='utf-8') as f:
            self.points = [(float(r['x']), float(r['y'])) for r in csv.DictReader(f)]
        if len(self.points) < 2:
            raise ValueError(f'route has too few points: {route_csv}')
        self.cp_ids = [cp['id'] for cp in course['checkpoints']]
        self.missions = course.get('missions', [])
        # 경로를 따라 잰 누적 거리
        self.s = [0.0]
        for (ax, ay), (bx, by) in zip(self.points, self.points[1:]):
            self.s.append(self.s[-1] + math.hypot(bx - ax, by - ay))
        # 각 게이트 가운데 점 -> 경로 점 번호 (앞 게이트 번호부터 앞으로만 찾아 순서가 뒤집히지 않게)
        self.cp_idx = []
        start = 0
        for cp in course['checkpoints']:
            mx = (cp['a'][0] + cp['b'][0]) / 2
            my = (cp['a'][1] + cp['b'][1]) / 2
            i = min(range(start, len(self.points)),
                    key=lambda k: math.hypot(self.points[k][0] - mx, self.points[k][1] - my))
            self.cp_idx.append(i)
            start = i
        self.reset(0)

    # ---- 시작·복귀 -------------------------------------------------------------------------------------
    def reset(self, checkpoint=0):
        """시작(0)·체크포인트 n 복귀. n > 0이면 그 게이트 위치부터 앞으로 찾습니다."""
        if 0 < checkpoint <= len(self.cp_idx):
            self.idx = self.cp_idx[checkpoint - 1]
        else:
            self.idx = None

    # ---- 매 주기 ---------------------------------------------------------------------------------------
    def update(self, x, y, yaw):
        self.idx = self._nearest(x, y, yaw)
        passed = sum(1 for i in self.cp_idx if self.idx >= i)
        prev_cp = 'start' if passed == 0 else self.cp_ids[passed - 1]
        next_cp = 'finish' if passed >= len(self.cp_ids) else self.cp_ids[passed]
        if passed < len(self.cp_idx):
            dist = self.s[self.cp_idx[passed]] - self.s[self.idx]
        else:
            dist = self.s[-1] - self.s[self.idx]
        missions = [m for m in self.missions if m.get('entry_cp') == prev_cp and m.get('exit_cp') == next_cp]
        return CPState(self.idx, passed, prev_cp, next_cp, max(0.0, dist), self.s[self.idx], missions)

    def _yaw_at(self, i):
        j = max(0, min(i, len(self.points) - 2))
        (ax, ay), (bx, by) = self.points[j], self.points[j + 1]
        return math.atan2(by - ay, bx - ax)

    def _nearest(self, x, y, yaw):
        pts = self.points
        if self.idx is None:                      # 전체 탐색: 진행 방향이 맞는 점 중 가장 가까운 점
            cands = [i for i in range(len(pts))
                     if abs(_angle_diff(self._yaw_at(i), yaw)) < HEADING_GATE_RAD] or range(len(pts))
        else:                                     # 지난번 점부터 앞쪽 40 m
            cands = range(self.idx, min(self.idx + SEARCH_AHEAD_PTS, len(pts)))
        return min(cands, key=lambda i: math.hypot(pts[i][0] - x, pts[i][1] - y))


def _angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi
