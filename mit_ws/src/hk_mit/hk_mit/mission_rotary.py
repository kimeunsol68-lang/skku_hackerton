"""로터리 미션 (담당: 기본 주행·제어 + 로터리). 이 파일만 고치세요.

담당 구간
    CP5 -> CP6   로터리 (course.json mission 'm_round', type 'roundabout')
                 중심 (108.84, 118.78), NPC 차량 3대가 반지름 11.0 m로 반시계 방향으로 계속 돕니다.
                 우리 경로는 동쪽에서 중심 쪽으로 들어와 원(반지름 11 m) 위를 약 76°(14°~90°)만 반시계로 돌고 북쪽으로 나갑니다.
    정확한 값은 inp.info['center'], inp.info['npc_radius']로 받습니다 (좌표를 코드에 쓰지 않음).

동작 (20 Hz)
    1. 기하 계산(처음 한 번): course.json의 중심·반경을 초기값으로, route.csv에서 그 원 위(반경 +-0.8 m)의 경로 점으로 원을 다시 맞춰
       (cx, cy, R)을 정하고(안 맞으면 course.json 값 그대로), 경로가 원 반경 R+1.0 m 안으로 들어오는 첫 점을 진입선 s_entry,
       나가는 점을 s_exit, 진입선의 원 각도를 theta_entry로 구합니다.
    2. 접근·정지: 앞 범퍼(s_ego + 3.845)가 정지선(s_entry - 3.0 m)에 닿기 전부터 속도 한계(감속 1.5 m/s^2 곡선)를 경로에 붙여 줄이고,
       범퍼가 정지선에 닿으면 override_cmd = (0, 1, steer, 'D')로 완전히 정지해 기다립니다.
       (처음 지시한 s_entry - 1.5 m는 시뮬레이터 NPC 모델로 재 보니 정지한 차 앞 범퍼와 지나가는 NPC 사이가 0.77 m뿐이고, 정지
        오차가 조금만 있어도 닿아서 3.0 m 앞으로 물렸습니다.)
    3. NPC 감지: LaserScan(lidar_link)을 차 자세와 장착 위치(뒤차축 기준 앞 1.15 m)로 map 좌표로 바꾸고, 원의 차선 밴드
       (반지름 R-1.8 ~ R+1.8 m) 안의 점을 NPC로 봅니다 (중앙 섬·외곽 벽은 반경 필터로 제외).
       점마다 진입선 기준 호 길이 dθ·R을 구합니다 (음수 = 상류, 다가오는 쪽). 위험 조건:
         - 상류: 가장 가까운 NPC까지 호 길이가 u_safe 안 (u_safe = 측정한 NPC 속도 x 3.7 s, 9~15 m로 제한. 속도를 못 재면 3.5 m/s로 가정
           해서 약 13 m)
         - 하류: 진입선 뒤 1.5 m 안에 NPC 점이 있음 (아직 진입선을 막고 있는 앞차)
       (처음 지시한 "상류 13 m + 하류 5 m"는 NPC 3대(간격 23 m, 차체 4.6 m)에서는 두 구간이 동시에 비는 순간이 0.4 m(0.1 s)뿐이라
        영원히 못 들어가서, 하류를 1.5 m로 줄이고 상류를 NPC 속도에 맞췄습니다.)
       정지해서 기다리는 동안 3 s 이상 같은 자리에 계속 잡히는 점(벽 등 정지물)은 NPC에서 제외합니다.
       위험 구간이 새 LiDAR 프레임으로 연속 2번 비면 self.entered = True로 바꾸고 출발합니다. 한 번 출발하면 다시 멈추지 않습니다.
    4. 진입 후: 경로(override_path)에 속도 한계 5.75 m/s를 붙여 원을 따라 빠르게 빠져나갑니다. 내 전방 경로 위 6.5 m 안에
       LiDAR 점(앞차)이 있으면 속도 한계를 2.5 m/s로 낮춥니다(없어진 뒤 0.5 s 유지). 원을 빠져나가면 is_done = True.

입력 inp: mission_base.MissionInput (inp.scan = LaserScan, inp.cp.index = 전역 경로에서 차의 위치 번호, inp.info = m_round 항목)
출력 MissionOutput
    override_path   [(x, y, yaw, v_limit), ...]  접근·진입 중 (planner가 Path.pose.position.z에 실어 control.py로 전달)
    override_cmd    정지선에서 대기하는 동안 (0.0, 1.0, steer, 'D'). 출발하면 None
    is_done         원을 빠져나가면 True
"""
import math
from collections import Counter, deque

import numpy as np

from .mission_base import MissionBase, MissionInput, MissionOutput

# ---- 차량·센서 (IONIQ 5) -------------------------------------------------------------------------
LIDAR_X, LIDAR_Y, LIDAR_YAW = 1.15, 0.0, 0.0     # 뒤차축 기준 (config/sensor_mounts.yaml에서 읽고, 못 읽으면 이 값)
FRONT_OVERHANG = 3.845                           # 뒤차축에서 앞 범퍼까지 (m)
WHEELBASE = 3.0
MAX_STEER = math.radians(35.4)
# ---- 정지선 ------------------------------------------------------------------------------------
STOP_NEAR = 3.0         # 앞 범퍼가 s_entry - 3.0 m에 닿으면 완전 정지. 시뮬레이터 NPC 모델로 잰 값: 1.5 m 앞이면 지나가는 NPC와 0.77 m,
                        # 0.3 m 앞이면 닿음. 정지 오차를 감안해 1 m 이상 여유(3.0 m 앞이면 2.27 m)를 둠
A_DECEL = 2.5           # 접근 감속 곡선 (m/s^2). 2.0 -> 2.5: cp5 -> 정지선 42 m에 11 s가 걸려 단축
V_FLOOR = 1.0           # 접근 속도 한계의 하한 (m/s). 이 속도로 정지 구간까지 들어간 뒤 브레이크
STOPPED_SPEED = 0.3     # 이보다 느리면 멈춘 것으로 봄 (m/s)
APPROACH_VIEW_M = 60.0  # 정지선까지 이만큼 이내가 되면 접근 속도 한계를 내보냄 (m). cp5 직후부터 적용
APPROACH_VMAX = 7.5     # 접근 최고 속도 (m/s). 경사 헤어핀에서 생긴 옆 오차가 11 m/s 직선에서 수렴하지 못해 R=15 커브에서 벽 접촉(seed 7)
# ---- 원·NPC ------------------------------------------------------------------------------------
BAND = 1.8              # 회전 차선 밴드: R +- 1.8 m
FIT_BAND = 0.8          # 원 맞춤에 쓰는 경로 점 밴드: R +- 0.8 m
MERGE_BAND = 1.0        # 경로가 R + 이만큼 안으로 들어오면 원에 합류한 것으로 봄
UPSTREAM_SCAN_M = 30.0  # NPC 속도를 재기 위해 진입선 상류 이 호 길이까지 봄 (m)
DOWNSTREAM_M = 1.5      # 진입선 하류 이 호 길이 안의 NPC(아직 진입선을 막고 있는 앞차) = 위험
T_CLEAR = 3.9           # 3.7 -> 3.9 (4.2 waited a whole NPC lap on seed 1111111). 정지 상태에서 출발해 합류를 마치는 데 필요한 시간 + 여유 (s). 상류 NPC가 이 시간 안에 도착하면 위험
U_SAFE_MIN, U_SAFE_MAX = 9.0, 16.0   # 상류 안전 호 길이 범위 (m) = clamp(NPC 속도 x T_CLEAR)
V_NPC_DEFAULT = 3.5     # NPC 속도를 못 쟀을 때 가정 (m/s). 시뮬레이터 설정은 3.0 m/s x 시드 배율 0.9~1.3 = 2.7~3.9 m/s
MIN_DANGER_POINTS = 2   # 위험 구간 안의 점이 이 개수 이상이어야 NPC로 봄 (낱개 노이즈 무시)
CLEAR_FRAMES = 2        # 위험 구간이 이만큼 연속 새 프레임 동안 비면 출발
STATIC_FRAMES = 30      # 정지물 판정에 쓰는 최근 프레임 수 (10 Hz에서 3 s)
STATIC_FRAC = 0.9       # 이 비율 이상의 프레임에서 같은 칸에 계속 잡히면 정지물(벽 등)로 보고 제외
CELL_M = 0.5            # 정지물 판정용 격자 크기 (m)
MAX_RANGE_M = 40.0
WAIT_FAILSAFE_S = 20.0  # diagnostic timeout only; never enter an occupied merge
LOG_PERIOD_S = 1.0      # 대기 중 상태 로그 간격 (s)
# ---- 정지 없이 합류 (rolling entry) ---------------------------------------------------------------
ROLLING_ENTRY = True    # 정지 위치 ROLL_DECIDE_M 안에서 "도착 시점에 위험 구간이 빈다"가 연속 확인되면 멈추지 않고 진입. False면 항상 정지
ROLL_DECIDE_M = 10.0    # 정지 위치(뒤차축)까지 이 거리 안에서만 판단 (m): 예측 시간이 짧아 NPC 위치 예측 오차가 작음
ROLL_CLEAR_FRAMES = 3   # 연속 새 LiDAR 프레임 수
ROLL_V_NPC = (2.7, 4.2) # NPC 속도 가정 범위 (m/s): 시뮬레이터 2.7~3.9 + 위쪽 여유. 도착 시점 NPC 위치를 이 범위 전체로 봄
ROLL_UP_SAFE_M = 16.0  # 11 -> 16: 무정지 합류 NPC 충돌 2회(seed 0).   # 도착 시점에 진입선 상류 이 호 길이 안에 NPC가 있으면 위험 (m). 달리는 채로 합류하면 정지 출발(3.7 s)보다 빨리 끝나므로
                        # 정지 대기 기준(9~15 m)의 중간값. NPC 4.2 m/s 기준 2.6 s 여유
ROLL_DOWN_CLEAR_M = 6.0 # 도착 시점에 진입선 하류 이 호 길이 안에 NPC가 있으면 위험 (바로 앞차 뒤에 끼어들지 않음)
                        # 주의: 상류+하류 금지 구간이 NPC 간격(약 23 m)보다 길면 영원히 조건이 안 맞음. 현재 약 18 m
ROLL_LATENCY_S = 0.3    # 도착 시간에 더하는 여유 (조향·구동 지연)
# ---- 진입 후 -----------------------------------------------------------------------------------
GO_SPEED = 5.75         # 진입 후 속도 한계 (m/s): 5.5 ~ 6.0
EXIT_SPEED = 5.0        # 출구 뒤 R~15 m 커브 속도 한계 (m/s). 7.9 m/s + GNSS 오차 0.5 m로 벽에 닿은 기록(seed 1111111)
EXIT_CAP_M = 27.0       # 출구(s_exit)에서 이만큼 지날 때까지 EXIT_SPEED 유지 (커브 끝 s~370)
FOLLOW_SPEED = 2.5      # 앞차를 따라잡았을 때 최저 속도 한계 (m/s). 실제는 측정한 NPC 속도 - 0.2 (최대 4.0)
FOLLOW_AHEAD_M = 6.5    # 내 앞 범퍼 기준 이만큼 앞 경로 위에 점이 있으면 앞차로 봄 (m)
FOLLOW_HALF_W = 1.8     # 경로 중심선에서 이 안쪽 점만 (차 반폭 0.95 + 여유, m)
FOLLOW_HOLD_S = 0.5     # 앞차가 안 보이게 된 뒤에도 느린 속도를 유지하는 시간 (s)
SEARCH_BEHIND, SEARCH_AHEAD = 12, 120   # 전역 경로에서 찾는 범위 (점 번호, 0.5 m 간격)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class RotaryMission(MissionBase):
    NAME = 'rotary'
    SECTIONS = (('cp5', 'cp6'),)

    def __init__(self, logger=None):
        self.pts = None            # 전역 경로 (N, 2), 처음 쓸 때 읽음
        self.geo = None
        self.lidar = (LIDAR_X, LIDAR_Y, LIDAR_YAW)
        super().__init__(logger)

    def reset(self):
        self.npc_v = V_NPC_DEFAULT
        self.enabled = None        # 이번 시도에서 로터리 옵션이 켜졌는가 (처음 update에서 읽음)
        self.entered = False       # 출발 결정을 내렸는가 (한 번 True가 되면 다시 멈추지 않음)
        self.clear_frames = 0
        self.last_stamp = None
        self.wait_since = None
        self.follow_until = -1.0
        self.logged_stop = False
        self.npc_hist = deque(maxlen=8)          # (시각, 진입선까지 가장 가까운 상류 NPC 호 길이)
        self.cell_hist = deque(maxlen=STATIC_FRAMES)   # 최근 프레임들의 회전 차선 밴드 안 점이 잡힌 격자 칸
        self.last_log = -1e9
        self.timeout_warned = False
        self.roll_clear = 0        # 정지 없이 합류: 도착 시점 위험 구간이 비어 있던 연속 프레임 수

    def on_enter(self, section):
        self.reset()

    # ---- 준비 ---------------------------------------------------------------------------------------
    def _load(self):
        from .course import COURSE_DIR, Route
        self.pts = np.asarray(Route(str(COURSE_DIR / 'route.csv')).points, dtype=float)
        d = np.diff(self.pts, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.vstack([d, d[-1:]])
        t[1:-1] = d[1:] + d[:-1]
        self.tan = t / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        self.nor = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)
        try:                                                           # LiDAR 장착 위치
            import yaml
            with open(COURSE_DIR.parent / 'config' / 'sensor_mounts.yaml', encoding='utf-8') as f:
                m = yaml.safe_load(f)['lidar']
            self.lidar = (float(m['x']), float(m['y']), math.radians(float(m.get('yaw', 0.0))))
        except Exception:
            pass

    def _geometry(self, info):
        """원 (cx, cy, R)과 진입선·출구의 경로 누적거리·원 각도. 실패하면 None."""
        center, r_nom = info.get('center'), info.get('npc_radius')
        if not center or not r_nom:
            return None
        c0 = np.asarray(center, dtype=float)
        r = np.hypot(self.pts[:, 0] - c0[0], self.pts[:, 1] - c0[1])
        near = np.nonzero(r < r_nom + 20.0)[0]                          # 중심 근처 경로 점 (이 코스에서는 로터리 구간 하나)
        if len(near) == 0:
            return None
        near = near[near <= near[0] + 600]                              # 첫 접근 구간만
        ring = near[np.abs(r[near] - r_nom) <= FIT_BAND]
        cx, cy, R = float(c0[0]), float(c0[1]), float(r_nom)
        if len(ring) >= 12:                                              # 경로의 원 위 점으로 원을 다시 맞춤
            q = self.pts[ring]
            sol = np.linalg.lstsq(np.c_[2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))], (q ** 2).sum(1), rcond=None)[0]
            fcx, fcy = float(sol[0]), float(sol[1])
            fr = math.sqrt(max(float(sol[2]) + fcx * fcx + fcy * fcy, 0.0))
            if math.hypot(fcx - cx, fcy - cy) < 1.5 and abs(fr - R) < 1.5:
                cx, cy, R = fcx, fcy, fr
        r = np.hypot(self.pts[:, 0] - cx, self.pts[:, 1] - cy)
        inside = near[r[near] <= R + MERGE_BAND]                         # 경로가 원 안으로 들어와 있는 점들
        if len(inside) == 0:
            return None
        i_in, i_out = int(inside[0]), int(inside[-1])
        geo = dict(cx=cx, cy=cy, R=R, s_entry=float(self.s[i_in]), s_exit=float(self.s[i_out]),
                   theta_entry=math.atan2(self.pts[i_in, 1] - cy, self.pts[i_in, 0] - cx))
        if self.log:
            self.log.info(f'로터리 원 중심=({cx:.2f}, {cy:.2f}) R={R:.2f} 진입선 s={geo["s_entry"]:.1f} 출구 s={geo["s_exit"]:.1f} '
                          f'진입각={math.degrees(geo["theta_entry"]):.0f}도')
        return geo

    def _frenet(self, xy, i0, i1):
        """map 점들 (M, 2) -> (route 점 번호, s, d). 번호 범위 [i0, i1)의 가장 가까운 경로 점 기준."""
        cand = self.pts[i0:i1]
        diff = xy[:, None, :] - cand[None, :, :]
        gi = np.argmin(np.einsum('mkc,mkc->mk', diff, diff), axis=1) + i0
        rel = xy - self.pts[gi]
        return gi, self.s[gi] + np.einsum('mc,mc->m', rel, self.tan[gi]), np.einsum('mc,mc->m', rel, self.nor[gi])

    def _lidar_points(self, inp):
        """LaserScan -> map 좌표 점 (M, 2)."""
        scan = inp.scan
        if scan is None:
            return np.zeros((0, 2))
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r >= max(scan.range_min, 0.3)) & (r <= min(scan.range_max, MAX_RANGE_M))
        lx, ly, lyaw = self.lidar
        x, y, heading = inp.scan_pose or (inp.x, inp.y, inp.yaw)
        ox = x + math.cos(heading) * lx - math.sin(heading) * ly
        oy = y + math.sin(heading) * lx + math.cos(heading) * ly
        a = ang[ok] + heading + lyaw
        return np.stack([ox + r[ok] * np.cos(a), oy + r[ok] * np.sin(a)], axis=1)

    def _assess(self, xy, t):
        """진입선 주변의 NPC 상황. 반환 dict: danger(위험?), n(밴드 안 점), up(가장 가까운 상류 호 길이), down(하류 1.5 m 안 점 수),
        v(추정 NPC 속도), u_safe(상류 안전 호 길이), static(정지물로 제외한 점 수)."""
        g = self.geo
        out = dict(danger=False, n=0, up=None, down=0, v=None, u_safe=None, static=0)
        if len(xy):
            rr = np.hypot(xy[:, 0] - g['cx'], xy[:, 1] - g['cy'])
            pts = xy[np.abs(rr - g['R']) <= BAND]                         # 회전 차선 밴드 (중앙 섬·외곽 벽은 여기서 제외)
        else:
            pts = np.zeros((0, 2))
        cells = [tuple(c) for c in np.floor(pts / CELL_M).astype(int)] if len(pts) else []
        self.cell_hist.append(set(cells))
        static = set()
        if len(self.cell_hist) >= STATIC_FRAMES:                          # 오래 같은 칸에 잡히는 점 = 벽 같은 정지물
            counts = Counter(c for frame in self.cell_hist for c in frame)
            static = {c for c, k in counts.items() if k >= STATIC_FRAC * len(self.cell_hist)}
        if cells:
            moving = np.array([c not in static for c in cells])
            out['static'] = int((~moving).sum())
            pts = pts[moving]
        out['n'] = len(pts)
        if len(pts) == 0:
            self.npc_hist.clear()
            return out
        arc = wrap_arr(np.arctan2(pts[:, 1] - g['cy'], pts[:, 0] - g['cx']) - g['theta_entry']) * g['R']   # 음수 = 상류
        up = -arc[(arc < 0.0) & (arc >= -UPSTREAM_SCAN_M)]                # 상류 호 길이 (양수)
        down = int(((arc >= 0.0) & (arc <= DOWNSTREAM_M)).sum())
        out['down'] = down
        if len(up):
            out['up'] = float(up.min())
            self.npc_hist.append((t, float(up.min())))
        # NPC 속도: 가장 가까운 상류 점이 진입선 쪽으로 다가오는 속도의 중앙값 (다음 NPC로 넘어가는 순간의 튐은 버림)
        speeds = []
        hist = list(self.npc_hist)
        for (t0, u0), (t1, u1) in zip(hist, hist[1:]):
            if 0.05 < t1 - t0 < 0.5 and 0.0 < u0 - u1 < 3.0:
                speeds.append((u0 - u1) / (t1 - t0))
        v = float(np.median(speeds)) if len(speeds) >= 2 else V_NPC_DEFAULT
        v = min(max(v, 2.0), 8.0)
        u_safe = min(max(v * getattr(self, 't_clear', T_CLEAR), U_SAFE_MIN), U_SAFE_MAX)
        out['v'], out['u_safe'] = v, u_safe
        if len(speeds) >= 2:
            self.npc_v = v                                                 # 원 안에서 앞 NPC를 따라갈 속도
        near_up = int((up < u_safe).sum())
        out['danger'] = near_up >= MIN_DANGER_POINTS or down >= MIN_DANGER_POINTS
        return out

    def _assess_rolling(self, xy, t_arr):
        """정지하지 않고 진입할 때: 도착 시점(t_arr초 뒤)에 위험 구간 [-u_safe, ROLL_DOWN_CLEAR_M]에 들어올 수 있는 NPC 점 수.
        NPC 속도를 ROLL_V_NPC 범위로 가정해, 각 점이 도착 시점에 있을 수 있는 호 길이 구간 전체가 위험 구간과 겹치는지 봅니다."""
        g = self.geo
        if len(xy) == 0:
            return 0
        rr = np.hypot(xy[:, 0] - g['cx'], xy[:, 1] - g['cy'])
        pts = xy[np.abs(rr - g['R']) <= BAND]
        if len(pts) == 0:
            return 0
        arc = wrap_arr(np.arctan2(pts[:, 1] - g['cy'], pts[:, 0] - g['cx']) - g['theta_entry']) * g['R']
        lo, hi = arc + ROLL_V_NPC[0] * t_arr, arc + ROLL_V_NPC[1] * t_arr
        return int(((hi >= -ROLL_UP_SAFE_M) & (lo <= getattr(self, 'roll_down_m', ROLL_DOWN_CLEAR_M))).sum())

    @staticmethod
    def _fmt(a):
        up = '없음' if a['up'] is None else f"{a['up']:.1f}m"
        v = '-' if a['v'] is None else f"{a['v']:.1f}m/s"
        us = '-' if a['u_safe'] is None else f"{a['u_safe']:.1f}m"
        return (f"밴드 안 점 {a['n']}개(정지물 제외 {a['static']}개), 상류 최근접 {up} (안전 {us}), 하류 1.5m 안 {a['down']}점, "
                f"NPC 속도 추정 {v} -> {'위험' if a['danger'] else '비어 있음'}")

    def _steer(self, inp):
        """정지 중 바퀴를 경로 방향으로 두기 위한 Stanley 조향 (-1~1)."""
        path = inp.local_path
        if not path:
            return 0.0
        fx = inp.x + WHEELBASE * math.cos(inp.yaw)
        fy = inp.y + WHEELBASE * math.sin(inp.yaw)
        px, py, pyaw = min(((p[0], p[1], p[2]) for p in path), key=lambda p: math.hypot(p[0] - fx, p[1] - fy))
        heading = wrap(pyaw - inp.yaw)
        cross = -math.sin(pyaw) * (px - fx) + math.cos(pyaw) * (py - fy)
        wheel = heading + math.atan2(cross, 1.0 + abs(inp.speed))
        return max(-1.0, min(1.0, wheel / MAX_STEER))

    # ---- 메인 ---------------------------------------------------------------------------------------
    def update(self, inp: MissionInput) -> MissionOutput:
        if self.enabled is None:                        # 옵션이 꺼져 있으면 NPC가 없으니 정지·대기하지 않음
            from .course import load_options, option_on
            opts = getattr(self, 'options', None) or load_options()
            self.enabled = option_on(opts, 'roundabout')
            from .course import fast_mode
            # 공격 모드(2차 시도): 합류 여유를 줄이고 출구·접근·원 안 속도를 올림
            self.exit_cap_m = 20.0 if fast_mode(opts) else EXIT_CAP_M
            self.a_decel = 3.0 if fast_mode(opts) else A_DECEL
            # 무정지 합류(공격 모드만): 이전 충돌 3회는 5 m/s로 합류해 앞 NPC(2.7~3.5 m/s)를 추돌한 것으로 보임 ->
            # 앞 NPC가 도착 시점에 12 m 이상 앞일 때만 합류, 원 안에서는 10 m 앞까지 보고 NPC보다 0.5 m/s 느리게 따라감
            # 공격 모드: 거의 섰을 때(0.8 m/s)부터 판단하고 빈 프레임 1개로 출발 (출발 자체는 이미 풀 가속)
            self.stopped_speed, self.clear_need = (0.8, 1) if fast_mode(opts) else (STOPPED_SPEED, CLEAR_FRAMES)
            self.rolling = False                # 무정지 합류: 조건 강화 후에도 NPC 충돌(seed 7 x2) -> 끔
            self.roll_down_m = 12.0
            self.follow_ahead_m, self.follow_margin = FOLLOW_AHEAD_M, 0.2   # 무정지 합류를 껐으므로 기존 추종 (6.5 m, NPC-0.2)
            self.t_clear, self.exit_speed, self.approach_vmax, self.go_speed = \
                (3.1, 8.0, 8.0, 6.3) if fast_mode(opts) else (T_CLEAR, EXIT_SPEED, APPROACH_VMAX, GO_SPEED)
            if self.log and not self.enabled:
                self.log.info('로터리 옵션이 꺼져 있어 로터리 미션을 건너뜁니다')
        if not self.enabled:
            return MissionOutput(is_done=True)
        if self.pts is None:
            self._load()
        if self.geo is None:
            self.geo = self._geometry(inp.info)
            if self.geo is None:
                return MissionOutput()
        g = self.geo
        ego_i = inp.cp.index if inp.cp is not None else int(np.argmin(np.hypot(*(self.pts - [inp.x, inp.y]).T)))
        ego_i = min(max(ego_i, 0), len(self.pts) - 2)
        rel = np.array([inp.x, inp.y]) - self.pts[ego_i]
        s_ego = float(self.s[ego_i] + rel @ self.tan[ego_i])
        s_front = s_ego + FRONT_OVERHANG
        if s_ego > g['s_exit'] + getattr(self, 'exit_cap_m', EXIT_CAP_M):                            # 원과 출구 커브를 모두 빠져나감
            return MissionOutput(is_done=True)
        if not inp.local_path:
            return MissionOutput()
        if s_ego > g['s_exit'] + 2.0:                                   # 원은 빠져나감: 출구 R=15 커브는 EXIT_SPEED로
            return MissionOutput(override_path=self._with_limits(inp.local_path, lambda s_p: self.exit_speed))

        xy = self._lidar_points(inp)
        stamp = None if inp.scan is None else (inp.scan.header.stamp.sec, inp.scan.header.stamp.nanosec)
        new_frame = stamp is not None and stamp != self.last_stamp
        if new_frame:
            self.last_stamp = stamp

        if not self.entered:
            return self._approach(inp, xy, new_frame, s_ego, s_front)
        return self._inside(inp, xy, s_front)

    # ---- 접근·정지·대기 -----------------------------------------------------------------------------
    def _approach(self, inp, xy, new_frame, s_ego, s_front):
        g = self.geo
        stop_front = g['s_entry'] - STOP_NEAR                            # 앞 범퍼가 여기 닿으면 정지
        stop_rear = stop_front - FRONT_OVERHANG                           # 그때 뒤차축 위치
        if stop_rear - s_ego > APPROACH_VIEW_M:
            return MissionOutput()
        at_line = s_front >= stop_front
        stopped = abs(inp.speed) < getattr(self, 'stopped_speed', STOPPED_SPEED)
        if at_line and stopped:                                          # 정지선에서 기다리는 중: 위험 구간이 비는지 본다
            if self.wait_since is None:
                self.wait_since = inp.t
            if new_frame:
                a = self._assess(xy, inp.t)
                self.clear_frames = 0 if a['danger'] else self.clear_frames + 1
                if self.log and inp.t - self.last_log >= LOG_PERIOD_S:
                    self.last_log = inp.t
                    self.log.info(f'로터리 대기 {inp.t - self.wait_since:.0f}s: ' + self._fmt(a))
            if inp.t - self.wait_since > WAIT_FAILSAFE_S and not self.timeout_warned:
                self.timeout_warned = True
                if self.log:
                    self.log.warn('로터리 대기 장기화: 위험 구간이 비었다는 새 관측이 있어야 출발합니다')
            if self.clear_frames >= getattr(self, 'clear_need', CLEAR_FRAMES) and new_frame:
                self.entered = True
                if self.log:
                    self.log.info('로터리 빈 공간 확인: 진입')
                return self._inside(inp, xy, s_front)
        else:
            self.cell_hist.clear()                                       # 움직이는 동안의 프레임은 정지물 판정에 섞지 않음
            self.npc_hist.clear()
            if not at_line:
                self.wait_since = None
                self.clear_frames = 0
            dist = stop_rear - s_ego
            in_window = ROLLING_ENTRY and getattr(self, 'rolling', False) and not at_line and 0.0 < dist <= ROLL_DECIDE_M and inp.speed > 1.0
            if in_window and not new_frame:
                pass                                                     # 20 Hz 중 새 스캔 없는 주기: 카운터 유지 (예전엔 0으로 지워 무정지 합류가 불가능했음)
            elif in_window:
                t_arr = dist / max(inp.speed, 1.0) + ROLL_LATENCY_S      # 정지 위치 도착까지 걸릴 시간
                n_danger = self._assess_rolling(xy, t_arr)
                self.roll_clear = 0 if n_danger >= MIN_DANGER_POINTS else self.roll_clear + 1
                if self.roll_clear >= ROLL_CLEAR_FRAMES:                 # 도착 시점에 비어 있음이 연속 확인됨: 멈추지 않고 합류
                    self.entered = True
                    if self.log:
                        self.log.info(f'로터리 진입: 도착 {t_arr:.1f}s 뒤 위험 구간 비어 있음 ({self.roll_clear}프레임 연속) - 정지 없이 합류')
                    return self._inside(inp, xy, s_front)
            else:
                self.roll_clear = 0

        # 접근 속도 한계: 정지 위치까지 남은 거리에서 A_DECEL로 멈출 수 있는 속도 (V_FLOOR 이상)
        path = self._with_limits(inp.local_path, lambda s_p: min(self.approach_vmax, max(math.sqrt(2.0 * getattr(self, 'a_decel', A_DECEL) * max(stop_rear - s_p, 0.0)), V_FLOOR)))
        cmd = (0.0, 1.0, self._steer(inp), 'D') if at_line else None
        if cmd is not None and not self.logged_stop and self.log:
            self.log.info(f'로터리 정지선 도착 (앞 범퍼 s={s_front:.1f}, 정지 기준 {stop_front:.1f}): 정지하고 대기')
            self.logged_stop = True
        return MissionOutput(override_path=path, override_cmd=cmd)

    # ---- 진입 후 ------------------------------------------------------------------------------------
    def _inside(self, inp, xy, s_front):
        """원을 따라 빠르게 빠져나감. 앞 6.5 m 경로 위에 점이 있으면 느리게."""
        v = self.go_speed
        if len(xy):
            ego_i = inp.cp.index if inp.cp is not None else 0
            gi, s_p, d_p = self._frenet(xy, max(0, ego_i - SEARCH_BEHIND), min(len(self.pts), ego_i + SEARCH_AHEAD))
            ahead = (s_p > s_front) & (s_p <= s_front + getattr(self, 'follow_ahead_m', FOLLOW_AHEAD_M)) & (np.abs(d_p) <= FOLLOW_HALF_W)
            if int(ahead.sum()) >= MIN_DANGER_POINTS:
                self.follow_until = inp.t + FOLLOW_HOLD_S
        if inp.t < self.follow_until:
            v = min(max(FOLLOW_SPEED, self.npc_v - getattr(self, 'follow_margin', 0.2)), 4.0)              # NPC와 같은 속도: 뒤 NPC가 추돌하지 않게
        return MissionOutput(override_path=self._with_limits(inp.local_path, lambda s_p: v))

    # ---- 경로에 속도 한계 붙이기 ------------------------------------------------------------------------
    def _with_limits(self, local_path, limit_fn):
        """local_path의 각 점에 limit_fn(그 점의 경로 누적거리 s)를 4번째 값(속도 한계 m/s)으로 붙입니다."""
        lp = np.asarray([p[:2] for p in local_path], dtype=float)
        mid = len(lp) // 2
        i_mid = int(np.argmin(np.hypot(self.pts[:, 0] - lp[mid, 0], self.pts[:, 1] - lp[mid, 1])))
        _, s_p, _ = self._frenet(lp, max(0, i_mid - 80), min(len(self.pts), i_mid + 80))
        return [(float(p[0]), float(p[1]), float(p[2]), float(limit_fn(float(s)))) for p, s in zip(local_path, s_p)]


def wrap_arr(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
