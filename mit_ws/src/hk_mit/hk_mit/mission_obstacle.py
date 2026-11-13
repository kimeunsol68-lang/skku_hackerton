"""장애물 회피 미션 (담당: 기본 주행·제어 + 장애물). 이 파일만 고치세요.

담당 구간
    CP1 -> CP2   장애물 1개 (course.json mission 'm_obs1', type 'obstacle1', zone x 15~45, y ±3.75)
    CP6 -> CP7   장애물 3개 (course.json mission 'm_obs3', type 'obstacle3', zone x 12.8~58.8, y 150~157.5)
    장애물의 위치(진행 방향 s)와 좌/우(d)는 시도(seed)마다 바뀌므로 좌표를 하드코딩하지 않고 매 프레임 LiDAR로 찾습니다.

동작 (20 Hz): "가장 가까운 아직 안 지나친 장애물 1개씩" 순차 회피
    1. 감지: LiDAR(LaserScan, lidar_link)를 차 자세와 장착 위치(뒤차축 기준 앞 1.15 m)로 map 좌표로 바꾸고, 전역 경로(route.csv)
       기준 Frenet (s: 누적 거리, d: 중심선 기준 왼쪽 +)로 바꿉니다. |d| < 3.2 m(도로 벽 제외), s_ego-1.5 ~ s_ego+18 m, 미션 zone
       안의 점을 0.9 m 이내끼리 묶어 3점 미만은 버리고, 묶음의 중심 (s, d)와 범위를 구합니다.
    2. 추적: 같은 물체를 여러 프레임에 걸쳐 map 좌표로 추적합니다. 한 프레임이라도 잡히면 즉시 확정하고, 그때 회피 방향을 고정합니다:
       중심선 왼쪽(d >= 0, side > 0)이면 오른쪽 차선 d_target = -2.05 m, 오른쪽이면 왼쪽 차선 d_target = +2.05 m.
       다른 장애물에 가려져 점이 안 들어와도 장애물을 지나치기 전에는 추적을 지우지 않고 기억합니다(최소 2.5 s 이상).
       감지 범위는 앞 24 m라서 앞 장애물 너머 장애물의 윤곽이 조금만 보여도 바로 추적을 시작합니다.
    3. 타겟 선택: 장애물 점유 구간은 [smin, max(smax, smin + 4.3)](LiDAR는 앞면만 봐서 smax가 짧으므로 실제 길이 4.3 m로 보정).
       뒤차축 s_ego < 뒷끝 + 1.2 m(차체가 아직 완전히 지나치지 않은) 장애물만 후보이고, 그중 앞끝 s가 가장 작은 1개(A)가
       현재 타겟입니다. 지나친 장애물은 즉시 후보에서 빠져 다음 장애물로 타겟이 넘어갑니다.
    4. 경로 (모두 코사인 곡선):
       - A의 window(차체가 A와 길이 방향으로 겹치는 구간: 앞은 앞끝 - 3.845(앞 오버행) - 1.5 = 앞끝 5.3 m 전, 뒤는 뒷끝 + 0.5)에서는
         d_target 유지. window 시작 전 램프(약 6.8 m)로 중앙에서 d_target으로 이동해, 앞끝 5.3 m 전에는 이미 목표 차선에 도착합니다.
       - A 뒷끝에서 B 앞끝까지 12 m 이내에 B가 이미 보이면, A를 지나치는 순간(s > A 뒷끝 + 0.5) 중앙으로 안 돌아오고 바로 B의
         d_target으로 이어서 코사인으로 넘어갑니다(반대쪽이면 A window 끝 ~ B window 시작 사이, 그 간격이 RAMP_MIN(4.2 m)보다
         좁으면 두 지점의 중간을 중심으로 4.2 m 램프를 대칭 배치. 같은 쪽이면 차선 유지).
       - B가 아직 안 보이면 A를 지난 뒤 4.5 m에 걸쳐 0 m(중앙)로 복귀합니다.
       - 지나친 장애물도 복귀 곡선이 끝날 때까지는 경로 연속성용으로만 유지합니다(바로 지우면 경로가 중앙으로 튐).
    5. 출력: 뒤차축 뒤 2 m ~ 앞 18 m를 0.5 m 간격으로 경로 중심선에서 옆으로 민 점 (x, y, yaw, v_limit)로 냅니다.
       v_limit = 3.5 m/s. planner가 Path.pose.position.z에 실어 보내고 control.py가 목표 속도에 반영합니다.

입력 inp: mission_base.MissionInput (inp.scan = LaserScan, inp.cp.index = 전역 경로에서 차의 위치 번호)
출력 MissionOutput
    override_path   [(x, y, yaw, v_limit), ...]  장애물이 있을 때만. 없으면 None(기본 경로)
    override_cmd    None (직접 제어 안 함)
    is_done         False (구간이 끝나면 매니저가 알아서 부르지 않음)
"""
import math

import numpy as np

from .mission_base import MissionBase, MissionInput, MissionOutput
from .course import load_options, option_on
from .sensing import stamp

# ---- 차량·센서 (IONIQ 5) -------------------------------------------------------------------------
LIDAR_X, LIDAR_Y, LIDAR_YAW = 1.15, 0.0, 0.0     # 뒤차축 기준 (config/sensor_mounts.yaml에서 읽고, 못 읽으면 이 값)
FRONT_OVERHANG = 3.845                           # 뒤차축에서 앞 범퍼까지 (m) = 4.635 - 0.79
# ---- 감지·추적 ---------------------------------------------------------------------------------
D_FILTER = 2.75         # remove boundary points BEFORE clustering so they cannot join a real car
BEHIND_M = 1.5          # 차 뒤 이만큼까지 (m)
AHEAD_M = 24.0          # 차 앞 이만큼까지 (m). 앞 장애물에 가려진 다음 장애물의 윤곽이 조금이라도 보이면 바로 추적 시작
MAX_RANGE_M = 40.0      # LiDAR 점 최대 거리 (m)
ZONE_MARGIN_M = 2.0     # 미션 zone 바깥 이만큼까지 후보로 인정 (m)
CLUSTER_DIST = 0.9      # 같은 장애물로 묶는 거리 (m)
MIN_POINTS = 3          # 이보다 적은 묶음은 노이즈
ASSOC_DIST = 1.8        # 추적 중인 장애물과 같은 물체로 보는 거리 (m)
ASSOC_PAD = 0.8         # 추적 중인 장애물의 범위 안팎 이만큼 이내에 들어오면 같은 물체 (m)
NEW_MIN_AHEAD = 0.3     # 새 장애물은 차보다 이만큼 앞에서만 만듦 (지나친 장애물의 뒷면 조각이 새 장애물이 되는 것 방지)
CONFIRM_COUNT = 2       # two distinct sensor frames, never duplicate planner ticks
CULL_BEHIND = 12.0      # retain the vehicle until the longer return ramp is complete
# ---- 회피 경로 ---------------------------------------------------------------------------------
LANE_OFFSET = 2.05      # 회피 목표 오프셋 (m), 항상 ±2.05 유지
OBSTACLE_LENGTH = 4.3   # 장애물 실제 길이 (m). LiDAR는 앞면·옆면 일부만 보므로 smax가 실제 뒷끝보다 짧게 잡힘
MARGIN_FRONT = 1.5      # 앞 오버행(3.845 m) 말고 더 두는 앞쪽 여유 (m): 장애물 앞끝보다 5.3 m 전에 이미 목표 차선에 도착
PASS_MARGIN = 0.5       # 장애물 뒷면(s_max)에서 이만큼 지나면 다음 목표로 넘어가기 시작 (m)
DROP_MARGIN = 1.2       # 뒤차축이 s_max + 이만큼 지나면 후보에서 제외 (m)
NEXT_RANGE = 12.0       # A 뒷끝에서 B 앞끝까지가 이만큼(m) 이내면 B는 A에서 바로 이어서 회피
RETURN_M = 10.0         # return gently enough to keep the front corners inside the road
RAMP_MIN = 4.2          # 코사인 전환 최소 길이 (m)
KAPPA_LIM = 0.10        # about 10 m lane-change ramps, checked against the whole body
SPEED_LIMIT = 4.5       # maneuver only; ~2 m/s^2 lateral on the 0.10 1/m ramps. Was 3.2 (too slow to finish in 150 s)
PATH_BEHIND, PATH_AHEAD, PATH_STEP = 2.0, 18.0, 0.5   # override_path 범위 (뒤차축 기준 m)
SEARCH_BEHIND, SEARCH_AHEAD = 12, 100   # 전역 경로에서 찾는 범위 (점 번호, 0.5 m 간격)


class Track:
    """map 좌표에서 추적하는 장애물 하나."""

    def __init__(self, s, d, smin, smax, dmin, dmax, x, y, t):
        self.s, self.d, self.x, self.y = s, d, x, y
        self.smin, self.smax, self.dmin, self.dmax = smin, smax, dmin, dmax
        self.count, self.t, self.side = 1, t, 0
        self.offset = LANE_OFFSET

    def update(self, s, d, smin, smax, dmin, dmax, x, y, t, alpha=0.3):
        self.s += alpha * (s - self.s)
        self.d += alpha * (d - self.d)
        self.x += alpha * (x - self.x)
        self.y += alpha * (y - self.y)
        self.smin, self.smax = (min(self.smin, smin), min(max(self.smax, smax), min(self.smin, smin) + 5.5))
        self.dmin = 0.7 * self.dmin + 0.3 * dmin
        self.dmax = 0.7 * self.dmax + 0.3 * dmax
        if t > self.t + 1e-4:
            self.count += 1
        self.t = t

    @property
    def confirmed(self):
        return self.count >= CONFIRM_COUNT

    def contains(self, s, d):
        return (self.smin - ASSOC_PAD <= s <= self.smax + ASSOC_PAD) and (self.dmin - ASSOC_PAD <= d <= self.dmax + ASSOC_PAD)

    def back(self):
        """장애물 뒷끝 s. 점유 구간은 [smin, max(smax, smin + 4.3)]: 앞끝(smin)과 실제 길이 4.3 m로 보정."""
        return max(self.smax, self.smin + OBSTACLE_LENGTH)

    def lo(self):
        """이 s까지 목표 차선에 도착해 있어야 합니다: 장애물 앞끝보다 앞 오버행 + 1.5 m(약 5.3 m) 전."""
        return self.smin - FRONT_OVERHANG - MARGIN_FRONT

    def hi(self):
        """뒤차축이 이 s를 넘으면 장애물을 지나친 것으로 보고 다음 목표로 넘어가기 시작합니다."""
        return self.back() + PASS_MARGIN

    def target(self):
        """회피 목표 오프셋: 장애물이 왼쪽(side>0)이면 오른쪽(-) 차선."""
        return -self.offset * self.side


class ObstacleMission(MissionBase):
    NAME = 'obstacle'
    SECTIONS = (('cp1', 'cp2'), ('cp6', 'cp7'))

    def __init__(self, logger=None):
        self.pts = None            # 전역 경로 (N, 2), 처음 쓸 때 읽음
        self.lidar = (LIDAR_X, LIDAR_Y, LIDAR_YAW)
        super().__init__(logger)

    def reset(self):
        self.tracks = []
        self.zone_box = None
        self.last_scan_stamp = None
        self.zone_half_width = 3.75
        self.last_warning = -math.inf

    def on_enter(self, section):
        self.reset()

    # ---- 준비 ---------------------------------------------------------------------------------------
    def _load(self):
        import json
        from .course import COURSE_DIR, Route
        with open(COURSE_DIR / 'course.json', encoding='utf-8') as f:
            self.obstacle_zones = [m for m in json.load(f)['course']['missions'] if str(m.get('type', '')).startswith('obstacle')]
        self.pts = np.asarray(Route(str(COURSE_DIR / 'route.csv')).points, dtype=float)
        d = np.diff(self.pts, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.vstack([d, d[-1:]])                                    # 점마다 접선 (앞 구간 방향)
        t[1:-1] = d[1:] + d[:-1]
        self.tan = t / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        self.nor = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)   # 왼쪽 법선
        try:                                                           # LiDAR 장착 위치
            import yaml
            with open(COURSE_DIR.parent / 'config' / 'sensor_mounts.yaml', encoding='utf-8') as f:
                m = yaml.safe_load(f)['lidar']
            self.lidar = (float(m['x']), float(m['y']), math.radians(float(m.get('yaw', 0.0))))
        except Exception:
            pass

    def _frenet(self, xy, i0, i1):
        """map 점들 (M, 2) -> (route 점 번호, s, d, 창 가장자리가 아닌지). 번호 범위 [i0, i1)의 가장 가까운 경로 점 기준."""
        cand = self.pts[i0:i1]
        diff = xy[:, None, :] - cand[None, :, :]
        idx = np.argmin(np.einsum('mkc,mkc->mk', diff, diff), axis=1)
        gi = idx + i0
        rel = xy - self.pts[gi]
        s = self.s[gi] + np.einsum('mc,mc->m', rel, self.tan[gi])
        d = np.einsum('mc,mc->m', rel, self.nor[gi])
        return gi, s, d, (idx > 0) & (idx < len(cand) - 1)

    # ---- 감지 ---------------------------------------------------------------------------------------
    def _detect(self, inp, ego_i, s_ego):
        scan = inp.scan
        if scan is None:
            return []
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r >= max(scan.range_min, 0.3)) & (r <= min(scan.range_max, MAX_RANGE_M))
        if not ok.any():
            return []
        lx, ly, lyaw = self.lidar
        x, y, heading = inp.scan_pose or (inp.x, inp.y, inp.yaw)
        ox = x + math.cos(heading) * lx - math.sin(heading) * ly
        oy = y + math.sin(heading) * lx + math.cos(heading) * ly
        a = ang[ok] + heading + lyaw
        xy = np.stack([ox + r[ok] * np.cos(a), oy + r[ok] * np.sin(a)], axis=1)
        i0, i1 = max(0, ego_i - SEARCH_BEHIND), min(len(self.pts), ego_i + SEARCH_AHEAD)
        _, s, d, inner = self._frenet(xy, i0, i1)
        keep = inner & (np.abs(d) < D_FILTER) & (s >= s_ego - BEHIND_M) & (s <= s_ego + AHEAD_M)
        if self.zone_box is not None:
            x0, x1, y0, y1 = self.zone_box
            keep &= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        # A localization offset can move an entire wall inside D_FILTER. Peel
        # long narrow boundary lines BEFORE connected components; rejecting a
        # whole merged component would also discard a car beside that wall.
        for side in (-1, 1):
            edge = keep & (side*d > 2.15)
            if edge.sum() < 10 or np.ptp(s[edge]) < 8.0:
                continue
            inliers = edge.copy()
            for _ in range(3):
                if inliers.sum() < 8 or np.ptp(s[inliers]) < 8.0:
                    break
                line = np.polyfit(s[inliers], d[inliers], 1)
                inliers = edge & (np.abs(d - np.polyval(line, s)) < 0.16)
            if inliers.sum() >= 8 and np.ptp(s[inliers]) >= 8.0:
                keep &= ~((side*d > 1.8) & (np.abs(d - np.polyval(line, s)) < 0.18))
        xy, s, d = xy[keep], s[keep], d[keep]
        n = len(xy)
        if n < MIN_POINTS:
            return []
        dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        adj = dist < CLUSTER_DIST
        label = -np.ones(n, dtype=int)
        clusters = []
        for seed in range(n):                                          # 연결 요소 (0.9 m 이내 사슬)
            if label[seed] >= 0:
                continue
            stack, members = [seed], []
            label[seed] = len(clusters)
            while stack:
                u = stack.pop()
                members.append(u)
                for v in np.nonzero(adj[u] & (label < 0))[0]:
                    label[v] = len(clusters)
                    stack.append(v)
            clusters.append(members)
        out = []
        for members in clusters:
            if len(members) < MIN_POINTS:
                continue
            m = np.asarray(members)
            # The wall at |d|~3.1 was repeatedly treated as a second car.
            # Keep real vehicle surfaces near the lane, discard thin long boundaries.
            if abs(float(np.median(d[m]))) > 2.65:
                continue
            # The obstacle is 4.3 m across the road. Without LiDAR noise its front face is a perfectly
            # straight line ~3.2 m wide, which a 2.8 m width limit threw away -> obstacle never seen.
            if np.ptp(s[m]) > 6.0 or np.ptp(d[m]) > OBSTACLE_LENGTH + 0.3:
                continue
            out.append((float(s[m].mean()), float(d[m].mean()), float(s[m].min()), float(s[m].max()),
                        float(d[m].min()), float(d[m].max()), float(xy[m, 0].mean()), float(xy[m, 1].mean())))
        return out

    # ---- 추적 ---------------------------------------------------------------------------------------
    def _track(self, detections, s_ego, t):
        matched = set()
        for det in detections:
            best, best_score = None, ASSOC_DIST
            for tr in self.tracks:
                if id(tr) in matched:
                    continue
                score = 0.0 if tr.contains(det[0], det[1]) else math.hypot(tr.s - det[0], tr.d - det[1])
                if score < best_score:
                    best, best_score = tr, score
            if best is not None:
                best.update(*det, t)
                matched.add(id(best))
            elif det[0] >= s_ego + NEW_MIN_AHEAD:
                tr = Track(*det, t)
                self.tracks.append(tr)
                matched.add(id(tr))
        for tr in self.tracks:
            if tr.confirmed and tr.side == 0:                           # 확정 때 회피 방향을 고정
                tr.side = 1 if tr.d >= 0.0 else -1
                if self.log:
                    self.log.info(f'장애물 확정 s={tr.s:.1f} d={tr.d:+.2f} -> {"오른쪽" if tr.side > 0 else "왼쪽"} 차선으로 회피')
        self.tracks = [tr for tr in self.tracks if s_ego < tr.back() + CULL_BEHIND
                       and t - tr.t < (5.0 if tr.confirmed else 0.45)]

    # ---- 회피 곡선 ----------------------------------------------------------------------------------
    @staticmethod
    def _ramp_len(amplitude):
        """오프셋 amplitude(m)를 코사인으로 옮길 때 곡률이 KAPPA_LIM을 넘지 않는 길이 (최소 RAMP_MIN)."""
        return max(RAMP_MIN, math.pi * math.sqrt(abs(amplitude) / (2.0 * KAPPA_LIM)))

    def _anchors(self, elems):
        """elems = [지나친 장애물(복귀 중, 선택), A, B(선택)] 를 이어 주는 (s, d) 기준점. 기준점 사이는 코사인."""
        anchors = []
        prev = None                                                     # (이전 plateau 끝 s, 이전 목표 d)
        for el in elems:
            lo, hi, D = el.lo(), el.hi(), el.target()
            ramp = self._ramp_len(D)
            if prev is None:
                anchors.append((lo - ramp, 0.0))                       # 중앙에서 출발하는 램프
            else:
                ph, pD = prev
                gap = lo - ph
                if pD * D > 0:                                          # 같은 쪽
                    if gap >= 2.0 * ramp + 2.0:                         # 멀면 중앙으로 갔다가 다시
                        anchors += [(ph + RETURN_M, 0.0), (lo - ramp, 0.0)]
                    elif lo <= ph + 0.1:                                # 겹침: 차선 유지하며 plateau를 늘림
                        anchors[-1] = (max(hi, ph), D)
                        prev = (anchors[-1][0], D)
                        continue
                else:                                                   # 반대쪽: A 끝 ~ B 시작을 코사인으로 바로 이음
                    if gap < RAMP_MIN:                                  # 좁으면 중간 지점 기준 4.2 m 램프를 대칭 배치
                        a = max(0.5 * (ph + lo) - 0.5 * RAMP_MIN, anchors[-2][0] + 0.2)
                        anchors[-1] = (a, pD)
                        lo = a + RAMP_MIN
            anchors += [(lo, D), (max(hi, lo + 0.3), D)]
            prev = (anchors[-1][0], D)
        anchors.append((prev[0] + RETURN_M, 0.0))                       # 마지막 장애물 뒤: 4.5 m에 걸쳐 중앙 복귀
        clean = [anchors[0]]
        for s_k, d_k in anchors[1:]:                                    # s가 반드시 증가하도록
            if s_k <= clean[-1][0] + 0.05:
                s_k = clean[-1][0] + 0.05
            clean.append((s_k, d_k))
        return clean

    @staticmethod
    def _cosine(anchors, s):
        """기준점 사이를 코사인으로 잇는 오프셋 d(s). 범위 밖은 0."""
        a_s = np.array([a[0] for a in anchors])
        a_d = np.array([a[1] for a in anchors])
        k = np.clip(np.searchsorted(a_s, s, side='right') - 1, 0, len(a_s) - 2)
        s0, s1, d0, d1 = a_s[k], a_s[k + 1], a_d[k], a_d[k + 1]
        u = np.clip((s - s0) / np.maximum(s1 - s0, 1e-6), 0.0, 1.0)
        out = d0 + (d1 - d0) * 0.5 * (1.0 - np.cos(math.pi * u))
        out[(s < a_s[0]) | (s > a_s[-1])] = 0.0
        return out

    # ---- 메인 ---------------------------------------------------------------------------------------
    def update(self, inp: MissionInput) -> MissionOutput:
        options = getattr(self, 'options', None)
        if options is None:
            options = load_options()
        option = 'obstacle3' if inp.cp and inp.cp.section == ('cp6', 'cp7') else 'obstacle1'
        if not option_on(options, option):
            self.reset()
            return MissionOutput(is_done=True)
        if self.pts is None:
            self._load()
        if self.zone_box is None:                       # 이 구간의 앞뒤 CP를 입구로 하는 장애물 zone (course.json)
            sec = inp.cp.section if inp.cp is not None else ()
            for m in self.obstacle_zones:
                if m.get('entry_cp') in sec and m.get('zone'):
                    zx = [p[0] for p in m['zone']]
                    zy = [p[1] for p in m['zone']]
                    self.zone_box = (min(zx) - ZONE_MARGIN_M, max(zx) + ZONE_MARGIN_M, min(zy) - ZONE_MARGIN_M, max(zy) + ZONE_MARGIN_M)
                    self.zone_half_width = min(max(zx)-min(zx), max(zy)-min(zy)) / 2.0
            if self.zone_box is None:
                return MissionOutput()
        ego_i = inp.cp.index if inp.cp is not None else int(np.argmin(np.hypot(*(self.pts - [inp.x, inp.y]).T)))
        ego_i = min(max(ego_i, 0), len(self.pts) - 2)
        rel = np.array([inp.x, inp.y]) - self.pts[ego_i]
        s_ego = float(self.s[ego_i] + rel @ self.tan[ego_i])

        if inp.scan is not None and stamp(inp.scan) != self.last_scan_stamp:
            self.last_scan_stamp = stamp(inp.scan)
            self._track(self._detect(inp, ego_i, s_ego), s_ego, stamp(inp.scan))
        known = [tr for tr in self.tracks if tr.confirmed and tr.side != 0]
        for tr in known:
            tr.offset = min(LANE_OFFSET, self.zone_half_width - 1.892/2 - 0.20)
        # 타겟 선택: 아직 안 지나친 장애물 중 가장 가까운 1개(A), 바로 뒤 12 m 이내에 보이는 다음 1개(B)
        ahead = sorted((tr for tr in known if s_ego < tr.back() + DROP_MARGIN), key=lambda tr: tr.smin)
        elems = []
        passed = [tr for tr in known if s_ego >= tr.back() + DROP_MARGIN and s_ego < tr.hi() + RETURN_M + 0.5]
        if passed:                                      # 복귀 곡선이 끝나기 전까지는 경로 연속성용으로만 유지
            elems.append(max(passed, key=lambda tr: tr.back()))
        if ahead:
            elems.append(ahead[0])
            if len(ahead) > 1 and ahead[1].smin - ahead[0].back() <= NEXT_RANGE:
                elems.append(ahead[1])
        if not elems:
            return MissionOutput()

        anchors = self._anchors(elems)
        s_hi = min(s_ego + PATH_AHEAD, float(self.s[-1]))
        s_lo = max(s_ego - PATH_BEHIND, 0.0)
        n = int(round((s_hi - s_lo) / PATH_STEP)) + 1
        if n < 2:
            return MissionOutput()
        s_grid = s_lo + PATH_STEP * np.arange(n)
        off = self._cosine(anchors, s_grid)
        base = np.stack([np.interp(s_grid, self.s, self.pts[:, 0]), np.interp(s_grid, self.s, self.pts[:, 1])], axis=1)
        nor = np.stack([np.interp(s_grid, self.s, self.nor[:, 0]), np.interp(s_grid, self.s, self.nor[:, 1])], axis=1)
        nor /= np.maximum(np.hypot(nor[:, 0], nor[:, 1]), 1e-9)[:, None]
        new = base + off[:, None] * nor
        dxy = np.diff(new, axis=0)
        yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
        yaw = np.append(yaw, yaw[-1])
        # Check the swept body, not only the rear-axle centre. Stop if the
        # observed obstacle/road geometry cannot accommodate this maneuver.
        route_heading = np.arctan2(np.interp(s_grid, self.s, self.tan[:, 1]),
                                   np.interp(s_grid, self.s, self.tan[:, 0]))
        rel = yaw - route_heading
        safe = True
        safety_reason = ''
        margin = 0.15 + min(0.15, max(0.0, inp.pose_sigma) * 0.4)
        for front in (-0.79, FRONT_OVERHANG):
            for side in (-1.892/2, 1.892/2):
                corner_d = off + front*np.sin(rel) + side*np.cos(rel)
                if np.any(np.abs(corner_d) > self.zone_half_width - margin):
                    safe = False
                    safety_reason = f'도로 여유 (모서리 {np.max(np.abs(corner_d)):.2f}, 한계 {self.zone_half_width-margin:.2f}m)'
        # Conservative rectangle projection at each path sample against each
        # confirmed vehicle. This does not use true/simulator object positions.
        half_width = 1.892 / 2.0
        s_min = s_grid + np.minimum(-.79*np.cos(rel), FRONT_OVERHANG*np.cos(rel)) - half_width*np.abs(np.sin(rel))
        s_max = s_grid + np.maximum(-.79*np.cos(rel), FRONT_OVERHANG*np.cos(rel)) + half_width*np.abs(np.sin(rel))
        d_min = off + np.minimum(-.79*np.sin(rel), FRONT_OVERHANG*np.sin(rel)) - half_width*np.abs(np.cos(rel))
        d_max = off + np.maximum(-.79*np.sin(rel), FRONT_OVERHANG*np.sin(rel)) + half_width*np.abs(np.cos(rel))
        for tr in known:
            overlap = (s_max > tr.smin-margin) & (s_min < tr.back()+margin)
            if np.any(overlap & (d_max > tr.dmin-margin) & (d_min < tr.dmax+margin)):
                safe = False
                safety_reason = f'장애물 여유 s={tr.s:.1f}, d={tr.d:.2f}'
        if not safe:
            if self.log and inp.t - self.last_warning > 1.0:
                self.log.warn('회피 경로 차체 여유 부족: ' + safety_reason)
                self.last_warning = inp.t
            return MissionOutput(override_cmd=(0.0, 1.0, 0.0, 'D'))
        end_maneuver = max(tr.hi() + RETURN_M for tr in elems)
        path = [(float(x), float(y), float(h), SPEED_LIMIT if s <= end_maneuver else 0.0)
                for (x, y), h, s in zip(new, yaw, s_grid)]
        return MissionOutput(override_path=path)
