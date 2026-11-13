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
import math
import os

import numpy as np
from .sensing import predict_motion
from .course import load_options, option_on

from .mission_base import MissionBase, MissionInput, MissionOutput



GEAR_DRIVE, GEAR_REVERSE = 4, 2          # CtrlCmd.GEAR_DRIVE / GEAR_REVERSE

WHEELBASE = 3.0
MAX_STEER = math.radians(35.4)
CAR_LEN, CAR_WID = 4.635, 1.892
REAR_OVERHANG = 0.79
FRONT_OVERHANG = CAR_LEN - REAR_OVERHANG - WHEELBASE
LIDAR_X = 1.15                           # 뒷축에서 LiDAR까지 앞쪽 거리

PARAMS = {                               # config/bringup.yaml (hk_mit_control)에서 바꿉니다
    'park_enabled': 1.0,                 # 0이면 평행 주차를 하지 않음
    'park_slot_x': -18.1147,             # 슬롯 중심 x (슬롯은 y 방향으로만 움직임)
    'park_slot_y': 120.7754,             # 슬롯 중심 y 기본값 (LiDAR로 못 찾을 때)
    'park_slot_yaw': -math.pi / 2,
    'park_slot_len': 7.4,
    'park_slot_wid': 3.6,
    'park_start_y': 140.0,               # 이 y보다 내려오면 주차 모드 시작
    'park_end_y': 100.0,                 # (주차 전) 이 y보다 내려가면 주차 모드가 아님
    'park_v_approach': 7.0,              # 접근 속력 (m/s)
    'park_v_reverse': 2.3,               # 후진 속력 (m/s)
    'park_v_reverse_fast': 2.6,          # straight portions only
    'park_delay': 0.25,                  # planner replaces this with actual option delay
    'park_v_exit': 3.5,                  # 평행 주차 탈출 전진 속력 (m/s)
    'park_v_adjust': 1.4,                # 보정·탈출 속력 (m/s)
    'park_phase_timeout_s': 9.0,         # 후진/전진 등 한 phase 최대 시간 (s). 넘으면 다음 단계로 강제 전환 (정상 후진 ~7 s는 끊지 않게)
    'park_total_timeout_s': 22.0,        # 전체 주차 시도 최대 시간 (s). 넘으면 멈춘 자리에서 완료 처리 (영구 멈춤 방지)
    'park_decel': 2.2,                   # 접근 감속 계획 (m/s^2)
    'park_a_shift': 0.8,                 # 목표를 슬롯 중심보다 뒤로 당기는 양 (m). 앞 벽 모서리와의 간격 확보
    'park_radius': 4.5,                  # 후진 호 반지름 (m). 최소 4.22 (조향 최대)
    'park_stop_margin': 0.3,             # 정지 목표 앞당김 (m)
    'park_lookahead': 2.0,               # 후진 pure pursuit 전방 거리 (m)
    'park_kp': 0.6,
    'park_ki': 0.25,
    'park_hold_s': 3.1,                  # 정지 유지 시간 (성공 조건은 3초)
    'park_tol_margin': 0.1,              # 차 네 모서리가 슬롯 안쪽으로 이만큼 더 들어와야 완료로 봄 (m)
    'park_exit_heading_deg': 40.0,       # 탈출 때 이만큼 왼쪽으로 돌면 경로 추종으로 넘김
    'park_exit_rear_clear': 0.4,         # 탈출 후진 때 뒤차와 남길 간격 (m)
    'park_debug': 1.0,                   # 1이면 ~/park/ 에 추적 CSV·LiDAR 점을 남김
}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Parking:
    def __init__(self, params=None, log=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.log = log or (lambda s: None)
        self.reset()

    # ------------------------------------------------------------------------------------------------
    def reset(self):
        self.state = 'IDLE'
        self.t_state = 0.0
        self.pts = np.zeros((0, 2))          # 슬롯 띠 안의 LiDAR 점 (지도 좌표)
        self.last_scan_stamp = None
        self.c = None                        # 슬롯 중심 y 추정
        self.c_hits = 0
        self.int = 0.0
        self.path = None                     # 슬롯 좌표계 경로 [(a, b, 이동 방향)]
        self.stopped_since = None
        self.adjust_n = 0
        self.last_t = None
        self.trace = None
        self.t_hold0 = None
        self.dir = 1
        self.exit_a_lim = None
        self.scan_n = 0
        self.hist = []
        self.predicted_slot = None
        self.t_enter = None                  # 이번 주차 시도 시작 시각 (전체 시간 초과 감시용)

    @property
    def done(self):
        return self.state == 'DONE'

    def engaged(self, x, y):
        """주차 모드인지 (True면 control이 update()의 값을 보냄)."""
        if self.p['park_enabled'] < 0.5 or self.state == 'DONE':
            return False
        if self.state == 'IDLE':
            return self.p['park_end_y'] < y < self.p['park_start_y'] and x < -8.0
        return True

    # ---- 좌표 ---------------------------------------------------------------------------------------
    def _c(self):
        return self.c if self.c is not None else self.p['park_slot_y']

    def to_slot(self, x, y, yaw):
        c = self._c()
        sy = self.p['park_slot_yaw']
        # 슬롯 방향 단위벡터 e_a = (cos sy, sin sy), e_b = (-sin sy, cos sy), 원점 (slot_x, c)
        dx, dy = x - self.p['park_slot_x'], y - c
        a = dx * math.cos(sy) + dy * math.sin(sy)
        b = -dx * math.sin(sy) + dy * math.cos(sy)
        return a, b, wrap(yaw - sy)

    def to_map(self, a, b):
        c = self._c()
        sy = self.p['park_slot_yaw']
        return (self.p['park_slot_x'] + a * math.cos(sy) - b * math.sin(sy),
                c + a * math.sin(sy) + b * math.cos(sy))

    @staticmethod
    def corners(a, b, psi):
        """뒷축 (a, b), 방향 psi인 차의 네 모서리 (슬롯 좌표)."""
        ca, sa = math.cos(psi), math.sin(psi)
        out = []
        for la in (-REAR_OVERHANG, WHEELBASE + FRONT_OVERHANG):
            for lb in (-CAR_WID / 2, CAR_WID / 2):
                out.append((a + la * ca - lb * sa, b + la * sa + lb * ca))
        return out

    def inside(self, a, b, psi, margin):
        hl, hw = self.p['park_slot_len'] / 2 - margin, self.p['park_slot_wid'] / 2 - margin
        return all(abs(ca) <= hl and abs(cb) <= hw for ca, cb in self.corners(a, b, psi))

    def a_target(self):
        """뒷축이 있어야 할 a (차 중심이 슬롯 중심이 되는 곳)."""
        return -(CAR_LEN / 2 - REAR_OVERHANG) - self.p['park_a_shift']

    # ---- LiDAR ----------------------------------------------------------------------------------------
    def _ingest(self, scan, x, y, yaw):
        stamp = (scan.header.stamp.sec, scan.header.stamp.nanosec)
        if stamp == self.last_scan_stamp:
            return
        self.last_scan_stamp = stamp
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > 0.5) & (r < 40.0)
        if not ok.any():
            return
        bx = LIDAR_X + r[ok] * np.cos(ang[ok])
        by = r[ok] * np.sin(ang[ok])
        cy, sy_ = math.cos(yaw), math.sin(yaw)
        X = x + bx * cy - by * sy_
        Y = y + bx * sy_ + by * cy
        xs = self.p['park_slot_x']
        c0 = self.p['park_slot_y']
        keep = (X > xs + 1.05) & (X < xs + 2.9) & (Y > c0 - 10.0) & (Y < c0 + 10.0)
        if keep.any():
            self.pts = np.vstack([self.pts, np.column_stack([X[keep], Y[keep]])])[-20000:]
        self.scan_n += 1
        if self.p['park_debug'] > 0.5 and self.state == 'APPROACH' and self.scan_n % 10 == 1:
            d = os.path.expanduser('~/park')
            try:
                os.makedirs(d, exist_ok=True)
                np.savetxt(os.path.join(d, f'scan_{self.scan_n:04d}_{x:.1f}_{y:.1f}_{yaw:.2f}.txt'),
                           np.column_stack([X, Y]), fmt='%.2f')
            except OSError:
                pass
        if self.scan_n % 3 == 0:
            self._find_slot()

    @staticmethod
    def _close(occ, k):
        out = occ.copy()
        idx = np.flatnonzero(occ)
        for i0, i1 in zip(idx[:-1], idx[1:]):
            if 1 < i1 - i0 <= k:
                out[i0:i1] = True
        return out

    def _find_slot(self):
        if len(self.pts) < 20:
            return
        # Estimate the row's lateral position instead of requiring GNSS to
        # place all returns in a fixed 65 cm strip.
        edges = np.arange(self.pts[:, 0].min(), self.pts[:, 0].max() + 0.2, 0.1)
        hist, edges = np.histogram(self.pts[:, 0], bins=edges)
        peak = int(np.argmax(np.convolve(hist, np.ones(3), mode='same')))
        peak = min(peak, len(edges)-2)
        wall_x = (edges[peak] + edges[peak+1]) / 2.0
        points = self.pts[np.abs(self.pts[:, 0] - wall_x) < 0.4]
        if len(points) < 20:
            return
        c0, L = self.p['park_slot_y'], self.p['park_slot_len']
        lo, hi = c0 - 10.0, c0 + 10.0
        n = int(round((hi - lo) / 0.1))
        idx = np.clip(((points[:, 1] - lo) / 0.1).astype(int), 0, n - 1)
        cnt = np.bincount(idx, minlength=n)
        occ = cnt >= 2
        for k in range(1, 9):                               # 0.8 m 이하의 구멍은 벽이 이어진 것으로 봄
            occ[k:] |= False
        occ = self._close(occ, 8)
        # 빈 구간(run) 찾기
        best = None
        i = 0
        while i < n:
            if occ[i]:
                i += 1
                continue
            j = i
            while j < n and not occ[j]:
                j += 1
            g0, g1 = i, j                                   # [g0, g1) 비어 있음
            length = (g1 - g0) * 0.1
            before = occ[max(0, g0 - 12):g0].sum() * 0.1    # 양쪽에 막힌 곳(앞·뒤 차)이 1.2 m 안에 있어야 함
            after = occ[g1:g1 + 12].sum() * 0.1
            if g0 > 0 and g1 < n and before >= 0.4 and after >= 0.4 and L - 0.8 <= length <= L + 1.5:
                score = abs(length - L)
                if best is None or score < best[0]:
                    best = (score, lo + (g0 * 0.1), lo + (g1 * 0.1))
            i = j
        if best is None:
            return
        center = 0.5 * (best[1] + best[2])
        if self.c is not None and abs(center - self.c) < 0.3:
            self.c_hits += 1
        else:
            self.c_hits = 1
        self.c = center if self.c is None else 0.6 * self.c + 0.4 * center
        self.gap = (best[1], best[2])

    # ---- 제어 도우미 ----------------------------------------------------------------------------------
    def _speed(self, target, v_mag, dt):
        err = target - v_mag
        old_int = self.int
        candidate = clamp(old_int + err * dt, -2.0, 2.0)
        u_candidate = self.p['park_kp'] * err + self.p['park_ki'] * candidate
        if not ((u_candidate > 1.0 and err > 0.0) or (u_candidate < -1.0/0.6 and err < 0.0)):
            self.int = candidate
        u = self.p['park_kp'] * err + self.p['park_ki'] * self.int
        if target <= 0.0 and v_mag < 0.15:
            return 0.0, 1.0
        if u >= 0:
            return min(u, 1.0), 0.0
        return 0.0, min(-u * 0.6, 1.0)

    def _set(self, state, now):
        self.log(f'parking: {self.state} -> {state}')
        self.state, self.t_state = state, now
        self.int = 0.0
        self.stopped_since = None

    def _stopped(self, v_mag, now, hold=0.4):
        if v_mag < 0.03:
            if self.stopped_since is None:
                self.stopped_since = now
            return now - self.stopped_since >= hold
        self.stopped_since = None
        return False

    @staticmethod
    def _stanley(path, x, y, yaw, speed):
        fx, fy = x + WHEELBASE * math.cos(yaw), y + WHEELBASE * math.sin(yaw)
        px, py, pyaw = min(path, key=lambda q: math.hypot(q[0] - fx, q[1] - fy))
        he = wrap(pyaw - yaw)
        ce = -math.sin(pyaw) * (px - fx) + math.cos(pyaw) * (py - fy)
        return clamp((he + math.atan2(1.0 * ce, 1.0 + abs(speed))) / MAX_STEER, -1.0, 1.0)

    # ---- 경로 계획 (슬롯 좌표) -------------------------------------------------------------------------
    @staticmethod
    def _tail(pts, length):
        if not pts:
            return []
        a, b, th = pts[-1]
        out = []
        for i in range(int(length / 0.1)):
            a += 0.1 * math.cos(th)
            b += 0.1 * math.sin(th)
            out.append((a, b, th))
        return out

    def _plan_entry_exact(self, a0, b0):
        """호 길이를 정확히 맞춘 진입 경로 (이동 방향 th = pi에서 시작)."""
        R = self.p['park_radius']
        at = self.a_target()
        ds = 0.05
        phi = math.acos(clamp(1.0 - b0 / (2 * R), -1.0, 1.0)) if b0 > 0.15 else 0.0
        arc_back = 2 * R * math.sin(phi)
        straight = max(0.0, (a0 - arc_back) - at)
        pts = []
        a, b, th = a0, b0, math.pi
        for _ in range(int(straight / ds)):
            a += ds * math.cos(th)
            b += ds * math.sin(th)
            pts.append((a, b, th))
        for sign in (+1, -1):
            m = max(1, int(R * phi / ds))
            step = R * phi / m if phi > 0 else 0.0
            dth = sign * phi / m
            for _ in range(m):
                th += dth / 2
                a += step * math.cos(th)
                b += step * math.sin(th)
                th += dth / 2
                pts.append((a, b, th))
        self.main_n = len(pts)
        pts.extend(self._tail(pts, 3.0))
        return pts

    def a_start(self, b0):
        """이 a에서 후진을 시작하면 직진 구간 없이 딱 목표에 닿는 a."""
        R = self.p['park_radius']
        phi = math.acos(clamp(1.0 - b0 / (2 * R), -1.0, 1.0)) if b0 > 0.15 else 0.0
        return self.a_target() + 2 * R * math.sin(phi)

    # ---- 후진/전진 pure pursuit -----------------------------------------------------------------------
    def _pursuit(self, path, a, b, psi, reverse):
        """path (슬롯 좌표, 이동 방향 포함)를 뒷축 기준으로 따라가는 조향(-1~1)과 남은 점 수."""
        i = min(range(len(path)), key=lambda k: (path[k][0] - a) ** 2 + (path[k][1] - b) ** 2)
        Ld = self.p['park_lookahead']
        j = i
        acc = 0.0
        while j + 1 < len(path) and acc < Ld:
            acc += math.hypot(path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1])
            j += 1
        ta, tb, _ = path[j]
        travel = wrap(psi + math.pi) if reverse else psi
        alpha = wrap(math.atan2(tb - b, ta - a) - travel)
        dist = max(math.hypot(ta - a, tb - b), 0.5)
        kappa = 2 * math.sin(alpha) / dist
        delta = math.atan(WHEELBASE * kappa)
        if reverse:
            delta = -delta
        return clamp(delta / MAX_STEER, -1.0, 1.0), i

    # ---- 메인 ---------------------------------------------------------------------------------------------
    def update(self, x, y, yaw, speed, scan, path, now, scan_pose=None):
        dt = 0.02 if self.last_t is None else clamp(now - self.last_t, 0.001, 0.1)
        self.last_t = now
        v = abs(speed)
        if scan is not None and self.state in ('IDLE', 'APPROACH', 'STOP'):
            self._ingest(scan, *(scan_pose or (x, y, yaw)))
        a, b, psi = self.to_slot(x, y, yaw)
        predicted = predict_motion(x, y, yaw, speed, now, self.p['park_delay'], self.hist)
        self.predicted_slot = self.to_slot(*predicted)
        out = self._step(x, y, yaw, speed, v, a, b, psi, path, now, dt)
        self.hist.append((now, out[2] * MAX_STEER))
        while len(self.hist) > 1 and self.hist[1][0] < now - 1.0:
            self.hist.pop(0)
        self._trace(now, a, b, psi, speed, out)
        return out

    _DONE_STATES = ('HOLD', 'EXIT_SHIFT_R', 'EXIT_BACK', 'EXIT_STOP', 'EXIT_SHIFT_D', 'EXIT_FWD', 'DONE')

    def _step(self, x, y, yaw, speed, v, a, b, psi, path, now, dt):
        p = self.p
        st = self.state
        if self.t_enter is None:
            self.t_enter = now
        if st not in self._DONE_STATES and v < 0.1 and now - self.t_enter > p['park_total_timeout_s']:
            # 전체 시간 초과: 보정을 더 시도하지 않고 멈춘 자리에서 완료 처리 (영구 멈춤 방지)
            self.log(f'parking: 전체 시간 초과({p["park_total_timeout_s"]:.0f}s) - 지금 자리에서 완료 처리')
            self._set('HOLD', now)
            return 0.0, 1.0, 0.0, GEAR_REVERSE if self.dir < 0 else GEAR_DRIVE
        if st == 'IDLE':
            self._set('APPROACH', now)
            st = 'APPROACH'
        if st == 'APPROACH':
            valid = self.c is not None and self.c_hits >= 3
            c_use = self.c if valid else p['park_slot_y']
            b_now = max(b, 0.0)
            a_stop = self.a_start(b_now) - p['park_stop_margin']
            # 슬롯을 못 찾은 채 기본 위치 근처까지 오면 기본값으로 진행
            fallback = (not valid) and a > 3.0
            if fallback:
                return 0.0, 1.0, 0.0, GEAR_DRIVE     # wait for a measured slot, not a guessed seed position
            if (valid or fallback) and a >= a_stop - 0.01:
                if not valid:
                    self.log('parking: LiDAR로 슬롯을 못 찾아 기본 위치 사용')
                self._set('STOP', now)
                return 0.0, 1.0, 0.0, GEAR_DRIVE
            target = p['park_v_approach']
            if valid:
                target = min(target, math.sqrt(max(2 * p['park_decel'] * (a_stop - a), 0.0)) + 0.15)
            acc, brk = self._speed(target, v, dt)
            steer = self._stanley(path, x, y, yaw, speed) if path else 0.0
            return acc, brk, steer, GEAR_DRIVE
        if st == 'STOP':
            if self._stopped(v, now):
                self._plan_reverse(a, b)
                self._set('SHIFT_R', now)
            return 0.0, 1.0, 0.0, GEAR_DRIVE
        if st == 'SHIFT_R':
            steer, _ = self._pursuit(self.path, a, b, psi, True)
            if now - self.t_state > 0.4:
                self._set('REVERSE', now)
            return 0.0, 1.0, steer, GEAR_REVERSE
        if st == 'REVERSE':
            steer, _ = self._pursuit(self.path, *(self.predicted_slot or (a, b, psi)), True)
            idx = min(range(self.main_n), key=lambda k:(self.path[k][0]-a)**2+(self.path[k][1]-b)**2)
            if idx >= self.main_n - 3 or a <= self.a_target() - 0.3 or now - self.t_state > p['park_phase_timeout_s']:
                self._set('STOP_REV', now)
                return 0.0, 1.0, 0.0, GEAR_REVERSE
            v_t = p['park_v_reverse']
            j = min(idx + 12, self.main_n - 1)
            turn = abs(wrap(self.path[j][2] - self.path[idx][2])) / max(0.05, (j-idx)*0.05)
            if turn < 0.05:
                v_t = p['park_v_reverse_fast']
            left = (self.main_n - 1 - idx) * 0.05
            v_t = min(v_t, math.sqrt(0.35**2 + 2.0 * 0.8 * max(0.0, left-0.15)))
            acc, brk = self._speed(v_t, v, dt)
            return acc, brk, steer, GEAR_REVERSE
        if st == 'STOP_REV':
            if self._stopped(v, now):
                self._after_stop(a, b, psi, now)
            return 0.0, 1.0, 0.0, GEAR_REVERSE
        if st in ('ADJUST_SHIFT',):
            if now - self.t_state > 0.4:
                self._set('ADJUST', now)
            return 0.0, 1.0, 0.0, GEAR_REVERSE if self.dir < 0 else GEAR_DRIVE
        if st == 'ADJUST':
            at = self.a_target()
            err = a - at
            moving_back = self.dir < 0
            reached = (err <= 0.05) if moving_back else (err >= -0.05)
            if reached or now - self.t_state > p['park_phase_timeout_s']:
                self._set('STOP_ADJ', now)
                return 0.0, 1.0, 0.0, GEAR_REVERSE if moving_back else GEAR_DRIVE
            # 직선 b=0 위에서 방향 0을 맞추며 이동
            line = [(a + self.dir * 0.1 * k, 0.0, math.pi if moving_back else 0.0) for k in range(60)]
            steer, _ = self._pursuit(line, *(self.predicted_slot or (a,b,psi)), moving_back)
            acc, brk = self._speed(p['park_v_adjust'], v, dt)
            return acc, brk, steer, GEAR_REVERSE if moving_back else GEAR_DRIVE
        if st == 'STOP_ADJ':
            if self._stopped(v, now):
                self._after_stop(a, b, psi, now)
            return 0.0, 1.0, 0.0, GEAR_REVERSE if self.dir < 0 else GEAR_DRIVE
        if st == 'HOLD':
            if self.t_hold0 is None:
                self.t_hold0 = now
            if now - self.t_hold0 >= p['park_hold_s']:
                self.exit_a_lim = -(p['park_slot_len'] / 2) + REAR_OVERHANG + p['park_exit_rear_clear']
                if a > self.exit_a_lim + 0.1:
                    self._set('EXIT_SHIFT_R', now)
                else:
                    self._set('EXIT_SHIFT_D', now)
            return 0.0, 1.0, 0.0, GEAR_REVERSE if self.dir < 0 else GEAR_DRIVE
        if st == 'EXIT_SHIFT_R':
            if now - self.t_state > 0.4:
                self._set('EXIT_BACK', now)
            return 0.0, 1.0, 0.0, GEAR_REVERSE
        if st == 'EXIT_BACK':
            if a <= self.exit_a_lim or now - self.t_state > 6.0:
                self._set('EXIT_STOP', now)
                return 0.0, 1.0, 0.0, GEAR_REVERSE
            acc, brk = self._speed(p['park_v_adjust'], v, dt)
            steer, _ = self._pursuit([(a - 0.1 * k, 0.0, math.pi) for k in range(60)], *(self.predicted_slot or (a,b,psi)), True)
            return acc, brk, steer, GEAR_REVERSE
        if st == 'EXIT_STOP':
            if self._stopped(v, now, 0.3):
                self._set('EXIT_SHIFT_D', now)
            return 0.0, 1.0, 0.0, GEAR_REVERSE
        if st == 'EXIT_SHIFT_D':
            if now - self.t_state > 0.4:
                self._set('EXIT_FWD', now)
            return 0.0, 1.0, 1.0, GEAR_DRIVE
        if st == 'EXIT_FWD':
            exit_psi = self.predicted_slot[2] if self.predicted_slot else psi
            if exit_psi >= math.radians(p['park_exit_heading_deg']) or now - self.t_state > 8.0:
                self._set('DONE', now)
                return 0.0, 0.0, 0.0, GEAR_DRIVE
            acc, brk = self._speed(p['park_v_exit'], v, dt)
            return acc, brk, 1.0, GEAR_DRIVE
        return 0.0, 1.0, 0.0, GEAR_DRIVE

    def _plan_reverse(self, a, b):
        self.path = self._plan_entry_exact(a, max(b, 0.0))
        self.log(f'parking: 후진 계획 a={a:.2f} b={b:.2f} c={self._c():.2f} 점 {len(self.path)}')

    def _relaxed_ok(self, a, b, psi):
        """완화된 완료 기준: 중심 기준 세로 0.50 m, 가로 0.35 m, 방향 8도 이내 + (정지는 호출 전 이미 확인됨)."""
        return (abs(a - self.a_target()) <= 0.50 and abs(b) <= 0.35
                and abs(wrap(psi)) <= math.radians(8.0))

    def _after_stop(self, a, b, psi, now):
        p = self.p
        ok = self.inside(a, b, psi, 0.05 + p['park_tol_margin']) or self._relaxed_ok(a, b, psi)
        self.log(f'parking: 정지 a={a:.2f} b={b:.2f} psi={math.degrees(psi):.1f}도 안={ok} 보정 {self.adjust_n}')
        if ok:
            self._set('HOLD', now)
            return
        if self.adjust_n >= 4:
            # 보정 한도 초과: 더 시도하지 않고 멈춘 자리에서 완료 처리 (예전엔 여기서 RuntimeError로 영구 멈춤)
            self.log('parking: 평행 주차 보정 한도 초과 - 지금 자리에서 완료 처리')
            self._set('HOLD', now)
            return
        self.adjust_n += 1
        err = a - self.a_target()
        self.dir = -1 if err > 0 else 1
        if abs(err) < 0.2:                                  # 앞뒤는 맞는데 옆·방향이 틀림: 조금 앞으로 갔다가 다시 후진
            self.dir = 1
        self._set('ADJUST_SHIFT', now)

    # ---- 기록 ----------------------------------------------------------------------------------------------
    def _trace(self, now, a, b, psi, speed, out):
        if self.p['park_debug'] < 0.5:
            return
        d = os.path.expanduser('~/park')
        try:
            if self.trace is None:
                os.makedirs(d, exist_ok=True)
                self.trace = open(os.path.join(d, 'park_trace.csv'), 'a')
                self.trace.write('t,state,a,b,psi_deg,speed,accel,brake,steer,gear,c\n')
            self.trace.write(f'{now:.2f},{self.state},{a:.3f},{b:.3f},{math.degrees(psi):.2f},{speed:.3f},'
                             f'{out[0]:.2f},{out[1]:.2f},{out[2]:.2f},{out[3]},{self._c():.3f}\n')
            self.trace.flush()
        except OSError:
            pass


RPARAMS = {                              # 후면 주차 (T자 통로) 값
    'rpark_slot_x': 0.3353,              # 슬롯 중심 x
    'rpark_gate_to_slot': 6.4,           # 통로 입구 중심 y -> 슬롯 중심 y (seed 0 추정: 통로 74.87, 슬롯 약 81.25)
    'rpark_gate_y': 74.5,                # 통로 입구 중심 y 기본값 (LiDAR로 못 찾을 때)
    'rpark_wall_x': -10.25,              # 도로 왼쪽 벽 x
    'rpark_gate_w': 5.0,                 # 통로 폭 (m)
    'rpark_lane_x': -13.16,              # 진입 때 따라갈 도로 x
    'rpark_r1': 4.6,                     # 진입 왼쪽 회전 반지름
    'rpark_r2': 4.6,                     # 안쪽에서 오른쪽(남쪽) 회전 반지름
    'rpark_gin': 0.5,                    # 통로 안에서 중심선보다 북쪽으로 붙는 양 (m)
    'rpark_v_approach': 6.5,             # 접근: 평행주차 구간에서 후면주차 통로까지 이동 속력 (옛 하드코딩 2.5)
    'rpark_v_enter': 3.5,                # 진입: 호(회전) 구간 속력
    'rpark_v_enter_fast': 8.0,           # 진입: 직선 구간 최고 속력 (호 앞에서 미리 감속)
    'rpark_v_exit': 3.5,                 # 탈출: 호 구간 속력
    'rpark_v_exit_fast': 7.0,            # 탈출: 직선 구간 최고 속력
    'rpark_a_dec': 3.0,                  # 호 앞 감속 계획 (m/s^2)
    'rpark_v_reverse': 3.5,              # 후면 주차 후진 속력 (끝이 가까우면 감속)
    'rpark_exit_north': 0.3,             # 탈출 때 통로 안에서 중심선보다 북쪽으로 붙는 양 (m)
    'rpark_exit_r3': 4.6,
    'rpark_exit_r4': 4.6,
    'rpark_slot_len': 5.4,
    'rpark_slot_wid': 3.6,
    'rpark_tol_margin': 0.03,
    'rpark_delay': 0.1,                  # 조향 지연 보정용 가정 (s). 0.1: 지연이 꺼져 있어도(0) 켜져 있어도(0.25~0.35) 대체로 안전함
    'rpark_ld_min': 2.0,                 # 전진 추종 전방 거리 최소 (m)
    'rpark_ld_k': 0.6,                   # 전방 거리 = max(최소, 이 값 x 속력 + 1.0)
    'rpark_commit_y': 0.5,               # 통로를 못 찾았을 때 기본 위치로 진입을 결정하는 y (기본 통로 + 반지름 + 6 + 이 값)
    'rpark_gap_stop': 0.12,              # 후진 중 뒤 범퍼가 안쪽 벽과 이 거리(m)가 되면 정지
    'rpark_wall_to_c': 2.8,              # 안쪽 벽 y -> 슬롯 중심 y (m)
}


def build_path(start, prims, ds=0.1):
    """(x, y, 방향)에서 ('S', 길이) / ('A', 각도(+왼쪽), 반지름) 조각을 이어 붙인 경로 [(x, y, 방향)]."""
    x, y, h = start
    out = [(x, y, h)]
    for p in prims:
        if p[0] == 'S':
            for _ in range(max(1, int(p[1] / ds))):
                x += ds * math.cos(h)
                y += ds * math.sin(h)
                out.append((x, y, h))
        else:
            ang, R = p[1], p[2]
            n = max(1, int(abs(ang) * R / ds))
            dh, step = ang / n, abs(ang) * R / n
            for _ in range(n):
                h += dh / 2
                x += step * math.cos(h)
                y += step * math.sin(h)
                h += dh / 2
                out.append((x, y, h))
    return out


class RearParking(Parking):
    """후면 주차: 도로 왼쪽 벽의 틈 -> 통로(동쪽) -> 안쪽 구역에서 남쪽을 보고 선 뒤 -> 북쪽 슬롯으로 후진."""

    def __init__(self, params=None, log=None):
        p = dict(RPARAMS)
        if params:
            p.update(params)
        super().__init__(p, log)

    def reset(self):
        super().reset()
        self.g = None
        self.g_hits = 0
        self.gpts = np.zeros((0, 2))
        self.main_n = 0
        self.tries = 0
        self.y_tgt = 0.0
        self.rev_tol = 0.3
        self.wall_y = None
        self.wall_stop = True
        self.scan_now = None
        self.hist = []                         # (시각, 조향 명령 rad): 지연 보정에 사용
        self.g_locked = False

    # ---- 통로 입구 찾기 ---------------------------------------------------------------------------------
    def _ingest_gate(self, scan, x, y, yaw):
        stamp = (scan.header.stamp.sec, scan.header.stamp.nanosec)
        if stamp == self.last_scan_stamp:
            return
        self.last_scan_stamp = stamp
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > 0.5) & (r < 60.0)
        if not ok.any():
            return
        bx, by = LIDAR_X + r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])
        X = x + bx * math.cos(yaw) - by * math.sin(yaw)
        Y = y + bx * math.sin(yaw) + by * math.cos(yaw)
        wx = self.p['rpark_wall_x']
        g0 = self.p['rpark_gate_y']
        keep = (X > wx - 1.3) & (X < wx + 1.3) & (Y > g0 - 14.0) & (Y < g0 + 16.0)      # 위치 추정 오차(GNSS 노이즈)로 벽이 옆으로 어긋나 보여도 잡도록 넓게
        if keep.any():
            self.gpts = np.vstack([self.gpts, np.column_stack([X[keep], Y[keep]])])[-20000:]
        self.scan_n += 1
        if self.scan_n % 3 == 0:
            self._find_gate()

    def _find_gate(self):
        if len(self.gpts) < 20:
            return
        # 벽의 x 위치를 데이터에서 찾는다: 가장 점이 많이 모인 x(0.1 m 폭)를 벽선으로 보고 그 주변 +-0.4 m만 씀
        xs_ = self.gpts[:, 0]
        hx, ex = np.histogram(xs_, bins=np.arange(xs_.min(), xs_.max() + 0.2, 0.1))
        k = int(np.argmax(np.convolve(hx, np.ones(3), mode='same')))
        wall_x = 0.5 * (ex[k] + ex[min(k + 1, len(ex) - 1)])
        self.gwall_x = wall_x
        gp = self.gpts[np.abs(xs_ - wall_x) < 0.4]
        if len(gp) < 20:
            return
        g0, W = self.p['rpark_gate_y'], self.p['rpark_gate_w']
        lo, hi = g0 - 14.0, g0 + 16.0
        n = int(round((hi - lo) / 0.1))
        idx = np.clip(((gp[:, 1] - lo) / 0.1).astype(int), 0, n - 1)
        occ = self._close(np.bincount(idx, minlength=n) >= 1, 20)      # 먼 벽은 점이 드문드문 -> 2 m 이하 구멍은 벽으로 이어 봄
        best = None
        i = 0
        while i < n:
            if occ[i]:
                i += 1
                continue
            j = i
            while j < n and not occ[j]:
                j += 1
            length = (j - i) * 0.1
            before = occ[max(0, i - 15):i].sum() * 0.1
            after = occ[j:j + 15].sum() * 0.1
            if i > 0 and j < n and before >= 0.3 and after >= 0.3 and W - 1.8 <= length <= W + 2.2:
                sc = abs(length - W)
                if best is None or sc < best[0]:
                    best = (sc, lo + i * 0.1, lo + j * 0.1)
            i = j
        if best is None:
            return
        center = 0.5 * (best[1] + best[2])
        self.g_hits = self.g_hits + 1 if (self.g is not None and abs(center - self.g) < 0.3) else 1
        self.g = center if self.g is None else 0.6 * self.g + 0.4 * center

    def _dump_gate(self, x, y, yaw, valid):
        if self.p['park_debug'] < 0.5:
            return
        try:
            d = os.path.expanduser('~/park')
            os.makedirs(d, exist_ok=True)
            tag = 'ok' if valid else 'FALLBACK'
            np.savetxt(os.path.join(d, f'gate_{tag}_{self.last_t:.1f}_{x:.1f}_{y:.1f}.txt'), self.gpts, fmt='%.2f')
        except OSError:
            pass

    def slot_c(self):
        g = self.g if self.g is not None else self.p['rpark_gate_y']
        return g + self.p['rpark_gate_to_slot']

    # ---- 경로 ---------------------------------------------------------------------------------------------
    def _entry_path(self, y_now):
        p = self.p
        g = self.g if self.g is not None else p['rpark_gate_y']
        R1, R2, xl = p['rpark_r1'], p['rpark_r2'], p['rpark_lane_x']
        yt = g + p['rpark_gin'] + R1                         # 왼쪽 회전을 시작할 y
        xb = p['rpark_slot_x'] - R2
        sx = max(0.0, xb - (xl + R1))
        lead = max(0.5, y_now - yt)
        prims = [('S', lead), ('A', math.pi / 2, R1), ('S', sx), ('A', -math.pi / 2, R2), ('S', 0.5)]
        pts = build_path((xl, y_now, -math.pi / 2), prims)
        self.main_n = len(pts)
        for k in range(1, 31):
            pts.append((pts[-1][0], pts[-1][1] - 0.1, -math.pi / 2))
        self.turn_d = self._turn_dist(pts)
        return pts

    @staticmethod
    def _turn_dist(path):
        """각 점에서 다음 회전(호) 시작까지 경로를 따라 남은 거리 (m). 호 안이면 0."""
        n = len(path)
        d = [0.0] * n
        for i in range(n - 2, -1, -1):
            curved = abs(wrap(path[i + 1][2] - path[i][2])) > 1e-4
            d[i] = 0.0 if curved else d[i + 1] + math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        d[n - 1] = d[n - 2] if n > 1 else 0.0
        return d

    def _path_speed(self, i, v_fast, v_arc):
        d = self.turn_d[min(i, len(self.turn_d) - 1)]
        if d <= 0.0:
            return v_arc
        return min(v_fast, math.sqrt(v_arc ** 2 + 2 * self.p['rpark_a_dec'] * d))

    def _reverse_path(self, x, y):
        pts = [(self.p['rpark_slot_x'] + (x - self.p['rpark_slot_x']) * max(0.0, 1 - 0.025 * k), y + 0.1 * k, math.pi / 2)
               for k in range(0, 200)]
        return pts

    def _exit_path(self, x, y):
        p = self.p
        R3, R4 = p['rpark_exit_r3'], p['rpark_exit_r4']
        g = self.g if self.g is not None else p['rpark_gate_y']
        s0 = max(0.3, y - (g + p['rpark_exit_north'] + R3))      # 통로 중심선(+약간 북쪽)에서 서쪽을 보도록 오른쪽 회전 시작점
        xe = p['rpark_lane_x'] + R4
        sx = max(0.0, (x - R3) - xe)
        pts = build_path((x, y, -math.pi / 2), [('S', s0), ('A', -math.pi / 2, R3), ('S', sx), ('A', math.pi / 2, R4), ('S', 2.0)])
        self.main_n = len(pts)
        for k in range(1, 31):
            pts.append((pts[-1][0], pts[-1][1] - 0.1, -math.pi / 2))
        self.turn_d = self._turn_dist(pts)
        return pts

    def _in_slot(self, x, y, yaw):
        p = self.p
        c = self.slot_c()
        if getattr(self, 'wall_y', None) is not None:
            c = self.wall_y - p['rpark_wall_to_c']
        a = -(y - c)                                         # 슬롯 방향(-y)으로 앞
        b = -(x - p['rpark_slot_x'])                         # 그 왼쪽 (방향 -y의 왼쪽은 +x)
        b = x - p['rpark_slot_x']
        psi = wrap(yaw + math.pi / 2)
        hl, hw = p['rpark_slot_len'] / 2 - 0.05 - p['rpark_tol_margin'], p['rpark_slot_wid'] / 2 - 0.05 - p['rpark_tol_margin']
        return all(abs(ca) <= hl and abs(cb) <= hw for ca, cb in self.corners(a, b, psi)), a, b, psi

    def _predict(self, x, y, yaw, speed, now):
        """조향 지연(rpark_delay) 동안 이미 보낸 명령이 반영된 뒤의 자세를 앞서 계산한다 (전진 전용)."""
        return predict_motion(x, y, yaw, speed, now, self.p['rpark_delay'], self.hist)

    def _reverse_pursuit(self, path, x, y, yaw, speed, now):
        """후진 추종: 지연 보정 + 속력에 비례한 전방 거리 (전진과 같은 방식, 속력은 음수로 예측)."""
        px, py, pyaw = self._predict(x, y, yaw, -abs(speed), now)
        Ld = max(self.p['rpark_ld_min'], self.p['rpark_ld_k'] * abs(speed) + 1.0)
        old = self.p['park_lookahead']
        self.p['park_lookahead'] = Ld
        try:
            return self._pursuit(path, px, py, pyaw, True)
        finally:
            self.p['park_lookahead'] = old

    def _forward_pursuit(self, path, x, y, yaw, speed, now):
        px, py, pyaw = self._predict(x, y, yaw, abs(speed), now)
        Ld = max(self.p['rpark_ld_min'], self.p['rpark_ld_k'] * abs(speed) + 1.0)
        old = self.p['park_lookahead']
        self.p['park_lookahead'] = Ld
        try:
            return self._pursuit(path, px, py, pyaw, False)
        finally:
            self.p['park_lookahead'] = old

    def _rear_gap(self, scan):
        """뒤 범퍼에서 바로 뒤 장애물(벽)까지 거리 (m). 못 재면 None."""
        if scan is None:
            return None
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > 0.3)
        px, py = r * np.cos(ang), r * np.sin(ang)
        sel = ok & (px < -0.3) & (np.abs(py) < 0.9)
        if sel.sum() < 2:
            return None
        return float(np.min(-px[sel])) - (REAR_OVERHANG + LIDAR_X)

    # ---- 메인 ---------------------------------------------------------------------------------------------
    def update(self, x, y, yaw, speed, scan, path, now, scan_pose=None):
        self.scan_now = scan
        dt = 0.05 if self.last_t is None else clamp(now - self.last_t, 0.001, 0.2)
        self.last_t = now
        v = abs(speed)
        if scan is not None and not self.g_locked and self.state in ('IDLE', 'APPROACH', 'ENTER'):
            self._ingest_gate(scan, *(scan_pose or (x, y, yaw)))
        out = self._rstep(x, y, yaw, speed, v, path, now, dt)
        self.hist.append((now, out[2] * MAX_STEER))
        while len(self.hist) > 1 and self.hist[1][0] < now - 1.0:
            self.hist.pop(0)
        self._trace(now, y, x, wrap(yaw), speed, out)
        return out

    _DONE_STATES_REAR = ('HOLD', 'EXIT_SHIFT_D', 'EXIT_FWD', 'DONE')

    def _rstep(self, x, y, yaw, speed, v, path, now, dt):
        p = self.p
        st = self.state
        D, R = GEAR_DRIVE, GEAR_REVERSE
        if self.t_enter is None:
            self.t_enter = now
        if st not in self._DONE_STATES_REAR and v < 0.1 and now - self.t_enter > p['park_total_timeout_s']:
            # 전체 시간 초과: 보정을 더 시도하지 않고 멈춘 자리에서 완료 처리 (영구 멈춤 방지)
            self.log(f'rear: 전체 시간 초과({p["park_total_timeout_s"]:.0f}s) - 지금 자리에서 완료 처리')
            self._set('HOLD', now)
            return 0.0, 1.0, 0.0, R
        if st == 'IDLE':
            self._set('APPROACH', now)
            st = 'APPROACH'
        if st == 'APPROACH':
            valid = self.g is not None and self.g_hits >= 2
            if not valid and y < p['rpark_gate_y'] + p['rpark_r1'] + 6.0 + p['rpark_commit_y']:
                return 0.0, 1.0, 0.0, D
            if valid:
                if not valid:
                    self.log('rear: LiDAR로 통로를 못 찾아 기본 위치 사용')
                self._dump_gate(x, y, yaw, valid)
                self.path = self._entry_path(y)
                self.log(f'rear: 통로 y={self.g if self.g is not None else p["rpark_gate_y"]:.2f} 슬롯 c={self.slot_c():.2f}')
                self._set('ENTER', now)
                return 0.0, 0.0, 0.0, D
            acc, brk = self._speed(p['rpark_v_approach'], v, dt)   # 평행주차 -> 후면주차 통로까지 이동하는 구간
            steer = self._stanley(path, x, y, yaw, speed) if path else 0.0
            return acc, brk, steer, D
        if st == 'ENTER':
            g_now = self.g if self.g is not None else p['rpark_gate_y']
            if y > g_now + p['rpark_gin'] + p['rpark_r1'] + 2.5:     # 아직 왼쪽 회전 시작 전: 더 가까워져 정확해진 통로 위치로 경로를 다시 계산
                self.path = self._entry_path(y)
            else:
                self.g_locked = True
            steer, i = self._forward_pursuit(self.path, x, y, yaw, speed, now)
            if i >= self.main_n - 3:
                self._set('STOP', now)
                return 0.0, 1.0, 0.0, D
            acc, brk = self._speed(self._path_speed(i, p['rpark_v_enter_fast'], p['rpark_v_enter']), v, dt)
            return acc, brk, steer, D
        if st == 'STOP':
            if self._stopped(v, now):
                self.y_tgt = self.slot_c() + (CAR_LEN / 2 - REAR_OVERHANG)
                self.path = self._reverse_path(x, y)
                self.rev_tol = 0.15                                  # 후진 정지 앞당김: 0.3이면 약 0.13 m 덜 들어가 보정이 한 번 필요했음
                self.wall_stop = True
                self._set('SHIFT_R', now)
            return 0.0, 1.0, 0.0, D
        if st == 'SHIFT_R':
            if now - self.t_state > 0.4:
                self._set('REVERSE', now)
            return 0.0, 1.0, 0.0, R
        if st == 'REVERSE':
            steer, _ = self._reverse_pursuit(self.path, x, y, yaw, speed, now)
            left = self.y_tgt - y
            gap = self._rear_gap(self.scan_now)
            if gap is not None and gap < 6.0 and self.wall_stop:
                self.wall_y = y + REAR_OVERHANG + gap            # 안쪽 벽의 y
                # 목표 = 벽 실측으로 잡은 슬롯 중심에 차 중심이 오는 위치 (통로 기준 추정값보다 정확)
                left = (self.wall_y - p['rpark_wall_to_c'] + (CAR_LEN / 2 - REAR_OVERHANG)) - y
            if left <= self.rev_tol:
                self._set('STOP_REV', now)
                return 0.0, 1.0, 0.0, R
            v_t = max(0.5, min(p['rpark_v_reverse'], 0.5 + 1.1 * left))      # 끝에 가까울수록 부드럽게 감속
            acc, brk = self._speed(v_t, v, dt)
            return acc, brk, steer, R
        if st == 'STOP_REV':
            if self._stopped(v, now):
                ok, a, b, psi = self._in_slot(x, y, yaw)
                self.log(f'rear: 정지 벽y={self.wall_y} x={x:.2f} y={y:.2f} a={a:.2f} b={b:.2f} psi={math.degrees(psi):.1f}도 안={ok} 시도 {self.tries}')
                if ok:
                    self._set('HOLD', now)
                else:
                    if self.tries >= 3:
                        # 보정 한도 초과: 더 시도하지 않고 멈춘 자리에서 완료 처리 (예전엔 여기서 RuntimeError로 영구 멈춤)
                        self.log('rear: 후면 주차 보정 한도 초과 - 지금 자리에서 완료 처리')
                        self._set('HOLD', now)
                        return 0.0, 1.0, 0.0, R
                    self.tries += 1
                    hl = p['rpark_slot_len'] / 2 - 0.05 - p['rpark_tol_margin']
                    hw = p['rpark_slot_wid'] / 2 - 0.05 - p['rpark_tol_margin']
                    cs = self.corners(a, b, psi)
                    front_over = max(c[0] for c in cs) - hl           # 앞(남쪽)으로 삐져나온 양
                    rear_over = -min(c[0] for c in cs) - hl           # 뒤(북쪽)로 삐져나온 양
                    lat_ok = all(abs(c[1]) <= hw for c in cs)
                    if lat_ok and front_over > 0 and rear_over <= 0.0:   # 조금 덜 들어감: 제자리에서 더 후진
                        self.y_tgt = y + front_over + 0.1
                        self.path = self._reverse_path(x, y)
                        self.rev_tol = 0.03
                        self.wall_stop = False
                        self._set('SHIFT_R', now)
                    elif lat_ok and rear_over > 0 and front_over <= 0.0:  # 너무 들어감: 조금 앞으로
                        self.path = build_path((x, y, -math.pi / 2), [('S', rear_over + 0.15)])
                        self.main_n = len(self.path)
                        self.path += [(self.path[-1][0], self.path[-1][1] - 0.1 * k, -math.pi / 2) for k in range(1, 31)]
                        self._set('REDO_SHIFT', now)
                    else:                                           # 옆·방향이 틀림: 앞으로 나와 다시
                        self.path = build_path((x, y, -math.pi / 2), [('S', 3.5)])
                        self.main_n = len(self.path)
                        self.path += [(self.path[-1][0], self.path[-1][1] - 0.1 * k, -math.pi / 2) for k in range(1, 31)]
                        self._set('REDO_SHIFT', now)
            return 0.0, 1.0, 0.0, R
        if st == 'REDO_SHIFT':
            if now - self.t_state > 0.4:
                self._set('REDO', now)
            return 0.0, 1.0, 0.0, D
        if st == 'REDO':
            steer, i = self._forward_pursuit(self.path, x, y, yaw, speed, now)
            if i >= self.main_n - 3:
                self._set('STOP', now)
                return 0.0, 1.0, 0.0, D
            acc, brk = self._speed(0.8, v, dt)
            return acc, brk, steer, D
        if st == 'HOLD':
            if self.t_hold0 is None:
                self.t_hold0 = now
            if now - self.t_hold0 >= p['park_hold_s']:
                self.path = self._exit_path(x, y)
                self._set('EXIT_SHIFT_D', now)
            return 0.0, 1.0, 0.0, R
        if st == 'EXIT_SHIFT_D':
            if now - self.t_state > 0.4:
                self._set('EXIT_FWD', now)
            return 0.0, 1.0, 0.0, D
        if st == 'EXIT_FWD':
            steer, i = self._forward_pursuit(self.path, x, y, yaw, speed, now)
            if i >= self.main_n - 3:
                self._set('DONE', now)
                return 0.0, 0.0, 0.0, D
            acc, brk = self._speed(self._path_speed(i, p['rpark_v_exit_fast'], p['rpark_v_exit']), v, dt)
            return acc, brk, steer, D
        return 0.0, 1.0, 0.0, D



class ParkingMission(MissionBase):
    NAME = 'parking'
    SECTIONS = (('cp8', 'cp9'), ('cp9', 'cp10'))

    def reset(self):
        self.phase = 'idle'          # 예: 'search' -> 'align' -> 'reverse' -> 'stop' -> 'exit' -> 'done'
        params = getattr(self, 'parameters', {})
        self.ppark = Parking(params, log=(self.log.info if self.log else None))
        self.rpark = RearParking(params, log=(self.log.info if self.log else None))

    def on_enter(self, section):
        self.phase = 'idle'
        self.ppark.reset()
        self.rpark.reset()

    def update(self, inp: MissionInput) -> MissionOutput:
        options = getattr(self, 'options', None)
        if options is None:
            options = load_options()
        name = 'parallel_parking' if inp.cp.section == ('cp8','cp9') else 'rear_parking'
        if not option_on(options, name):
            return MissionOutput(is_done=True)
        if inp.cp.section == ('cp8', 'cp9'):
            return self.parallel_parking(inp)
        return self.rear_parking(inp)

    def parallel_parking(self, inp: MissionInput) -> MissionOutput:
        pk = self.ppark
        if pk.done:
            return MissionOutput(is_done=True)
        accel, brake, steer, gear = pk.update(inp.x, inp.y, inp.yaw, inp.speed, inp.scan, inp.local_path, inp.t, inp.scan_pose)
        if pk.done:                                  # 탈출까지 끝남: 기본 경로로 돌아감
            return MissionOutput(is_done=True)
        return MissionOutput(override_cmd=(accel, brake, steer, 'R' if gear == GEAR_REVERSE else 'D'))

    def rear_parking(self, inp: MissionInput) -> MissionOutput:
        rp = self.rpark
        if rp.done:
            return MissionOutput(is_done=True)
        accel, brake, steer, gear = rp.update(inp.x, inp.y, inp.yaw, inp.speed, inp.scan, inp.local_path, inp.t, inp.scan_pose)
        if rp.done:                                  # 탈출까지 끝남: 기본 경로로 돌아감
            return MissionOutput(is_done=True)
        return MissionOutput(override_cmd=(accel, brake, steer, 'R' if gear == GEAR_REVERSE else 'D'))
