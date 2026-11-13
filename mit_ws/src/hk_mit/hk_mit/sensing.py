"""Timestamp alignment shared by missions. No ground-truth or simulator internals."""
import math
from collections import deque


def stamp(message):
    s = message.header.stamp
    return s.sec + s.nanosec * 1e-9


def fresh(message, now, timeout, since=-math.inf):
    return message is not None and since - 1e-6 <= stamp(message) <= now + 0.05 and now - stamp(message) <= timeout


class PoseHistory:
    def __init__(self):
        self.rows = deque()

    def clear(self):
        self.rows.clear()

    def add(self, message):
        t = stamp(message)
        p, q = message.pose.pose.position, message.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        if self.rows and t <= self.rows[-1][0]:
            return
        self.rows.append((t, p.x, p.y, yaw))
        while self.rows and t - self.rows[0][0] > 1.5:
            self.rows.popleft()

    def at(self, t):
        if not self.rows or t < self.rows[0][0] - 0.03 or t > self.rows[-1][0] + 1e-6:
            return None
        previous = self.rows[0]
        for row in self.rows:
            if row[0] >= t:
                a = 0.0 if row[0] == previous[0] else max(0.0, min(1.0, (t-previous[0])/(row[0]-previous[0])))
                dh = math.atan2(math.sin(row[3]-previous[3]), math.cos(row[3]-previous[3]))
                return (previous[1]+a*(row[1]-previous[1]), previous[2]+a*(row[2]-previous[2]), previous[3]+a*dh)
            previous = row
        return previous[1:]


def predict_motion(x, y, yaw, speed, now, delay, history, acceleration=0.0):
    """Predict signed forward/reverse motion using commands already in flight."""
    if delay <= 0.0:
        return x, y, yaw
    n = max(1, int(math.ceil(delay / 0.02)))
    dt = delay / n
    i, delta = 0, 0.0
    for k in range(n):
        command_time = now - delay + k * dt
        while i < len(history) and history[i][0] <= command_time:
            delta = history[i][1]
            i += 1
        v = speed + acceleration * (k+0.5) * dt
        if v * speed < 0.0:
            v = 0.0
        dyaw = v / 3.0 * math.tan(delta) * dt
        mid = yaw + dyaw * 0.5
        x += v * math.cos(mid) * dt
        y += v * math.sin(mid) * dt
        yaw += dyaw
    return x, y, yaw
