"""Rehab training counter engine."""
import math
import time

FX, FY, CX, CY = 525.0, 525.0, 320.0, 240.0
DEPTH_DIFF_THRESH_MM = 150  # 关节段倾斜阈值（3D长度-2D投影长度），约对应60度倾斜

STATE_EXTENDED = 'extended'
STATE_FLEXING = 'flexing'
STATE_FLEXED = 'flexed'
STATE_EXTENDING = 'extending'

EXERCISES = {
    'left_elbow':  ('\u5de6\u8098\u5c48\u4f38', 5, 7, 9, 70, 150, 20, 175, [(5, 7), (7, 9)]),
    'right_elbow': ('\u53f3\u8098\u5c48\u4f38', 6, 8, 10, 70, 150, 20, 175, [(6, 8), (8, 10)]),
    'left_knee':   ('\u5de6\u819d\u5c48\u4f38', 11, 13, 15, 90, 160, 10, 175, [(11, 13), (13, 15)]),
    'right_knee':  ('\u53f3\u819d\u5c48\u4f38', 12, 14, 16, 90, 160, 10, 175, [(12, 14), (14, 16)]),
}


def pixel_to_3d(x, y, depth_mm, fx=FX, fy=FY, cx=CX, cy=CY):
    z = depth_mm / 1000.0
    return ((x - cx) * z / fx, (y - cy) * z / fy, z)


def calc_angle(p1, p2, p3):
    n = len(p2)
    v1 = [p1[i] - p2[i] for i in range(n)]
    v2 = [p3[i] - p2[i] for i in range(n)]
    dot = sum(v1[i] * v2[i] for i in range(n))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))


class RepCounter:
    def __init__(self, flex_thresh, ext_thresh):
        self.flex_thresh = flex_thresh
        self.ext_thresh = ext_thresh
        self.state = STATE_EXTENDED
        self.reps = 0
        self.just_completed = False

    def update(self, angle):
        self.just_completed = False
        if self.state == STATE_EXTENDED:
            if angle < self.flex_thresh:
                self.state = STATE_FLEXING
        elif self.state == STATE_FLEXING:
            if angle < self.flex_thresh - 10:
                self.state = STATE_FLEXED
            elif angle > self.ext_thresh:
                self.state = STATE_EXTENDED
        elif self.state == STATE_FLEXED:
            if angle > self.flex_thresh + 10:
                self.state = STATE_EXTENDING
        elif self.state == STATE_EXTENDING:
            if angle > self.ext_thresh:
                self.state = STATE_EXTENDED
                self.reps += 1
                self.just_completed = True
            elif angle < self.flex_thresh - 10:
                self.state = STATE_FLEXED

    def reset(self):
        self.state = STATE_EXTENDED
        self.reps = 0
        self.just_completed = False


class RehabEngine:
    def __init__(self, exercise_key='left_elbow', target_reps=10, target_sets=3):
        if exercise_key not in EXERCISES:
            raise ValueError(f"\u672a\u77e5\u52a8\u4f5c: {exercise_key}, \u53ef\u7528: {list(EXERCISES.keys())}")
        self.exercise_key = exercise_key
        name, p1, p2, p3, flex_t, ext_t, min_safe, max_safe, depth_segs = EXERCISES[exercise_key]
        self.name = name
        self.p1_idx = p1
        self.p2_idx = p2
        self.p3_idx = p3
        self.min_safe = min_safe
        self.max_safe = max_safe
        self.depth_segments = depth_segs
        self.target_reps = target_reps
        self.target_sets = target_sets
        self.counter = RepCounter(flex_t, ext_t)
        self.sets_done = 0
        self.current_angle = None
        self.current_angle_3d = None
        self.current_depth_diff_mm = None
        self.current_depth_seg = None
        self.bad_form_this_rep = False
        self.bad_count = 0
        self.total_bad_count = 0
        self.completed = False
        self._last_feedback_reps = 0
        self._last_warning_time = 0
        self._rest_until = 0
        self._last_rest_feedback_time = 0
        self._depth_bad_frames = 0

    def update(self, keypoints):
        feedback = None
        now = time.time()
        if self.completed:
            return self.current_angle, self.counter.reps, self.counter.state, None

        resting = now < self._rest_until
        if resting:
            if now - self._last_rest_feedback_time >= 5:
                self._last_rest_feedback_time = now
                remaining = int(self._rest_until - now)
                feedback = f"\u4f11\u606f\u4e2d\uff0c\u8fd8\u5269 {remaining} \u79d2"

        self.current_angle_3d = None
        self.current_depth_diff_mm = None
        self.current_depth_seg = None

        depth_check = self._check_segment_depth_align(keypoints)
        if depth_check is not None:
            a, b, diff = depth_check
            self.current_depth_diff_mm = diff
            self.current_depth_seg = f"{a}-{b}"

        def get_pt(idx):
            kp = keypoints[idx]
            if isinstance(kp, dict):
                sc = kp.get('score', 0)
                if sc < 0.3:
                    return None
                d = kp.get('depth_mm', 0)
                if 300 <= d <= 8000:
                    return pixel_to_3d(kp['x'], kp['y'], d)
                return (kp['x'], kp['y'])
            x, y, sc = kp
            if sc < 0.3:
                return None
            return (x, y)

        p1 = get_pt(self.p1_idx)
        p2 = get_pt(self.p2_idx)
        p3 = get_pt(self.p3_idx)
        if p1 is None or p2 is None or p3 is None:
            self.current_angle = None
            return None, self.counter.reps, self.counter.state, feedback

        dims = {len(p) for p in (p1, p2, p3)}
        if len(dims) > 1:
            p1, p2, p3 = p1[:2], p2[:2], p3[:2]

        angle = calc_angle(p1, p2, p3)
        self.current_angle = angle
        self.current_angle_3d = angle
        if angle is None:
            return None, self.counter.reps, self.counter.state, feedback

        angle_bad = angle < self.min_safe or angle > self.max_safe
        if angle_bad and now - self._last_warning_time > 3:
            self._last_warning_time = now
            feedback = f"\u6ce8\u610f\uff0c{self.name}\u89d2\u5ea6\u5f02\u5e38\uff0c\u8bf7\u8c03\u6574\u59ff\u52bf"

        depth_bad = (self.current_depth_diff_mm is not None
                     and self.current_depth_diff_mm > DEPTH_DIFF_THRESH_MM)
        if depth_bad:
            self._depth_bad_frames += 1
        else:
            self._depth_bad_frames = 0

        if not resting:
            prev_state = self.counter.state
            self.counter.update(angle)

            if prev_state == STATE_EXTENDED and self.counter.state == STATE_FLEXING:
                self.bad_form_this_rep = False

            if self._depth_bad_frames >= 3:
                self.bad_form_this_rep = True
                if now - self._last_warning_time > 2:
                    self._last_warning_time = now
                    feedback = "\u624b\u81c2\u524d\u540e\u504f\u79fb\u8fc7\u5927\uff0c\u8bf7\u4fdd\u6301\u624b\u81c2\u4e0e\u8eab\u4f53\u5e73\u884c"

            if self.counter.just_completed:
                if self.bad_form_this_rep:
                    self.counter.reps -= 1
                    self.bad_count += 1
                    self.total_bad_count += 1
                    self.bad_form_this_rep = False
                    if feedback is None:
                        feedback = "\u59ff\u52bf\u4e0d\u6807\u51c6\uff0c\u672c\u6b21\u4e0d\u8ba1\u5165"
                else:
                    reps = self.counter.reps
                    if reps >= self.target_reps:
                        self.sets_done += 1
                        self.counter.reset()
                        self._last_feedback_reps = 0
                        self.bad_count = 0
                        if self.sets_done >= self.target_sets:
                            feedback = f"\u8bad\u7ec3\u5b8c\u6210\uff01\u5171\u5b8c\u6210 {self.sets_done} \u7ec4\uff0c\u505a\u5f97\u5f88\u597d"
                            self.completed = True
                            self._rest_until = now + 60
                        else:
                            feedback = f"\u7b2c {self.sets_done} \u7ec4\u5b8c\u6210\uff0c\u4f11\u606f30\u79d2"
                            self._rest_until = now + 30
                    elif reps % 5 == 0 and reps != self._last_feedback_reps:
                        self._last_feedback_reps = reps
                        if feedback is None:
                            feedback = f"\u5df2\u5b8c\u6210 {reps} \u6b21\uff0c\u7ee7\u7eed\u4fdd\u6301"

        return angle, self.counter.reps, self.counter.state, feedback

    def _check_segment_depth_align(self, keypoints):
        """检查关节段是否严重倾斜（不垂直于摄像头）。

        逻辑：计算 RGB 2D 投影长度 vs 3D 实际长度。
        若 3D 长度远大于 2D 投影，说明这段关节朝摄像头倾斜了。
        返回 (idx_a, idx_b, 3D长度 - 2D长度)，取最差的那段。
        """
        if not keypoints or not isinstance(keypoints[0], dict):
            return None
        worst = None
        for a, b in self.depth_segments:
            ka, kb = keypoints[a], keypoints[b]
            if ka.get('score', 0) < 0.3 or kb.get('score', 0) < 0.3:
                continue
            # 2D 投影长度（RGB 像素空间）
            x1, y1 = ka.get('x', 0), ka.get('y', 0)
            x2, y2 = kb.get('x', 0), kb.get('y', 0)
            len_2d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # 3D 实际长度（深度空间，单位 mm）
            da, db = ka.get('depth_mm', 0), kb.get('depth_mm', 0)
            if da <= 0 or db <= 0:
                continue
            p3d_a = pixel_to_3d(x1, y1, da)
            p3d_b = pixel_to_3d(x2, y2, db)
            len_3d_m = math.sqrt(sum((p3d_b[i] - p3d_a[i])**2 for i in range(3)))
            len_3d_mm = len_3d_m * 1000  # 转毫米

            # 2D 投影缩短量 = 3D 实际长度 - 2D 投影长度对应的 3D 估算
            # 粗略估算：假设深度 z ≈ (da + db)/2，用相似三角形反推 2D 对应的 3D 长度
            z_avg = (da + db) / 2000.0  # 转米
            len_2d_in_3d_mm = len_2d * z_avg / FX * 1000  # 像素→米→毫米

            # 差值 = 3D 实际 - 2D 投影对应的 3D，越大说明越倾斜
            tilt_mm = len_3d_mm - len_2d_in_3d_mm

            if worst is None or tilt_mm > worst[2]:
                worst = (a, b, tilt_mm)
        return worst

    def get_status(self):
        return {
            'exercise': self.name,
            'angle': self.current_angle,
            'angle_3d': self.current_angle_3d,
            'depth_diff_mm': self.current_depth_diff_mm,
            'depth_diff_seg': self.current_depth_seg,
            'bad_form': self.bad_form_this_rep,
            'bad_count': self.bad_count,
            'total_bad_count': self.total_bad_count,
            'reps': self.counter.reps,
            'sets': self.sets_done,
            'target_reps': self.target_reps,
            'target_sets': self.target_sets,
            'state': self.counter.state,
            'resting': time.time() < self._rest_until,
            'completed': self.completed,
        }
