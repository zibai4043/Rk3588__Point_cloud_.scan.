"""姿态关键点的时序平滑与行人检测降频。

为 true_total.py 提供两类能力，目的是提升计数准确度：
  1) OneEuroFilter / KeypointSmoother —— 对逐帧关键点做 1€ 滤波，
     消除 RTMPose 输出的抖动，避免角度跳变导致的误计/漏计。
  2) PersonDetector —— 把每帧都跑的 HOG 行人检测改成每 N 帧跑一次、
     其余帧复用缓存 bbox，提升帧率。

本模块为纯逻辑（仅依赖 math / 可选注入检测函数），便于单元测试。
"""
import math

# ── 默认参数（板子端可按需微调）────────────────────────────────
# 1€ 滤波：min_cutoff 越小越平滑但延迟越大；beta 越大对快速运动越跟手。
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.3
ONE_EURO_D_CUTOFF = 1.0

MIN_KP_SCORE = 0.3   # 与下游 sc > 0.3 门控保持一致
NUM_KEYPOINTS = 17

DETECT_INTERVAL = 6  # 每 N 帧真正跑一次行人检测，其余帧复用缓存


def _alpha(cutoff, freq):
    """由截止频率和采样频率计算一阶低通的平滑系数。"""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * freq)


class LowPassFilter:
    """一阶指数低通滤波，OneEuroFilter 的内部构件。"""

    def __init__(self):
        self._prev = None

    def __call__(self, value, alpha):
        if self._prev is None:
            self._prev = value
        else:
            self._prev = alpha * value + (1.0 - alpha) * self._prev
        return self._prev

    @property
    def last(self):
        return self._prev

    def reset(self):
        self._prev = None


class OneEuroFilter:
    """1€ 滤波器：低速时强平滑去抖，高速时降低延迟跟手。

    参考 Casiez et al. "1€ Filter" (CHI 2012)。对单个标量信号工作，
    需要外部提供时间戳（秒）以计算实际采样频率。
    """

    def __init__(self, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA,
                 d_cutoff=ONE_EURO_D_CUTOFF):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._last_t = None

    def __call__(self, value, t):
        value = float(value)
        if self._last_t is None or t <= self._last_t:
            # 首帧或时间戳异常：直接吃进当前值，不做导数估计
            freq = 30.0
        else:
            freq = 1.0 / (t - self._last_t)
        self._last_t = t

        prev = self._x.last
        dx = 0.0 if prev is None else (value - prev) * freq
        edx = self._dx(dx, _alpha(self.d_cutoff, freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(value, _alpha(cutoff, freq))

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._last_t = None


class KeypointSmoother:
    """对 17 个关键点逐点做 1€ 平滑。

    规则：
      - score >= MIN_KP_SCORE：平滑 x/y，并记住最新平滑值；
      - score < MIN_KP_SCORE：不喂进滤波器（避免向噪声收敛），
        若此前有平滑值则沿用其坐标，否则原样返回；
      - score 始终原样透传，保证下游 sc > 0.3 门控行为不变。
    输出 (int x, int y, float score)，与推理输出格式一致。
    """

    def __init__(self, num_keypoints=NUM_KEYPOINTS, min_score=MIN_KP_SCORE,
                 min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA):
        self.min_score = float(min_score)
        self._fx = [OneEuroFilter(min_cutoff, beta) for _ in range(num_keypoints)]
        self._fy = [OneEuroFilter(min_cutoff, beta) for _ in range(num_keypoints)]
        self._last = [None] * num_keypoints

    def apply(self, keypoints, t):
        if not keypoints:
            return keypoints
        out = []
        for i, (x, y, sc) in enumerate(keypoints):
            if i >= len(self._fx):
                out.append((int(x), int(y), sc))
                continue
            if sc >= self.min_score:
                sx = self._fx[i](x, t)
                sy = self._fy[i](y, t)
                self._last[i] = (sx, sy)
                out.append((int(round(sx)), int(round(sy)), sc))
            elif self._last[i] is not None:
                sx, sy = self._last[i]
                out.append((int(round(sx)), int(round(sy)), sc))
            else:
                out.append((int(x), int(y), sc))
        return out

    def reset(self):
        for f in self._fx:
            f.reset()
        for f in self._fy:
            f.reset()
        self._last = [None] * len(self._last)


class PersonDetector:
    """行人检测降频复用。

    每 interval 帧真正调用 detect_fn 一次，其余帧返回缓存 bbox；
    从未检出时返回整帧 [0, 0, w, h]。detect_fn 注入便于单测，
    在 true_total.py 中由 _detect_person 提供（HOG）。

    detect_fn(frame) 应返回 [x1, y1, x2, y2]，与缓存/整帧格式一致。
    """

    def __init__(self, detect_fn, interval=DETECT_INTERVAL):
        self.detect_fn = detect_fn
        self.interval = max(1, int(interval))
        self._cache = None
        self._frame_count = 0

    def get_bbox(self, frame):
        run_now = (self._cache is None) or (self._frame_count % self.interval == 0)
        self._frame_count += 1
        if run_now:
            bbox = self.detect_fn(frame)
            if bbox is not None:
                self._cache = bbox
        if self._cache is not None:
            return self._cache
        h, w = frame.shape[:2]
        return [0, 0, w, h]

    def reset(self):
        self._cache = None
        self._frame_count = 0
