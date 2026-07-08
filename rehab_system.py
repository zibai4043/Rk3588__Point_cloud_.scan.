#!/usr/bin/env python3
"""
true_total.py - 板子端一体化真实计数版：
采集 OpenNI2 深度 + UVC RGB，使用 RTMPose 推理（RKNN NPU 优先，ONNX CPU 兜底），
并集成 RehabEngine、Flask Web(5001) 和语音对话（VAD + ASR + Chat + TTS）。

与旧 total.py 的区别：
- 移除假计数变量 _fake_reps/_fake_sets/_fake_bad/_red_until。
- /status 返回 RehabEngine 的真实计数。
- 遥控器移除 +1/-1 手动计数按钮，新增暂停/继续按钮。
- API 密钥后续建议从环境变量读取。

运行：
LD_LIBRARY_PATH=/usr/lib:/usr/lib/OpenNI2/Drivers python3 /mnt/sdcard/my_first_code/true_total.py
"""
import array
import collections
import ctypes
import datetime
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string, request

from rehab_engine import RehabEngine, EXERCISES, DEPTH_DIFF_THRESH_MM

try:
    import bt_audio
    _BT_AVAILABLE = True
except Exception as _bt_err:
    bt_audio = None
    _BT_AVAILABLE = False
    print(f'[BT] bt_audio 模块加载失败，蓝牙功能禁用: {_bt_err}')

try:
    from sht40_sensor import SHT40
    _SHT40_AVAILABLE = True
except Exception as _sht_err:
    SHT40 = None
    _SHT40_AVAILABLE = False
    print(f'[SHT40] sht40_sensor 模块加载失败，温湿度监控禁用: {_sht_err}')

# 配置

# 从同目录的 secrets.env 文件加载密钥到环境变量（若存在）。
# 该文件不应提交到版本库，格式为每行 KEY=VALUE。
def _load_secrets_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'secrets.env')
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # 已在真实环境变量中设置的优先，不覆盖
                if key and key not in os.environ:
                    os.environ[key] = val
        print('[Config] loaded secrets.env')
    except Exception as e:
        print(f'[Config] failed to load secrets.env: {e}')


_load_secrets_env()

FLASK_PORT   = 5001
COLOR_DEVICE = '/dev/video21'
RKNN_MODEL   = '/mnt/sdcard/my_first_code/rtmpose_m_256x192.rknn'
ONNX_MODEL   = '/mnt/sdcard/my_first_code/rtmpose-m.onnx'
MODEL_H, MODEL_W = 256, 192
HISTORY_FILE = '/mnt/sdcard/my_first_code/history.json'
PROFILE_FILE = '/mnt/sdcard/my_first_code/profile.json'

DEPTH_MIN, DEPTH_MAX = 300, 8000

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

ARK_API_KEY   = os.environ.get('ARK_API_KEY',   '')
CHAT_MODEL    = 'doubao-seed-2-0-lite-260215'
SYSTEM_PROMPT = '你是一个嵌入式康复设备上的语音助手，可以联网搜索实时信息。请用中文简短回复，不超过80字。说"再见"可以结束对话。'
ALIYUN_APPKEY        = os.environ.get('ALIYUN_APPKEY',        '')
ALIYUN_ACCESS_KEY    = os.environ.get('ALIYUN_ACCESS_KEY',    '')
ALIYUN_ACCESS_SECRET = os.environ.get('ALIYUN_ACCESS_SECRET', '')
ALSA_DEVICE     = 'plughw:1,0'
RECORD_RATE     = 16000
RECORD_CH       = 2
MAX_HISTORY     = 10
VAD_SILENCE_THRESH = 7000   # 调高：原 5000，现 7000（高于环境噪音 5500）
VAD_SPEECH_THRESH  = 12000  # 调高：原 6000，现 12000（低于说话音量 25000）
VAD_SILENCE_SECS   = 0.7    # 调低：原 1.0，说完更快结束录音
VAD_MAX_SECS       = 10
VAD_MIN_SECS       = 0.5
VAD_CHUNK_FRAMES   = 1600

ONI_STATUS_OK    = 0
ONI_SENSOR_DEPTH = 1
ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION = 1
ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR = 1

# OpenNI2 结构体定义

class OniVideoMode(ctypes.Structure):
    _fields_ = [('pixelFormat', ctypes.c_int), ('resolutionX', ctypes.c_int),
                ('resolutionY', ctypes.c_int), ('fps', ctypes.c_int)]

class OniFrame(ctypes.Structure):
    _fields_ = [
        ('dataSize', ctypes.c_int), ('_reserved', ctypes.c_int),
        ('data', ctypes.c_void_p), ('sensorType', ctypes.c_int),
        ('_pad1', ctypes.c_int), ('timestamp', ctypes.c_uint64),
        ('frameIndex', ctypes.c_int), ('width', ctypes.c_int),
        ('height', ctypes.c_int), ('videoMode', OniVideoMode),
        ('croppingEnabled', ctypes.c_int), ('cropOriginX', ctypes.c_int),
        ('cropOriginY', ctypes.c_int), ('stride', ctypes.c_int),
    ]

# 全局共享状态

_lock              = threading.Lock()
_latest_depth      = None   # np.uint16 (H,W)
_latest_color      = None   # np.uint8 BGR
_latest_frame_jpeg = None   # bytes，彩色视频叠加骨架后的 JPEG
_latest_depth_jpeg = None   # bytes，深度/点云画面的 JPEG
_latest_status     = {}
_rehab             = None

# === 真实计数状态 ===
_training_paused = False  # 训练暂停状态
_latest_keypoints = None  # 最新一帧关键点

_depth_azimuth   = 0     # 点云水平旋转角，-90~90 度
_depth_elevation = 30    # 点云俯仰角，0~89 度
_pc_skel_red     = False # 点云骨骼红色开关
_depth_kp_offset_x = float(os.environ.get('DEPTH_KP_OFFSET_X', '-35'))
_depth_kp_offset_y = float(os.environ.get('DEPTH_KP_OFFSET_Y', '0'))
_depth_kp_scale_x  = float(os.environ.get('DEPTH_KP_SCALE_X', '1.0'))
_depth_kp_scale_y  = float(os.environ.get('DEPTH_KP_SCALE_Y', '1.0'))
_depth_kp_mirror_x = os.environ.get('DEPTH_KP_MIRROR_X', '0') == '1'
_last_depth_shape_log = None
_cam_stop     = threading.Event()
_speak_lock   = threading.Lock()
_voice_lock   = threading.Lock()   # 语音线程专用锁，保护 _voice_log/_train_log/_voice_active
_speak_done   = threading.Event()  # TTS 完成信号
_speaking_event = threading.Event()
_token_cache  = {'token': None, 'expire': 0}
_chat_history = []
_voice_log    = collections.deque(maxlen=10)  # 语音对话日志（系统/用户/AI，自动播报）
_train_log    = collections.deque(maxlen=20)  # 训练反馈日志
_voice_active = False

# SHT40 温湿度传感器状态
_latest_temp     = None  # float (°C)
_latest_humidity = None  # float (%)
_sht40_sensor    = None  # SHT40 实例
_sht40_i2c_bus = int(os.environ.get('SHT40_I2C_BUS', '7'))
_sht40_i2c_addr = int(os.environ.get('SHT40_I2C_ADDR', '0x44'), 16)
_temp_warning_threshold = float(os.environ.get('TEMP_WARNING_THRESHOLD', '35.0'))  # 温度警告阈值
_sht40_error_count = 0
_sht40_max_errors = 5  # 连续失败 5 次后禁用
_sht40_last_error_msg = None
_sht40_auto_disabled = False

# 蓝牙音频状态：由 /bt_connect、/bt_disconnect 更新，供录音/播放动态选设备
_bt_state = {
    'mac': None,               # 当前连接的蓝牙 MAC，None 表示用板载
    'playback': ALSA_DEVICE,   # aplay -D 用的设备名
    'capture': ALSA_DEVICE,    # arecord -D 用的设备名
    'capture_ch': RECORD_CH,   # 录音声道数（板载双声道取右声道；蓝牙HFP单声道）
}
_bt_lock = threading.Lock()

# OpenNI2 初始化

def _load_openni2():
    for path in ('/usr/lib/libOpenNI2.so', '/openni2/libOpenNI2.so',
                 '/mnt/sdcard/openni2_arm64/libOpenNI2.so'):
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def _register_oni(oni):
    oni.oniInitialize.restype  = ctypes.c_int
    oni.oniInitialize.argtypes = [ctypes.c_int]
    oni.oniDeviceOpen.restype  = ctypes.c_int
    oni.oniDeviceOpen.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    oni.oniDeviceCreateStream.restype  = ctypes.c_int
    oni.oniDeviceCreateStream.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_void_p)]
    oni.oniStreamStart.restype  = ctypes.c_int
    oni.oniStreamStart.argtypes = [ctypes.c_void_p]
    oni.oniStreamStop.argtypes  = [ctypes.c_void_p]
    oni.oniStreamReadFrame.restype  = ctypes.c_int
    oni.oniStreamReadFrame.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.POINTER(OniFrame))]
    oni.oniFrameRelease.argtypes  = [ctypes.POINTER(OniFrame)]
    oni.oniStreamDestroy.argtypes = [ctypes.c_void_p]
    oni.oniDeviceClose.argtypes   = [ctypes.c_void_p]
    oni.oniShutdown.argtypes      = []
    oni.oniDeviceSetProperty.restype  = ctypes.c_int
    oni.oniDeviceSetProperty.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_int]

# 摄像头读取线程

def _depth_reader(oni, stream):
    global _latest_depth
    frame_ptr = ctypes.POINTER(OniFrame)()
    while not _cam_stop.is_set():
        rc = oni.oniStreamReadFrame(stream, ctypes.byref(frame_ptr))
        if rc != ONI_STATUS_OK or not frame_ptr:
            time.sleep(0.01)
            continue
        f = frame_ptr.contents
        w, h = f.width, f.height
        if w > 0 and h > 0 and f.dataSize == w * h * 2:
            arr = np.ctypeslib.as_array(
                (ctypes.c_uint16 * (w * h)).from_address(f.data)
            ).reshape(h, w).copy()
            with _lock:
                _latest_depth = arr
        oni.oniFrameRelease(frame_ptr)
        time.sleep(0.1)


def _color_reader(cap):
    global _latest_color
    while not _cam_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with _lock:
            _latest_color = frame
        time.sleep(0.033)


def _sht40_reader():
    """SHT40 温湿度传感器读取线程（带智能错误处理）"""
    global _latest_temp, _latest_humidity, _sht40_sensor
    global _sht40_error_count, _sht40_last_error_msg, _sht40_auto_disabled

    if not _SHT40_AVAILABLE:
        return

    try:
        _sht40_sensor = SHT40(bus_num=_sht40_i2c_bus, addr=_sht40_i2c_addr)
        print(f'[SHT40] 传感器已初始化 (I2C 总线 {_sht40_i2c_bus}, 地址 0x{_sht40_i2c_addr:02x})')
    except Exception as e:
        print(f'[SHT40] 初始化失败: {e}')
        print(f'[SHT40] 诊断提示:')
        print(f'  1. 检查 I2C 总线: ls /dev/i2c-*')
        print(f'  2. 扫描设备: i2cdetect -y {_sht40_i2c_bus}')
        print(f'  3. 检查权限: sudo chmod 666 /dev/i2c-{_sht40_i2c_bus}')
        print(f'  4. 配置总线号: export SHT40_I2C_BUS=<总线号>')
        return

    while not _cam_stop.is_set():
        try:
            result = _sht40_sensor.read_temp_humidity()
            if result:
                temp_c, humidity = result
                with _lock:
                    _latest_temp = temp_c
                    _latest_humidity = humidity

                # 读取成功，重置错误计数
                if _sht40_error_count > 0:
                    print(f'[SHT40] 恢复正常')
                    _sht40_error_count = 0
                    _sht40_auto_disabled = False

                # 温度过高警告
                if temp_c > _temp_warning_threshold:
                    print(f'[SHT40] 警告：温度过高 {temp_c:.1f}°C')
                    _log_train(f'环境温度过高 {temp_c:.1f}°C，注意休息')
            else:
                # 读取失败（传感器无响应或通信错误）
                _sht40_error_count += 1
                if _sht40_error_count == 1:
                    print(f'[SHT40] 读取失败（传感器无响应），将继续重试...')

        except Exception as e:
            error_msg = str(e)
            _sht40_error_count += 1

            # 仅在错误变化或首次失败时打印
            if error_msg != _sht40_last_error_msg:
                print(f'[SHT40] 读取错误: {e}')
                _sht40_last_error_msg = error_msg

            # 连续失败过多，自动禁用
            if _sht40_error_count >= _sht40_max_errors and not _sht40_auto_disabled:
                print(f'[SHT40] 连续失败 {_sht40_max_errors} 次，自动禁用温湿度监控')
                print(f'[SHT40] 可通过以下方式重新启动程序:')
                print(f'  export SHT40_I2C_BUS=<正确的总线号>')
                print(f'  python3 true_total.py')
                _sht40_auto_disabled = True
                break  # 退出线程

        time.sleep(2.0)

    # 清理
    if _sht40_sensor:
        try:
            _sht40_sensor.close()
            print('[SHT40] 传感器已关闭')
        except:
            pass

# 推理引擎初始化

_infer_fn   = None   # callable(bgr_frame) -> list of (x,y,score)
_infer_mode = 'none'

def _init_inference():
    global _infer_fn, _infer_mode
    # 尝试 RKNN NPU
    if os.path.exists(RKNN_MODEL):
        try:
            from rknnlite.api import RKNNLite
            rknn = RKNNLite()
            ret = rknn.load_rknn(RKNN_MODEL)
            if ret == 0:
                ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret == 0:
                def _rknn_infer(frame):
                    blob, meta = _preprocess(frame, _detect_person(frame))
                    # rknn.inference 接受 list[ndarray]，blob shape 为 (1,3,H,W)
                    outputs = rknn.inference(inputs=[blob[0]])
                    return _decode_simcc(outputs, meta)
                _infer_fn   = _rknn_infer
                _infer_mode = 'rknn'
                print('[Infer] RKNN NPU ready')
                return
            else:
                print(f'[Infer] RKNN init failed (ret={ret}), falling back')
        except Exception as e:
            print(f'[Infer] RKNN load error: {e}, falling back')

    # ONNX CPU 兜底
    if os.path.exists(ONNX_MODEL):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(ONNX_MODEL, providers=['CPUExecutionProvider'])
            iname = sess.get_inputs()[0].name
            def _onnx_infer(frame):
                blob, meta = _preprocess(frame, _detect_person(frame))
                outputs = sess.run(None, {iname: blob})
                return _decode_simcc(outputs, meta)
            _infer_fn   = _onnx_infer
            _infer_mode = 'onnx'
            print('[Infer] ONNX CPU ready')
            return
        except Exception as e:
            print(f'[Infer] ONNX load error: {e}')

    print('[Infer] WARNING: no model loaded, using dummy keypoints')
    _infer_fn   = lambda frame: [(0, 0, 0.0)] * 17
    _infer_mode = 'dummy'


# RTMPose 预处理 / 后处理

_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

_PREPROC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PREPROC_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BBOX_CACHE_FRAMES = 15
_bbox_cache = {'bbox': None, 'age': _BBOX_CACHE_FRAMES}  # age >= limit → re-detect


def _detect_person(frame):
    global _bbox_cache
    _bbox_cache['age'] += 1
    if _bbox_cache['age'] < _BBOX_CACHE_FRAMES and _bbox_cache['bbox'] is not None:
        return _bbox_cache['bbox']
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    rects, _ = _hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
    if len(rects) == 0:
        bbox = [0, 0, w, h]
    else:
        rects = [[x*2, y*2, bw*2, bh*2] for x, y, bw, bh in rects]
        x, y, bw, bh = max(rects, key=lambda r: r[2]*r[3])
        pad = 20
        bbox = [max(0, x-pad), max(0, y-pad), min(w, x+bw+pad), min(h, y+bh+pad)]
    _bbox_cache['bbox'] = bbox
    _bbox_cache['age'] = 0
    return bbox


def _preprocess(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame; x1 = y1 = 0
    resized = cv2.resize(crop, (MODEL_W, MODEL_H))
    blob = resized.astype(np.float32) / 255.0
    blob = (blob - _PREPROC_MEAN) / _PREPROC_STD
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    sx = (x2 - x1) / MODEL_W
    sy = (y2 - y1) / MODEL_H
    return blob, (x1, y1, sx, sy)


def _decode_simcc(outputs, meta):
    ox, oy, sx, sy = meta
    simcc_x = outputs[0] if outputs[0].ndim == 2 else outputs[0][0]
    simcc_y = outputs[1] if outputs[1].ndim == 2 else outputs[1][0]
    kx = np.argmax(simcc_x, axis=1) / 2.0
    ky = np.argmax(simcc_y, axis=1) / 2.0
    scores = (np.max(simcc_x, axis=1) + np.max(simcc_y, axis=1)) / 2.0
    return [(int(kx[i]*sx+ox), int(ky[i]*sy+oy), float(scores[i]))
            for i in range(len(kx))]


def _align_depth_to_color(depth_frame, color_frame):
    global _last_depth_shape_log
    if depth_frame is None or color_frame is None:
        return depth_frame
    dh, dw = depth_frame.shape[:2]
    ch, cw = color_frame.shape[:2]
    if dh == ch and dw == cw:
        return depth_frame
    shape_key = (dw, dh, cw, ch)
    if _last_depth_shape_log != shape_key:
        print(f'[Camera] depth/color size mismatch: depth={dw}x{dh}, color={cw}x{ch}; resizing depth')
        _last_depth_shape_log = shape_key
    return cv2.resize(depth_frame, (cw, ch), interpolation=cv2.INTER_NEAREST)


def _map_keypoints_to_depth(keypoints, depth_frame):
    if keypoints is None or depth_frame is None:
        return keypoints
    h, w = depth_frame.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    mapped = []
    for x, y, sc in keypoints:
        x = float(x)
        y = float(y)
        if _depth_kp_mirror_x:
            x = w - 1 - x
        dx = (x - cx) * _depth_kp_scale_x + cx + _depth_kp_offset_x
        dy = (y - cy) * _depth_kp_scale_y + cy + _depth_kp_offset_y
        dx = max(0, min(w - 1, int(round(dx))))
        dy = max(0, min(h - 1, int(round(dy))))
        mapped.append((dx, dy, sc))
    return mapped


def _get_3d_joints(keypoints, depth_frame):
    if depth_frame is None:
        return []
    h, w = depth_frame.shape
    joints = []
    for idx, (x, y, sc) in enumerate(keypoints):
        x = max(0, min(int(x), w-1))
        y = max(0, min(int(y), h-1))
        x0, x1 = max(0, x-4), min(w, x+5)
        y0, y1 = max(0, y-4), min(h, y+5)
        patch = depth_frame[y0:y1, x0:x1].ravel()
        valid = patch[patch > 0]
        d = int(np.median(valid)) if len(valid) > 0 else int(depth_frame[y, x])
        joints.append({'x': x, 'y': y, 'depth_mm': d, 'score': sc, 'idx': idx})
    return joints


def _depth_to_pointcloud_jpeg(depth_frame, az=0, el=30, keypoints=None, red=False):
    if depth_frame is None:
        return None
    if keypoints is not None and len(keypoints) < 17:
        keypoints = None
    h, w = depth_frame.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    nonzero = depth_frame > 0
    if np.any(nonzero):
        vals = depth_frame[nonzero].astype(np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax > vmin:
            t = (vals - vmin) / (vmax - vmin)
            blue = (120 + t * 135).astype(np.uint8)
            green = (20 + t * 140).astype(np.uint8)
        else:
            blue = np.full(vals.shape, 200, dtype=np.uint8)
            green = np.full(vals.shape, 80, dtype=np.uint8)
        out[nonzero, 0] = blue
        out[nonzero, 1] = green
        out[nonzero, 2] = 20

    if keypoints is not None:
        red_color = (0, 0, 255)
        kp_color = [red_color] * 17 if red else [
            (0, 220, 255), (0, 220, 255), (0, 220, 255), (0, 220, 255), (0, 220, 255),
            (255, 255, 255), (255, 255, 255),
            (0, 255, 100), (100, 255, 0), (0, 200, 80), (80, 200, 0),
            (255, 255, 255), (255, 255, 255),
            (255, 140, 0), (200, 80, 255), (255, 100, 0), (160, 60, 255),
        ]
        skel_color = {
            (0, 1): (0, 220, 255), (0, 2): (0, 220, 255), (1, 3): (0, 220, 255), (2, 4): (0, 220, 255),
            (5, 6): (255, 255, 255), (5, 11): (255, 255, 255), (6, 12): (255, 255, 255), (11, 12): (255, 255, 255),
            (5, 7): (0, 255, 100), (7, 9): (0, 200, 80), (6, 8): (100, 255, 0), (8, 10): (80, 200, 0),
            (11, 13): (255, 140, 0), (13, 15): (255, 100, 0), (12, 14): (200, 80, 255), (14, 16): (160, 60, 255),
        }
        for i, (x, y, sc) in enumerate(keypoints[:17]):
            if sc > 0.3:
                cv2.circle(out, (int(x), int(y)), 4, kp_color[i], -1, cv2.LINE_AA)
        for a, b in SKELETON:
            if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3:
                color = red_color if red else skel_color.get((a, b), (200, 200, 200))
                cv2.line(out,
                         (int(keypoints[a][0]), int(keypoints[a][1])),
                         (int(keypoints[b][0]), int(keypoints[b][1])),
                         color, 2, cv2.LINE_AA)

    dx = int(max(-90.0, min(90.0, float(az))) / 90.0 * w * 0.2)
    dy = int((30.0 - max(0.0, min(89.0, float(el)))) / 59.0 * h * 0.15)
    if dx or dy:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if red:
        cv2.rectangle(out, (4, 4), (w - 5, h - 5), (0, 0, 255), 4)
        # 警告文字已移除，仅保留红色边框提示
    _, jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


def _draw_skeleton(frame, keypoints, red=False, color_override=None):
    """绘制骨架线。color_override 优先于 red 参数，用于 RGB 画面高亮骨架。"""
    if len(keypoints) < 17:
        return
    if color_override:
        dot_color  = color_override
        line_color = color_override
    else:
        dot_color  = (0, 0, 255) if red else (0, 255, 0)
        line_color = (0, 0, 255) if red else (0, 180, 255)
    for x, y, sc in keypoints:
        if sc > 0.3:
            cv2.circle(frame, (x, y), 5, dot_color, -1)
    for a, b in SKELETON:
        if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3:
            cv2.line(frame, keypoints[a][:2], keypoints[b][:2], line_color, 2)

# 推理主循环线程

def _processing_loop():
    global _latest_frame_jpeg, _latest_depth_jpeg, _latest_status, _latest_keypoints, _rehab
    print('[Processing] started')
    while True:
        with _lock:
            color = _latest_color.copy() if _latest_color is not None else None
            depth = _latest_depth.copy() if _latest_depth is not None else None
            az = _depth_azimuth
            el = _depth_elevation

        if color is None:
            time.sleep(0.05)
            continue

        depth_for_color = _align_depth_to_color(depth, color)

        try:
            keypoints = _infer_fn(color)
        except Exception as e:
            print(f'[Processing] infer error: {e}')
            time.sleep(0.1)
            continue

        with _lock:
            _latest_keypoints = keypoints

        frame = color.copy()
        _draw_skeleton(frame, keypoints, color_override=(0, 255, 255))
        depth_keypoints = _map_keypoints_to_depth(keypoints, depth_for_color)

        status = {}
        completed = False
        feedback_text = None
        with _lock:
            rehab = _rehab

        if rehab is not None and not _training_paused:
            joints_3d = _get_3d_joints(depth_keypoints, depth_for_color)
            kp_input = joints_3d if joints_3d else keypoints

            with _lock:
                if rehab is _rehab:
                    _, _, _, feedback_text = rehab.update(kp_input)
                    status = rehab.get_status()
                    _latest_status = status
                    completed = status.get('completed', False)
                    if completed:
                        _rehab = None
                else:
                    feedback_text = None
                    status = {}
                    completed = False

            if feedback_text:
                print(f'[Feedback] {feedback_text}')
                _log_train(feedback_text)

                if completed:
                    _save_history(status['exercise'],
                                  status['sets'],
                                  status['sets'] * status['target_reps'],
                                  status.get('total_bad_count', status['bad_count']))

        with _lock:
            status_for_depth = dict(_latest_status)
            manual_red = _pc_skel_red
        depth_diff = status_for_depth.get('depth_diff_mm')
        red_now = (bool(status_for_depth.get('bad_form')) or
                   (depth_diff is not None and depth_diff > DEPTH_DIFF_THRESH_MM) or
                   manual_red)
        depth_jpeg = _depth_to_pointcloud_jpeg(depth_for_color, az=az, el=el, keypoints=depth_keypoints, red=red_now)
        if depth_jpeg:
            with _lock:
                _latest_depth_jpeg = depth_jpeg

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            _latest_frame_jpeg = jpeg.tobytes()

        time.sleep(0.033)

def _camera_init():
    oni = _load_openni2()
    if oni is None:
        print('[Camera] libOpenNI2.so not found, depth disabled')
    else:
        _register_oni(oni)
        rc = oni.oniInitialize(2)
        if rc != ONI_STATUS_OK:
            print(f'[Camera] oniInitialize failed: {rc}')
            oni = None

    depth_stream = None
    if oni is not None:
        device = ctypes.c_void_p()
        rc = oni.oniDeviceOpen(None, ctypes.byref(device))
        if rc == ONI_STATUS_OK:
            ds = ctypes.c_void_p()
            rc = oni.oniDeviceCreateStream(device, ONI_SENSOR_DEPTH, ctypes.byref(ds))
            if rc == ONI_STATUS_OK:
                rc = oni.oniStreamStart(ds)
                if rc == ONI_STATUS_OK:
                    depth_stream = ds
                    reg_val = ctypes.c_int(ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
                    reg_rc = oni.oniDeviceSetProperty(
                        device,
                        ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION,
                        ctypes.byref(reg_val),
                        ctypes.sizeof(reg_val)
                    )
                    print(f'[Camera] Depth-to-Color registration rc={reg_rc}')
                    print('[Camera] OpenNI2 depth stream started')

    cap = cv2.VideoCapture(COLOR_DEVICE)
    if not cap.isOpened():
        cap = cv2.VideoCapture(21)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f'[Camera] RGB opened: {COLOR_DEVICE}')
        threading.Thread(target=_color_reader, args=(cap,), daemon=True).start()
    else:
        print(f'[Camera] WARNING: cannot open RGB {COLOR_DEVICE}')

    if depth_stream is not None:
        threading.Thread(target=_depth_reader, args=(oni, depth_stream), daemon=True).start()

    # 启动 SHT40 温湿度传感器线程
    if _SHT40_AVAILABLE:
        threading.Thread(target=_sht40_reader, daemon=True).start()

# 时间同步

app = Flask(__name__)

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>&#24247;&#22797;&#35757;&#32451;&#31995;&#32479;</title><style>

:root{--bg:#faf8f5;--surface:#fffdfa;--text:#2f2a24;--muted:#91877a;--line:#ece5dc;--soft:#f3eee7;--primary:#b8642a;--danger:#c75b4d;--ok:#557b5f;--ease:cubic-bezier(.16,1,.3,1)}
*{box-sizing:border-box;margin:0;padding:0}body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}header{height:58px;padding:0 28px;display:flex;align-items:center;gap:22px;border-bottom:1px solid rgba(236,229,220,.7);background:rgba(250,248,245,.92);backdrop-filter:blur(16px)}h1{font-size:1rem;font-weight:620}.logo{display:flex;align-items:baseline;gap:10px}.logo-mark{width:9px;height:9px;border-radius:50%;background:var(--primary)}nav{display:flex;gap:18px;margin-left:auto}nav a{text-decoration:none;color:var(--muted);font-size:.78rem}nav a.active,nav a:hover{color:var(--text)}.toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,16px);background:#2f2a24;color:#fff;padding:10px 18px;border-radius:999px;font-size:.84rem;opacity:0;pointer-events:none;transition:opacity .35s var(--ease),transform .35s var(--ease)}.toast.show{opacity:1;transform:translate(-50%,0)}

html,body{height:100%}body{height:100vh;display:grid;grid-template-rows:auto 1fr;overflow:hidden}.hbadge{font-size:.68rem;color:var(--muted)}.conn-row{display:flex;align-items:center;gap:7px;font-size:.72rem;color:var(--muted)}.dot{width:7px;height:7px;border-radius:50%;background:#c9c0b5}.dot.on{background:var(--ok)}main{min-height:0;padding:20px 24px 24px;display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:20px;overflow:hidden}.stage{position:relative;min-height:0;border-radius:18px;background:#0f0f0f;overflow:hidden;border:2px solid transparent;transition:border-color .35s var(--ease),box-shadow .35s var(--ease)}.stage.alert{border-color:var(--danger);box-shadow:0 0 0 4px rgba(199,91,77,.16)}.stage img{width:100%;height:100%;object-fit:cover;display:block}.stage-label{position:absolute;left:18px;top:16px;color:rgba(255,255,255,.82);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase}.depth-pip{position:absolute;right:18px;bottom:18px;width:28%;min-width:220px;aspect-ratio:4/3;border-radius:14px;overflow:hidden;background:#111;box-shadow:0 18px 48px rgba(0,0,0,.22)}.depth-pip img{width:100%;height:100%;object-fit:cover}.depth-pip .stage-label{left:12px;top:10px;font-size:.64rem}.sidebar{min-height:0;display:flex;flex-direction:column;gap:14px;overflow:auto;padding-right:2px}.surface{background:var(--surface);border-radius:16px;padding:18px}.progress-card{padding:22px 20px}.eyebrow{font-size:.64rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px}.progress-main{display:grid;grid-template-columns:1fr 1fr;gap:18px}.prog-num{font-size:4.05rem;font-weight:360;line-height:.9;font-variant-numeric:tabular-nums;color:var(--text);transition:transform .34s var(--ease)}.prog-num.bump{animation:bump .42s var(--ease)}@keyframes bump{0%{transform:scale(1)}35%{transform:scale(1.045)}100%{transform:scale(1)}}.prog-total{font-size:1.34rem;color:var(--muted);font-weight:360}.prog-lbl{font-size:.64rem;color:var(--muted);margin-top:10px}.prog-bar{height:4px;background:var(--soft);border-radius:999px;overflow:hidden;margin-top:12px}.prog-fill{height:100%;width:0;background:var(--primary);border-radius:999px;transition:width .7s var(--ease)}.config-card{padding:13px 15px}.cfg-row{display:grid;grid-template-columns:minmax(0,1fr) 66px 66px auto;gap:9px;align-items:end}.fld{display:flex;flex-direction:column;gap:5px}.fld label,.slabel{font-size:.62rem;color:var(--muted)}select,input{height:36px;background:transparent;border:1px solid var(--line);border-radius:9px;color:var(--text);padding:0 10px;font-size:.8rem;outline:none}.btn-start{height:36px;border:0;border-radius:9px;background:var(--primary);color:#fff;padding:0 15px;font-size:.82rem;font-weight:560;cursor:pointer;transition:transform .28s var(--ease),background .28s var(--ease)}.btn-start:active{transform:scale(.96)}.stats-grid{display:grid;grid-template-columns:1fr 1fr;column-gap:18px;row-gap:16px}.slabel{margin-bottom:5px}.sval{font-size:1.06rem;font-weight:520;color:var(--text);font-variant-numeric:tabular-nums}.sval.red,.stat-item.alert .sval{color:var(--danger)}.sval.orange{color:var(--primary)}.voice-card{flex:1;min-height:160px;display:flex;flex-direction:column}.voice-head{display:flex;align-items:center;justify-content:space-between}.voice-state{font-size:.66rem;color:var(--muted)}.vlog{flex:1;overflow:auto;display:flex;flex-direction:column;gap:9px;padding-right:2px}.vi{font-size:.8rem;line-height:1.55;color:var(--text);padding-left:10px;border-left:1px solid var(--line)}.vi strong{font-weight:620}.vi-s{color:var(--muted)}@media(max-width:980px){main{grid-template-columns:1fr;overflow:auto}.stage{min-height:52vh}.sidebar{overflow:visible}.depth-pip{width:34%;min-width:150px}body{overflow:auto}}
</style></head><body>
<header><div class="logo"><span class="logo-mark"></span><h1>&#24247;&#22797;&#35757;&#32451;&#31995;&#32479;</h1></div><span class="hbadge">&#33258;&#21160;&#35745;&#25968;&#27169;&#24335;</span><nav><a href="/" class="active">&#20027;&#39029;</a><a href="/library">&#35757;&#32451;&#24211;</a><a href="/profile">&#20010;&#20154;&#20013;&#24515;</a><a href="/remote">&#36965;&#25511;&#22120;</a><a href="/settings">&#35774;&#32622;</a></nav><div class="conn-row"><span class="dot" id="dot"></span><span id="ctext">&#36830;&#25509;&#20013;</span></div></header>
<main><section class="stage" id="depth-stage"><img id="depth-feed" src="/depth_feed" alt="&#28145;&#24230;&#20998;&#26512;"><div class="stage-label">Depth Analysis</div><div class="depth-pip"><img id="rgb-feed" src="/video_feed" alt="&#23454;&#26102;&#30011;&#38754;"><div class="stage-label">RGB</div></div></section><aside class="sidebar">
<section class="surface progress-card"><div class="eyebrow">&#35757;&#32451;&#36827;&#24230;</div><div class="progress-main"><div><div class="prog-num" id="s-reps">0<span class="prog-total"> / 0</span></div><div class="prog-lbl">&#26412;&#36718;&#27425;&#25968;</div><div class="prog-bar"><div class="prog-fill" id="rbar"></div></div></div><div><div class="prog-num" id="s-sets">0<span class="prog-total"> / 0</span></div><div class="prog-lbl">&#23436;&#25104;&#32452;&#25968;</div><div class="prog-bar"><div class="prog-fill" id="sbar"></div></div></div></div></section>
<section class="surface config-card"><form id="cfg-form"><div class="cfg-row"><div class="fld"><label>&#21160;&#20316;</label><select name="exercise_key">{% for key, val in exercises.items() %}<option value="{{ key }}">{{ val[0] }}</option>{% endfor %}</select></div><div class="fld"><label>&#27425;&#25968;</label><input type="number" name="target_reps" value="10" min="1" max="50"></div><div class="fld"><label>&#32452;&#25968;</label><input type="number" name="target_sets" value="3" min="1" max="10"></div><button type="submit" class="btn-start" id="start-btn">&#24320;&#22987;</button></div></form></section>
<section class="surface"><div class="eyebrow">&#23454;&#26102;&#25968;&#25454;</div><div class="stats-grid"><div class="stat-item" id="stat-ex"><div class="slabel">&#24403;&#21069;&#21160;&#20316;</div><div class="sval" id="s-ex">--</div></div><div class="stat-item" id="stat-st"><div class="slabel">&#36816;&#21160;&#29366;&#24577;</div><div class="sval" id="s-st">--</div></div><div class="stat-item" id="stat-ang"><div class="slabel">&#20851;&#33410;&#35282;&#24230;</div><div class="sval" id="s-ang">--</div></div><div class="stat-item" id="stat-dev"><div class="slabel">&#28145;&#24230;&#20559;&#24046;</div><div class="sval" id="s-dev">--</div></div><div class="stat-item" id="stat-bad"><div class="slabel">&#19981;&#26631;&#20934;&#27425;&#25968;</div><div class="sval" id="s-bad">0</div></div><div class="stat-item" id="stat-mode"><div class="slabel">&#31995;&#32479;&#29366;&#24577;</div><div class="sval" id="s-mode">&#27491;&#24120;</div></div><div class="stat-item" id="stat-temp"><div class="slabel">&#29615;&#22659;&#28201;&#24230;</div><div class="sval" id="s-temp">--</div></div><div class="stat-item" id="stat-hum"><div class="slabel">&#29615;&#22659;&#28287;&#24230;</div><div class="sval" id="s-hum">--</div></div></div></section>
<section class="surface voice-card"><div class="voice-head"><div class="eyebrow">&#26234;&#33021;&#21161;&#25163;</div><span class="voice-state" id="vstatus">&#24453;&#26426;</span></div><div class="vlog" id="vlog"><div class="vi vi-s">&#35828;&#8220;&#20320;&#22909;&#23567;&#26126;&#8221;&#21796;&#37266;&#21161;&#25163;</div></div></section></aside></main>
<script>
const SL={extended:'\u4f38\u5c55',flexing:'\u5c48\u66f2\u4e2d',flexed:'\u5c48\u66f2',extending:'\u4f38\u5c55\u4e2d'};let lastReps=null,lastSets=null;function setText(id,text,cls){const e=document.getElementById(id);if(!e)return;e.textContent=text;e.className='sval '+(cls||'')}function flag(id,on){const e=document.getElementById(id);if(e)e.className='stat-item '+(on?'alert':'')}function setProgress(id,value,total){const e=document.getElementById(id);if(e)e.innerHTML=value+'<span class="prog-total"> / '+total+'</span>'}function bump(id){const e=document.getElementById(id);if(!e)return;e.classList.remove('bump');void e.offsetWidth;e.classList.add('bump')}async function poll(){try{const d=await(await fetch('/status')).json();document.getElementById('dot').className='dot on';document.getElementById('ctext').textContent='\u5df2\u8fde\u63a5';if(!d.exercise){setText('s-mode','\u5f85\u5f00\u59cb');return}setText('s-ex',d.exercise);setText('s-st',d.paused?'\u5df2\u6682\u505c':(SL[d.state]||d.state||'--'),d.paused?'orange':'');const ang=d.angle,dev=d.depth_diff_mm,bad=d.bad_count||0;const angBad=ang!=null&&(ang<20||ang>175),devBad=dev!=null&&dev>80;setText('s-ang',ang!=null?ang.toFixed(1)+'\u00b0':'--',angBad?'red':'');flag('stat-ang',angBad);setText('s-dev',dev!=null?dev.toFixed(0)+'mm':'--',devBad?'red':'');flag('stat-dev',devBad);setText('s-bad',String(bad),bad>0?'red':'');flag('stat-bad',bad>0);const temp=d.temperature,hum=d.humidity;setText('s-temp',temp!=null?temp.toFixed(1)+'\u00b0C':'--',d.temp_warning?'red':'');setText('s-hum',hum!=null?hum.toFixed(1)+'%':'--');flag('stat-temp',d.temp_warning);const reps=d.reps||0,tr=d.target_reps||1,sets=d.sets||0,ts=d.target_sets||1;if(lastReps!==null&&reps!==lastReps)bump('s-reps');if(lastSets!==null&&sets!==lastSets)bump('s-sets');lastReps=reps;lastSets=sets;setProgress('s-reps',reps,tr);setProgress('s-sets',sets,ts);document.getElementById('rbar').style.width=Math.min(100,reps/tr*100)+'%';document.getElementById('sbar').style.width=Math.min(100,sets/ts*100)+'%';document.getElementById('depth-stage').classList.toggle('alert', !!d.bad_form || devBad);setText('s-mode',d.completed?'\u5df2\u5b8c\u6210':(d.resting?'\u4f11\u606f\u4e2d':'\u6b63\u5e38'),(angBad||devBad||d.bad_form)?'red':'')}catch(e){document.getElementById('dot').className='dot';document.getElementById('ctext').textContent='\u672a\u8fde\u63a5'}}async function pollVoice(){try{const d=await(await fetch('/voice_status')).json();document.getElementById('vstatus').textContent=d.active?'\u5bf9\u8bdd\u4e2d':'\u5f85\u673a';const log=document.getElementById('vlog');if(d.log&&d.log.length){log.innerHTML=d.log.map(m=>{const name=m.role==='user'?'\u6211':m.role==='bot'?'AI':'\u7cfb\u7edf';const cls=m.role==='sys'?' vi-s':'';return '<div class="vi'+cls+'"><strong>'+name+'\uff1a</strong>'+m.text+'</div>'}).join('');log.scrollTop=log.scrollHeight}}catch(e){}}document.getElementById('cfg-form').addEventListener('submit',async e=>{e.preventDefault();const btn=document.getElementById('start-btn');btn.disabled=true;btn.textContent='\u542f\u52a8\u4e2d';try{const fd=new FormData(e.target);const d=await(await fetch('/start',{method:'POST',body:new URLSearchParams(fd)})).json();btn.textContent=d.ok?'\u91cd\u65b0\u5f00\u59cb':'\u5f00\u59cb'}catch(e){btn.textContent='\u5f00\u59cb'}btn.disabled=false});setInterval(poll,500);poll();setInterval(pollVoice,800);pollVoice();
</script></body></html>'''


PROFILE_HTML = r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>&#20010;&#20154;&#20013;&#24515;</title><style>
:root{--bg:#faf8f5;--surface:#fffdfa;--text:#2f2a24;--muted:#91877a;--line:#ece5dc;--soft:#f3eee7;--primary:#b8642a;--danger:#c75b4d;--ok:#557b5f;--ease:cubic-bezier(.16,1,.3,1)}
*{box-sizing:border-box;margin:0;padding:0}html,body{height:100%}body{height:100vh;display:grid;grid-template-rows:auto 1fr;overflow:hidden;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}header{height:58px;padding:0 28px;display:flex;align-items:center;gap:22px;border-bottom:1px solid rgba(236,229,220,.7);background:rgba(250,248,245,.92);backdrop-filter:blur(16px)}h1{font-size:1rem;font-weight:620}.logo{display:flex;align-items:baseline;gap:10px}.logo-mark{width:9px;height:9px;border-radius:50%;background:var(--primary)}nav{display:flex;gap:18px;margin-left:auto}nav a{text-decoration:none;color:var(--muted);font-size:.78rem}nav a.active,nav a:hover{color:var(--text)}.toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,16px);background:#2f2a24;color:#fff;padding:10px 18px;border-radius:999px;font-size:.84rem;opacity:0;pointer-events:none;transition:opacity .35s var(--ease),transform .35s var(--ease)}.toast.show{opacity:1;transform:translate(-50%,0)}
.page{min-height:0;padding:28px 24px;display:flex;flex-direction:column;gap:18px;overflow:auto}.surface{background:var(--surface);border-radius:16px;padding:22px}.eyebrow{font-size:.64rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:16px}.form-grid{display:grid;grid-template-columns:1fr 90px 1.2fr auto;gap:10px;align-items:end}.fld{display:flex;flex-direction:column;gap:5px}.fld label{font-size:.62rem;color:var(--muted)}input{height:38px;background:transparent;border:1px solid var(--line);border-radius:9px;color:var(--text);padding:0 10px;outline:none}.btn-save{height:38px;border:0;border-radius:9px;background:var(--primary);color:#fff;padding:0 18px;cursor:pointer}.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.snum{font-size:2.35rem;font-weight:360}.slbl{font-size:.64rem;color:var(--muted);margin-top:4px}table{width:100%;border-collapse:collapse;font-size:.84rem}th{text-align:left;color:var(--muted);font-size:.62rem;font-weight:520;text-transform:uppercase;letter-spacing:.08em;padding:8px 8px 12px}td{padding:13px 8px;border-top:1px solid var(--line)}.empty{color:var(--muted);font-size:.86rem;padding:30px 0;text-align:center}@media(max-width:760px){.form-grid,.stats-row{grid-template-columns:1fr}}</style></head><body><header><h1>&#24247;&#22797;&#35757;&#32451;&#31995;&#32479;</h1><nav><a href="/">&#20027;&#39029;</a><a href="/library">&#35757;&#32451;&#24211;</a><a href="/profile" class="active">&#20010;&#20154;&#20013;&#24515;</a><a href="/remote">&#36965;&#25511;&#22120;</a><a href="/settings">&#35774;&#32622;</a></nav></header><div class="page"><section class="surface"><div class="eyebrow">&#24739;&#32773;&#20449;&#24687;</div><form id="pform"><div class="form-grid"><div class="fld"><label>&#22995;&#21517;</label><input name="name" value="{{ profile.name }}" placeholder="&#35831;&#36755;&#20837;&#22995;&#21517;"></div><div class="fld"><label>&#24180;&#40836;</label><input name="age" value="{{ profile.age }}" type="number" min="1" max="120"></div><div class="fld"><label>&#35786;&#26029;</label><input name="diagnosis" value="{{ profile.diagnosis }}" placeholder="&#20363;&#22914;&#65306;&#33181;&#20851;&#33410;&#32622;&#25442;&#26415;&#21518;"></div><button class="btn-save" type="submit">&#20445;&#23384;</button></div></form></section><section class="surface"><div class="eyebrow">&#35757;&#32451;&#32479;&#35745;</div><div class="stats-row">{% set total_sets = history|sum(attribute='sets') %}{% set total_reps = history|sum(attribute='total_reps') %}{% set total_bad = history|sum(attribute='bad') %}{% set sessions = history|length %}<div><div class="snum">{{ sessions }}</div><div class="slbl">&#35757;&#32451;&#27425;&#25968;</div></div><div><div class="snum">{{ total_sets }}</div><div class="slbl">&#24635;&#32452;&#25968;</div></div><div><div class="snum">{{ total_reps }}</div><div class="slbl">&#24635;&#27425;&#25968;</div></div><div><div class="snum">{% if total_reps > 0 %}{{ (total_bad/total_reps*100)|int }}%{% else %}--{% endif %}</div><div class="slbl">&#19981;&#26631;&#20934;&#29575;</div></div></div></section><section class="surface"><div class="eyebrow">&#35757;&#32451;&#21382;&#21490;</div>{% if history %}<table><thead><tr><th>&#26102;&#38388;</th><th>&#21160;&#20316;</th><th>&#32452;&#25968;</th><th>&#27425;&#25968;</th><th>&#19981;&#26631;&#20934;</th></tr></thead><tbody>{% for h in history %}<tr><td>{{ h.time }}</td><td>{{ h.exercise }}</td><td>{{ h.sets }}</td><td>{{ h.total_reps }}</td><td>{{ h.bad }}</td></tr>{% endfor %}</tbody></table>{% else %}<div class="empty">&#26242;&#26080;&#35757;&#32451;&#35760;&#24405;</div>{% endif %}</section></div><div class="toast" id="toast"></div><script>function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');clearTimeout(toast._h);toast._h=setTimeout(()=>t.classList.remove('show'),1800)}document.getElementById('pform').addEventListener('submit',async e=>{e.preventDefault();try{const d=await(await fetch('/save_profile',{method:'POST',body:new URLSearchParams(new FormData(e.target))})).json();toast(d.ok?'\u4fe1\u606f\u5df2\u4fdd\u5b58':'\u4fdd\u5b58\u5931\u8d25')}catch{toast('\u7f51\u7edc\u5f02\u5e38')}});</script></body></html>'''


LIBRARY_HTML = r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>&#35757;&#32451;&#24211;</title><style>
:root{--bg:#faf8f5;--surface:#fffdfa;--text:#2f2a24;--muted:#91877a;--line:#ece5dc;--soft:#f3eee7;--primary:#b8642a;--danger:#c75b4d;--ok:#557b5f;--ease:cubic-bezier(.16,1,.3,1)}
*{box-sizing:border-box;margin:0;padding:0}html,body{height:100%}body{height:100vh;display:grid;grid-template-rows:auto 1fr;overflow:hidden;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}header{height:58px;padding:0 28px;display:flex;align-items:center;gap:22px;border-bottom:1px solid rgba(236,229,220,.7);background:rgba(250,248,245,.92);backdrop-filter:blur(16px)}h1{font-size:1rem;font-weight:620}.logo{display:flex;align-items:baseline;gap:10px}.logo-mark{width:9px;height:9px;border-radius:50%;background:var(--primary)}nav{display:flex;gap:18px;margin-left:auto}nav a{text-decoration:none;color:var(--muted);font-size:.78rem}nav a.active,nav a:hover{color:var(--text)}.toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,16px);background:#2f2a24;color:#fff;padding:10px 18px;border-radius:999px;font-size:.84rem;opacity:0;pointer-events:none;transition:opacity .35s var(--ease),transform .35s var(--ease)}.toast.show{opacity:1;transform:translate(-50%,0)}
.page{min-height:0;padding:30px 24px;display:flex;flex-direction:column;overflow:auto}.title{font-size:2rem;font-weight:360;margin-bottom:22px}.tabs{display:flex;gap:22px;margin-bottom:24px;flex-wrap:wrap}.tab{border:0;background:transparent;color:var(--muted);font-size:.9rem;cursor:pointer;padding:6px 0;border-bottom:2px solid transparent}.tab.active{color:var(--text);font-weight:620;border-bottom-color:var(--primary)}.panels{min-height:0;display:flex;flex-direction:column;flex:1}.panel{display:none;background:var(--surface);border-radius:18px;padding:28px;flex:1;min-height:0}.panel.active{display:grid;grid-template-columns:200px 1fr;gap:34px;flex:1}.demo-wrap{display:flex;flex-direction:column;gap:12px;color:var(--muted);font-size:.72rem;align-items:center}.demo-wrap svg{width:160px;height:160px}.panel-title{font-size:1.35rem;font-weight:560;margin-bottom:8px}.panel-desc{color:#665d53;line-height:1.7;font-size:.9rem;margin-bottom:20px}.steps{display:grid;gap:12px;margin-bottom:24px}.step{display:grid;grid-template-columns:32px 1fr;gap:12px}.step-num{color:var(--primary);font-size:.85rem;font-weight:600}.step-text{font-size:.88rem;line-height:1.55}.btn-start{height:42px;border:0;border-radius:9px;background:var(--primary);color:#fff;padding:0 22px;font-size:.9rem;font-weight:560;cursor:pointer;transition:transform .28s var(--ease),background .28s var(--ease)}.btn-start:hover{background:#a55822}.btn-start:active{transform:scale(.96)}@media(max-width:760px){.panel.active{grid-template-columns:1fr}.demo-wrap svg{width:120px;height:120px}}</style></head><body><header><h1>&#24247;&#22797;&#35757;&#32451;&#31995;&#32479;</h1><nav><a href="/">&#20027;&#39029;</a><a href="/library" class="active">&#35757;&#32451;&#24211;</a><a href="/profile">&#20010;&#20154;&#20013;&#24515;</a><a href="/remote">&#36965;&#25511;&#22120;</a><a href="/settings">&#35774;&#32622;</a></nav></header><div class="page"><div class="title">&#21160;&#20316;&#24211;</div><div class="tabs"><button class="tab active" onclick="switchTab(0)">&#24038;&#32920;&#23624;&#20280;</button><button class="tab" onclick="switchTab(1)">&#21491;&#32920;&#23624;&#20280;</button><button class="tab" onclick="switchTab(2)">&#24038;&#33181;&#23624;&#20280;</button><button class="tab" onclick="switchTab(3)">&#21491;&#33181;&#23624;&#20280;</button></div><div class="panels">{% set items=[('left_elbow','&#24038;&#32920;&#23624;&#20280;','&#36866;&#29992;&#20110;&#32920;&#20851;&#33410;&#26415;&#21518;&#24247;&#22797;&#21644;&#19978;&#32930;&#32908;&#21147;&#24674;&#22797;&#35757;&#32451;&#12290;','20&deg; - 175&deg;'),('right_elbow','&#21491;&#32920;&#23624;&#20280;','&#20445;&#25345;&#32937;&#37096;&#31283;&#23450;&#65292;&#32531;&#24930;&#23436;&#25104;&#23624;&#20280;&#12290;','20&deg; - 175&deg;'),('left_knee','&#24038;&#33181;&#23624;&#20280;','&#36866;&#29992;&#20110;&#33181;&#20851;&#33410;&#21151;&#33021;&#24674;&#22797;&#21644;&#19979;&#32930;&#31283;&#23450;&#24615;&#35757;&#32451;&#12290;','10&deg; - 175&deg;'),('right_knee','&#21491;&#33181;&#23624;&#20280;','&#21160;&#20316;&#35201;&#24179;&#31283;&#65292;&#36991;&#20813;&#31361;&#28982;&#21457;&#21147;&#12290;','10&deg; - 175&deg;')] %}{% for key,title,desc,angle in items %}<section class="panel {% if loop.first %}active{% endif %}"><div class="demo-wrap"><svg viewBox="0 0 100 100"><circle cx="50" cy="12" r="7" fill="#cfa57f"/><line x1="50" y1="20" x2="50" y2="48" stroke="#806f60" stroke-width="4" stroke-linecap="round"/><line x1="50" y1="48" x2="28" y2="36" stroke="#806f60" stroke-width="4" stroke-linecap="round"/><line x1="50" y1="48" x2="72" y2="36" stroke="#806f60" stroke-width="4" stroke-linecap="round"/><line x1="50" y1="48" x2="50" y2="72" stroke="#806f60" stroke-width="4" stroke-linecap="round"/><line x1="50" y1="72" x2="35" y2="92" stroke="#806f60" stroke-width="4" stroke-linecap="round"/><line x1="50" y1="72" x2="65" y2="92" stroke="#806f60" stroke-width="4" stroke-linecap="round"/><circle cx="50" cy="48" r="5" fill="#b8642a"/></svg><span>{{ angle }}</span></div><div><div class="panel-title">{{ title }}</div><div class="panel-desc">{{ desc }}</div><div class="steps"><div class="step"><div class="step-num">01</div><div class="step-text">&#20445;&#25345;&#36527;&#24178;&#31283;&#23450;&#65292;&#36827;&#20837;&#33298;&#36866;&#30340;&#36215;&#22987;&#23039;&#21183;&#12290;</div></div><div class="step"><div class="step-num">02</div><div class="step-text">&#32531;&#24930;&#23436;&#25104;&#23624;&#26354;&#65292;&#25269;&#36798;&#30446;&#26631;&#35282;&#24230;&#38468;&#36817;&#30701;&#26242;&#20572;&#30041;&#12290;</div></div><div class="step"><div class="step-num">03</div><div class="step-text">&#24179;&#31283;&#22238;&#21040;&#36215;&#22987;&#20301;&#32622;&#65292;&#36991;&#20813;&#20511;&#21147;&#21644;&#24555;&#36895;&#29993;&#21160;&#12290;</div></div></div><button class="btn-start" onclick="startEx('{{ key }}')">&#24320;&#22987;&#35757;&#32451;</button></div></section>{% endfor %}</div></div><div class="toast" id="toast"></div><script>function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');clearTimeout(toast._h);toast._h=setTimeout(()=>t.classList.remove('show'),1600)}function switchTab(n){document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===n));document.querySelectorAll('.panel').forEach((p,i)=>p.classList.toggle('active',i===n))}function startEx(key){fetch('/start',{method:'POST',body:new URLSearchParams({exercise_key:key,target_reps:10,target_sets:3})}).then(r=>r.json()).then(d=>{if(d.ok){toast('\u5df2\u9009\u62e9\uff0c\u8fd4\u56de\u4e3b\u9875');setTimeout(()=>location.href='/',700)}else toast(d.error||'\u542f\u52a8\u5931\u8d25')}).catch(()=>toast('\u7f51\u7edc\u5f02\u5e38'))}</script></body></html>'''


REMOTE_HTML = r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>&#36965;&#25511;&#22120;</title><style>
:root{--bg:#faf8f5;--surface:#fffdfa;--text:#2f2a24;--muted:#91877a;--line:#ece5dc;--soft:#f3eee7;--primary:#b8642a;--danger:#c75b4d;--ok:#557b5f;--ease:cubic-bezier(.16,1,.3,1)}
*{box-sizing:border-box;margin:0;padding:0}html,body{height:100%}body{height:100vh;display:grid;grid-template-rows:auto 1fr;overflow:hidden;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}header{height:58px;padding:0 28px;display:flex;align-items:center;gap:22px;border-bottom:1px solid rgba(236,229,220,.7);background:rgba(250,248,245,.92);backdrop-filter:blur(16px)}h1{font-size:1rem;font-weight:620}.logo{display:flex;align-items:baseline;gap:10px}.logo-mark{width:9px;height:9px;border-radius:50%;background:var(--primary)}nav{display:flex;gap:18px;margin-left:auto}nav a{text-decoration:none;color:var(--muted);font-size:.78rem}nav a.active,nav a:hover{color:var(--text)}.toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,16px);background:#2f2a24;color:#fff;padding:10px 18px;border-radius:999px;font-size:.84rem;opacity:0;pointer-events:none;transition:opacity .35s var(--ease),transform .35s var(--ease)}.toast.show{opacity:1;transform:translate(-50%,0)}
.wrap{min-height:0;padding:24px 18px;display:grid;grid-template-columns:1fr 1fr;gap:14px;overflow:auto}.panel{background:var(--surface);border-radius:16px;padding:18px;display:flex;flex-direction:column}.panel-title{font-size:.64rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px}.pause-panel{display:flex;align-items:center;gap:14px}.pause-state{font-size:1.25rem;font-weight:420}button{border:0;border-radius:11px;background:#f0e9e1;color:var(--text);cursor:pointer;transition:transform .28s var(--ease)}button:active{transform:scale(.96)}.pause-btn-big{height:48px;padding:0 34px;font-size:1rem}.pause-btn-big.paused{background:#dfeade;color:#31563a}.quick-btns{display:flex;flex-wrap:wrap;gap:9px}.qbtn,.toggle-btn{height:36px;padding:0 14px;font-size:.84rem}.toggle-btn.on{background:#f3d9d5;color:var(--danger)}.log-scroll{flex:1;min-height:120px;overflow:auto;display:flex;flex-direction:column;gap:8px}.log-item{font-size:.82rem;line-height:1.45;padding-left:10px;border-left:1px solid var(--line)}.role{font-weight:620}.play-btn{margin-left:8px;border:0;background:transparent;color:var(--primary)}.badge{color:var(--muted);font-size:.72rem}@media(max-width:900px){.wrap{grid-template-columns:1fr}}</style></head><body><header><h1>&#24247;&#22797;&#35757;&#32451;&#31995;&#32479;</h1><nav><a href="/">&#20027;&#39029;</a><a href="/library">&#35757;&#32451;&#24211;</a><a href="/profile">&#20010;&#20154;&#20013;&#24515;</a><a href="/remote" class="active">&#36965;&#25511;&#22120;</a><a href="/settings">&#35774;&#32622;</a></nav></header><div class="wrap"><section class="panel"><div class="panel-title">&#35757;&#32451;&#25511;&#21046;</div><div class="pause-panel"><span class="pause-state" id="pauseState">&#36827;&#34892;&#20013;</span><button class="pause-btn-big" id="pauseBtn" onclick="togglePause()">&#26242;&#20572;</button></div></section><section class="panel"><div class="panel-title">&#24555;&#25463;&#25773;&#25253;</div><div class="quick-btns"><button class="qbtn" onclick="playText('\u505a\u5f97\u5f88\u597d\uff0c\u7ee7\u7eed\u4fdd\u6301')">&#20570;&#24471;&#24456;&#22909;</button><button class="qbtn" onclick="playText('\u6ce8\u610f\u59ff\u52bf\uff0c\u7a0d\u5fae\u8c03\u6574\u4e00\u4e0b')">&#27880;&#24847;&#23039;&#21183;</button><button class="qbtn" onclick="playText('\u4f11\u606f\u4e00\u4e0b\uff0c\u8c03\u6574\u547c\u5438')">&#20241;&#24687;&#19968;&#19979;</button><button class="qbtn" onclick="playText('\u52a0\u6cb9\uff0c\u5feb\u5b8c\u6210\u4e86')">&#21152;&#27833;</button></div></section><section class="panel"><div class="panel-title">&#28857;&#20113;&#39592;&#39612;</div><button class="toggle-btn" id="pcSkelBtn" onclick="togglePcSkel()">&#27491;&#24120;&#33394;</button></section><section class="panel"><div class="panel-title">&#35757;&#32451;&#21453;&#39304;</div><div class="log-scroll" id="trainLog"><div class="badge">&#26242;&#26080;</div></div></section><section class="panel"><div class="panel-title">&#35821;&#38899;&#23545;&#35805; <span class="badge" id="voiceBadge">&#24453;&#26426;</span></div><div class="log-scroll" id="voiceLog"><div class="badge">&#35828;&#8220;&#20320;&#22909;&#23567;&#26126;&#8221;&#21796;&#37266;</div></div></section></div><div class="toast" id="toast"></div><script>function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');clearTimeout(toast._h);toast._h=setTimeout(()=>t.classList.remove('show'),1500)}function playText(text){fetch('/speak_text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})}).then(r=>r.json()).then(d=>toast(d.ok?'\u5df2\u64ad\u62a5':'\u64ad\u653e\u5931\u8d25')).catch(()=>toast('\u7f51\u7edc\u5f02\u5e38'))}function togglePause(){fetch('/pause_toggle',{method:'POST'}).then(r=>r.json()).then(d=>applyPause(d.paused)).catch(()=>toast('\u7f51\u7edc\u5f02\u5e38'))}function applyPause(paused){const btn=document.getElementById('pauseBtn'),state=document.getElementById('pauseState');btn.textContent=paused?'\u7ee7\u7eed':'\u6682\u505c';btn.classList.toggle('paused',paused);state.textContent=paused?'\u5df2\u6682\u505c':'\u8fdb\u884c\u4e2d';toast(paused?'\u8bad\u7ec3\u5df2\u6682\u505c':'\u8bad\u7ec3\u5df2\u7ee7\u7eed')}function togglePcSkel(){fetch('/toggle_pc_skel_red',{method:'POST'}).then(r=>r.json()).then(d=>{const btn=document.getElementById('pcSkelBtn');btn.textContent=d.red?'\u7ea2\u8272\u6a21\u5f0f':'\u6b63\u5e38\u8272';btn.classList.toggle('on',d.red);toast(d.red?'\u9aa8\u9abc\u5df2\u6807\u7ea2':'\u9aa8\u9abc\u6062\u590d\u6b63\u5e38')})}function updateTrain(){fetch('/train_log').then(r=>r.json()).then(d=>{const el=document.getElementById('trainLog');if(!d.log.length)return;el.innerHTML=d.log.slice().reverse().map(item=>`<div class="log-item"><span>${item.text}</span><button class="play-btn" onclick="playText('${item.text.replace(/'/g,"\\'")}')">\u64ad\u653e</button></div>`).join('')}).catch(()=>{})}function updateVoice(){fetch('/voice_status').then(r=>r.json()).then(d=>{document.getElementById('voiceBadge').textContent=d.active?'\u5bf9\u8bdd\u4e2d':'\u5f85\u673a';const el=document.getElementById('voiceLog');if(!d.log.length)return;el.innerHTML=d.log.map(item=>{const role=item.role==='user'?'\u6211':item.role==='bot'?'AI':'\u7cfb\u7edf';return `<div class="log-item"><span class="role">${role}\uff1a</span>${item.text}</div>`}).join('');el.scrollTop=el.scrollHeight}).catch(()=>{})}function syncPauseState(){fetch('/status').then(r=>r.json()).then(d=>applyPause(!!d.paused)).catch(()=>{})}setInterval(updateTrain,600);updateTrain();setInterval(updateVoice,900);updateVoice();setInterval(syncPauseState,2500);syncPauseState();</script></body></html>'''


SETTINGS_HTML = r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>设置</title><style>
:root{--bg:#faf8f5;--surface:#fffdfa;--text:#2f2a24;--muted:#91877a;--line:#ece5dc;--soft:#f3eee7;--primary:#b8642a;--danger:#c75b4d;--ok:#557b5f;--ease:cubic-bezier(.16,1,.3,1)}
*{box-sizing:border-box;margin:0;padding:0}html,body{height:100%}body{height:100vh;display:grid;grid-template-rows:auto 1fr;overflow:hidden;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{height:58px;padding:0 28px;display:flex;align-items:center;gap:22px;border-bottom:1px solid rgba(236,229,220,.7);background:rgba(250,248,245,.92);backdrop-filter:blur(16px)}h1{font-size:1rem;font-weight:620}nav{display:flex;gap:18px;margin-left:auto}nav a{text-decoration:none;color:var(--muted);font-size:.78rem}nav a.active,nav a:hover{color:var(--text)}
.toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,16px);background:#2f2a24;color:#fff;padding:10px 18px;border-radius:999px;font-size:.84rem;opacity:0;pointer-events:none;transition:opacity .35s var(--ease),transform .35s var(--ease);z-index:50}.toast.show{opacity:1;transform:translate(-50%,0)}
.wrap{min-height:0;padding:24px 18px;display:grid;grid-template-columns:1fr 1fr;gap:16px;overflow:auto}.panel{background:var(--surface);border-radius:16px;padding:20px;display:flex;flex-direction:column;min-height:0}
.phead{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}.ptitle{font-size:.64rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}.pico{font-size:1.05rem;font-weight:620;display:flex;align-items:center;gap:9px}.pico .d{width:9px;height:9px;border-radius:50%;background:#c9c0b5}.pico .d.on{background:var(--ok)}
.cur{background:var(--soft);border-radius:12px;padding:14px 16px;margin-bottom:16px}.cur-name{font-size:.95rem;font-weight:560}.cur-sub{font-size:.72rem;color:var(--muted);margin-top:3px}
.rowbtn{display:flex;gap:9px;margin-bottom:14px}button{border:0;border-radius:10px;background:#f0e9e1;color:var(--text);cursor:pointer;transition:transform .2s var(--ease),background .2s var(--ease);font-size:.84rem}button:active{transform:scale(.96)}button:disabled{opacity:.5;cursor:default}
.btn-scan{height:38px;padding:0 18px;font-weight:520}.btn-scan.busy{background:#e8ddd0}.btn-primary{background:var(--primary);color:#fff}.btn-danger{background:#f3d9d5;color:var(--danger)}
.list{flex:1;min-height:80px;overflow:auto;display:flex;flex-direction:column;gap:8px;padding-right:2px}
.dev{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 14px;border:1px solid var(--line);border-radius:11px;transition:border-color .2s}.dev:hover{border-color:#dcd2c6}.dev-l{min-width:0;flex:1}.dev-name{font-size:.86rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.dev-meta{font-size:.68rem;color:var(--muted);margin-top:2px}.dev-act{height:32px;padding:0 14px;font-size:.78rem;flex-shrink:0}.dev.connected{border-color:var(--ok);background:#f2f7f2}
.empty{color:var(--muted);font-size:.82rem;text-align:center;padding:24px 0}.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--line);border-top-color:var(--primary);border-radius:50%;animation:sp .7s linear infinite;vertical-align:-1px;margin-right:6px}@keyframes sp{to{transform:rotate(360deg)}}
.ap-row{display:flex;align-items:center;justify-content:space-between;padding:12px 0;border-top:1px solid var(--line);margin-top:6px}.ap-lbl{font-size:.82rem}.ap-sub{font-size:.68rem;color:var(--muted);margin-top:2px}.sw{width:46px;height:26px;border-radius:999px;background:#d8cec1;position:relative;cursor:pointer;transition:background .25s}.sw.on{background:var(--ok)}.sw::after{content:"";position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:#fff;transition:transform .25s var(--ease)}.sw.on::after{transform:translateX(20px)}
.pwd{display:none;gap:8px;margin-top:10px}.pwd.show{display:flex}.pwd input{flex:1;height:36px;border:1px solid var(--line);border-radius:9px;padding:0 12px;font-size:.82rem;background:transparent;outline:none}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
</style></head><body>
<header><h1>康复训练系统</h1><nav><a href="/">主页</a><a href="/library">训练库</a><a href="/profile">个人中心</a><a href="/remote">遥控器</a><a href="/settings" class="active">设置</a></nav></header>
<div class="wrap">
<section class="panel"><div class="phead"><div class="pico"><span class="d" id="bt-dot"></span>蓝牙</div><button class="btn-scan" id="bt-scan" onclick="btScan()">扫描设备</button></div>
<div class="cur" id="bt-cur"><div class="cur-name" id="bt-cur-name">未连接</div><div class="cur-sub" id="bt-cur-sub">连接蓝牙耳机后，语音播报将通过耳机输出</div></div>
<div class="list" id="bt-list"><div class="empty">点击「扫描设备」查找附近的蓝牙耳机</div></div></section>
<section class="panel"><div class="phead"><div class="pico"><span class="d" id="wf-dot"></span>Wi-Fi</div><button class="btn-scan" id="wf-scan" onclick="wfScan()">扫描网络</button></div>
<div class="cur" id="wf-cur"><div class="cur-name" id="wf-cur-name">未连接</div><div class="cur-sub" id="wf-cur-sub">连接网络后可联网校时与AI对话</div></div>
<div class="list" id="wf-list"><div class="empty">点击「扫描网络」查找可用Wi-Fi</div></div>
<div class="ap-row"><div><div class="ap-lbl">热点模式 (AP)</div><div class="ap-sub">开启后手机可直连本设备，无需路由器</div></div><div class="sw" id="ap-sw" onclick="apToggle()"></div></div></section>
</div><div class="toast" id="toast"></div>
<script>
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');clearTimeout(toast._h);toast._h=setTimeout(()=>t.classList.remove('show'),1800)}
function esc(s){return (s||'').replace(/[<>&"]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]))}
// ---- 蓝牙 ----
let btScanning=false;
function btScan(){const b=document.getElementById('bt-scan');b.disabled=true;b.classList.add('busy');b.textContent='扫描中...';btScanning=true;
 fetch('/bt_scan',{method:'POST'}).then(r=>r.json()).then(d=>{setTimeout(()=>{btRefresh();b.disabled=false;b.classList.remove('busy');b.textContent='扫描设备';btScanning=false},6000)}).catch(()=>{b.disabled=false;b.classList.remove('busy');b.textContent='扫描设备';btScanning=false;toast('扫描失败')})}
function btConnect(mac,name){toast('正在连接 '+name+'...');fetch('/bt_connect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mac})}).then(r=>r.json()).then(d=>{toast(d.ok?'已连接 '+name:('连接失败: '+(d.error||''))); btRefresh()}).catch(()=>toast('连接失败'))}
function btDisconnect(mac){toast('正在断开...');fetch('/bt_disconnect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mac})}).then(r=>r.json()).then(d=>{toast(d.ok?'已断开':'断开失败');btRefresh()}).catch(()=>toast('断开失败'))}
function btRefresh(){fetch('/bt_status').then(r=>r.json()).then(d=>{
 const dot=document.getElementById('bt-dot'),nm=document.getElementById('bt-cur-name'),sub=document.getElementById('bt-cur-sub');
 const conn=(d.devices||[]).find(x=>x.connected);
 if(conn){dot.classList.add('on');nm.textContent=conn.name;sub.textContent='已连接 · 语音将通过此设备播放';document.getElementById('bt-cur').classList.add('connected')}
 else{dot.classList.remove('on');nm.textContent='未连接';sub.textContent='连接蓝牙耳机后，语音播报将通过耳机输出';document.getElementById('bt-cur').classList.remove('connected')}
 const list=document.getElementById('bt-list');const devs=d.devices||[];
 if(!devs.length){list.innerHTML='<div class="empty">'+(btScanning?'<span class=spin></span>正在搜索...':'未发现设备，点击「扫描设备」')+'</div>';return}
 list.innerHTML=devs.map(x=>`<div class="dev${x.connected?' connected':''}"><div class="dev-l"><div class="dev-name">${esc(x.name)}</div><div class="dev-meta">${x.mac}${x.connected?' · 已连接':''}</div></div>${x.connected?`<button class="dev-act btn-danger" onclick="btDisconnect('${x.mac}')">断开</button>`:`<button class="dev-act btn-primary" onclick="btConnect('${x.mac}','${esc(x.name)}')">连接</button>`}</div>`).join('')
}).catch(()=>{})}
// ---- WiFi ----
let wfScanning=false,wfTarget=null;
function wfScan(){const b=document.getElementById('wf-scan');b.disabled=true;b.classList.add('busy');b.textContent='扫描中...';wfScanning=true;
 fetch('/wifi_scan',{method:'POST'}).then(r=>r.json()).then(d=>{setTimeout(()=>{wfRefresh();b.disabled=false;b.classList.remove('busy');b.textContent='扫描网络';wfScanning=false},4000)}).catch(()=>{b.disabled=false;b.classList.remove('busy');b.textContent='扫描网络';wfScanning=false;toast('扫描失败')})}
function wfPrompt(ssid,secure){if(!secure){wfConnect(ssid,'');return}wfTarget=ssid;wfRefresh(ssid)}
function wfConnect(ssid,pw){toast('正在连接 '+ssid+'...');fetch('/wifi_connect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ssid,password:pw})}).then(r=>r.json()).then(d=>{toast(d.ok?'已连接 '+ssid:('连接失败: '+(d.error||'')));wfTarget=null;wfRefresh()}).catch(()=>toast('连接失败'))}
function wfRefresh(expand){fetch('/wifi_status').then(r=>r.json()).then(d=>{
 const dot=document.getElementById('wf-dot'),nm=document.getElementById('wf-cur-name'),sub=document.getElementById('wf-cur-sub');
 if(d.connected){dot.classList.add('on');nm.textContent=d.ssid||'已连接';sub.textContent=(d.ip?('IP '+d.ip+' · '):'')+'已联网'}
 else{dot.classList.remove('on');nm.textContent='未连接';sub.textContent='连接网络后可联网校时与AI对话'}
 document.getElementById('ap-sw').classList.toggle('on',!!d.ap_active);
 const list=document.getElementById('wf-list');const nets=d.networks||[];
 if(d.unsupported){list.innerHTML='<div class="empty">此设备未检测到Wi-Fi管理工具<br>可使用下方热点模式</div>';return}
 if(!nets.length){list.innerHTML='<div class="empty">'+(wfScanning?'<span class=spin></span>正在搜索...':'点击「扫描网络」查找可用Wi-Fi')+'</div>';return}
 list.innerHTML=nets.map(x=>{const cur=d.connected&&x.ssid===d.ssid;const exp=(expand===x.ssid);return `<div class="dev${cur?' connected':''}"><div class="dev-l"><div class="dev-name">${esc(x.ssid)} ${x.secure?'🔒':''}</div><div class="dev-meta">信号 ${x.signal||'--'}%${cur?' · 已连接':''}</div>${exp?`<div class="pwd show"><input type="password" id="pw-in" placeholder="输入密码" onkeydown="if(event.key==='Enter')wfConnect('${esc(x.ssid)}',document.getElementById('pw-in').value)"><button class="dev-act btn-primary" onclick="wfConnect('${esc(x.ssid)}',document.getElementById('pw-in').value)">连接</button></div>`:''}</div>${cur?'':`<button class="dev-act btn-primary" onclick="wfPrompt('${esc(x.ssid)}',${x.secure?'true':'false'})">连接</button>`}</div>`}).join('')
}).catch(()=>{})}
function apToggle(){const sw=document.getElementById('ap-sw');const turnOn=!sw.classList.contains('on');toast(turnOn?'正在开启热点...':'正在关闭热点...');fetch('/ap_toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:turnOn})}).then(r=>r.json()).then(d=>{toast(d.ok?(turnOn?'热点已开启':'热点已关闭'):'操作失败');wfRefresh()}).catch(()=>toast('操作失败'))}
btRefresh();wfRefresh();setInterval(btRefresh,3000);setInterval(()=>{if(!wfTarget)wfRefresh()},5000);
</script></body></html>'''


@app.route('/')
def index():
    return render_template_string(INDEX_HTML, exercises=EXERCISES)


@app.route('/settings')
def settings():
    return render_template_string(SETTINGS_HTML)


# ===== 设置页后端接口（占位实现，等硬件确认后接入真实逻辑）=====
# 说明：以下接口目前返回安全的空数据/占位结果，保证 /settings 页面能正常渲染、
# 按钮点击不报错。真实的蓝牙(bluetoothctl/pactl)与 WiFi 逻辑将在 bt_audio.py /
# net_manager.py 中实现后接入。

@app.route('/bt_scan', methods=['POST'])
def bt_scan():
    if not _BT_AVAILABLE:
        return jsonify({'ok': False, 'error': '蓝牙不可用'})
    bt_audio.scan()
    return jsonify({'ok': True})


@app.route('/bt_status')
def bt_status():
    if not _BT_AVAILABLE:
        return jsonify({'devices': []})
    try:
        return jsonify({'devices': bt_audio.list_devices()})
    except Exception as e:
        print(f'[BT] status error: {e}')
        return jsonify({'devices': []})


@app.route('/bt_connect', methods=['POST'])
def bt_connect():
    data = request.get_json(force=True, silent=True) or {}
    mac = data.get('mac', '')
    if not _BT_AVAILABLE:
        return jsonify({'ok': False, 'error': '蓝牙不可用'})
    if not mac:
        return jsonify({'ok': False, 'error': '缺少 MAC'})
    try:
        res = bt_audio.connect(mac)
        if res.get('ok'):
            # 更新音频路由状态：播放走 pulse，录音探测 HFP 麦克风是否可用
            cap_dev = bt_audio.get_capture_device(mac)
            with _bt_lock:
                _bt_state['mac'] = mac
                _bt_state['playback'] = bt_audio.get_playback_device(mac)
                _bt_state['capture'] = cap_dev
                # 蓝牙 HFP 麦克风是单声道；回退板载时仍是双声道取右声道
                _bt_state['capture_ch'] = 1 if cap_dev == 'pulse' else RECORD_CH
            print(f'[BT] connected {mac}, playback={_bt_state["playback"]}, '
                  f'capture={_bt_state["capture"]}(ch={_bt_state["capture_ch"]})')
        return jsonify(res)
    except Exception as e:
        print(f'[BT] connect error: {e}')
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/bt_disconnect', methods=['POST'])
def bt_disconnect():
    data = request.get_json(force=True, silent=True) or {}
    mac = data.get('mac', '')
    if not _BT_AVAILABLE:
        return jsonify({'ok': False, 'error': '蓝牙不可用'})
    try:
        res = bt_audio.disconnect(mac)
        # 恢复板载音频
        with _bt_lock:
            _bt_state['mac'] = None
            _bt_state['playback'] = ALSA_DEVICE
            _bt_state['capture'] = ALSA_DEVICE
            _bt_state['capture_ch'] = RECORD_CH
        print('[BT] disconnected, audio restored to onboard')
        return jsonify(res)
    except Exception as e:
        print(f'[BT] disconnect error: {e}')
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/wifi_scan', methods=['POST'])
def wifi_scan():
    # TODO: 触发无线网卡扫描
    return jsonify({'ok': True})


@app.route('/wifi_status')
def wifi_status():
    try:
        # 检测 hostapd 进程（AP 是否激活）
        ap_active = False
        result = subprocess.run(
            ['pgrep', '-f', 'hostapd'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            ap_active = True

        # 检测 WiFi 连接状态
        connected = False
        ssid = ''
        ip = ''

        # 获取当前连接的 SSID
        result = subprocess.run(
            ['iwgetid', '-r'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            connected = True
            ssid = result.stdout.strip()

        # 获取 IP 地址
        result = subprocess.run(
            ['hostname', '-I'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            ips = result.stdout.strip().split()
            ip = ips[0] if ips else ''

        return jsonify({
            'connected': connected,
            'ssid': ssid,
            'ip': ip,
            'ap_active': ap_active,
            'unsupported': False,
            'networks': []
        })
    except Exception as e:
        print(f'[WiFi] 状态检测异常: {e}')
        return jsonify({
            'connected': False,
            'ssid': '',
            'ip': '',
            'ap_active': False,
            'unsupported': False,
            'networks': []
        })


@app.route('/wifi_connect', methods=['POST'])
def wifi_connect():
    data = request.get_json(force=True, silent=True) or {}
    ssid = data.get('ssid', '')
    # TODO: 连接指定 WiFi（注意：切换网络可能短暂断开当前热点连接）
    return jsonify({'ok': False, 'error': 'WiFi功能待接入', 'ssid': ssid})


@app.route('/ap_toggle', methods=['POST'])
def ap_toggle():
    data = request.get_json(force=True, silent=True) or {}
    turn_on = bool(data.get('on', False))

    try:
        if turn_on:
            # 启动热点
            result = subprocess.run(
                ['bash', 'ap-up.sh'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(__file__)
            )
            if result.returncode == 0:
                print('[AP] 热点已启动')
                return jsonify({'ok': True, 'on': True})
            else:
                print(f'[AP] 启动失败: {result.stderr}')
                return jsonify({'ok': False, 'error': result.stderr.strip(), 'on': False})
        else:
            # 关闭热点
            result = subprocess.run(
                ['bash', 'ap-down.sh'],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.path.dirname(__file__)
            )
            if result.returncode == 0:
                print('[AP] 热点已关闭')
                return jsonify({'ok': True, 'on': False})
            else:
                print(f'[AP] 关闭失败: {result.stderr}')
                return jsonify({'ok': False, 'error': result.stderr.strip(), 'on': False})
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'error': 'AP 操作超时', 'on': turn_on})
    except Exception as e:
        print(f'[AP] 异常: {e}')
        return jsonify({'ok': False, 'error': str(e), 'on': turn_on})


@app.route('/library')
def library():
    return render_template_string(LIBRARY_HTML)


@app.route('/profile')
def profile():
    try:
        with open(PROFILE_FILE, 'r') as f:
            prof = json.load(f)
    except Exception:
        prof = {'name': '', 'age': '', 'diagnosis': ''}
    try:
        with open(HISTORY_FILE, 'r') as f:
            hist = json.load(f)
    except Exception:
        hist = []
    return render_template_string(PROFILE_HTML, profile=prof, history=hist[-30:][::-1])


@app.route('/save_profile', methods=['POST'])
def save_profile():
    data = {
        'name':      request.form.get('name', '').strip(),
        'age':       request.form.get('age', '').strip(),
        'diagnosis': request.form.get('diagnosis', '').strip(),
    }
    try:
        with open(PROFILE_FILE, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/remote')
def remote():
    return render_template_string(REMOTE_HTML)


@app.route('/start', methods=['POST'])
def start_training():
    global _rehab, _training_paused
    exercise_key = request.form.get('exercise_key', 'left_elbow')
    try:
        target_reps = max(1, min(50, int(request.form.get('target_reps', 10))))
        target_sets = max(1, min(10, int(request.form.get('target_sets', 3))))
    except ValueError:
        return jsonify({'ok': False, 'error': 'invalid params'}), 400
    if exercise_key not in EXERCISES:
        return jsonify({'ok': False, 'error': 'unknown exercise'}), 400
    with _lock:
        _rehab = RehabEngine(exercise_key=exercise_key,
                             target_reps=target_reps, target_sets=target_sets)
    _training_paused = False  # 重置暂停状态
    print(f'[训练] 开始: {EXERCISES[exercise_key][0]} {target_reps}次x{target_sets}组')
    return jsonify({'ok': True})


@app.route('/status')
def status():
    with _lock:
        s = dict(_latest_status)
        if _rehab is not None:
            s['min_safe']    = _rehab.min_safe
            s['max_safe']    = _rehab.max_safe
            s['target_reps'] = _rehab.target_reps
            s['target_sets'] = _rehab.target_sets
            s['started']     = True
            # === 浣跨敤鐪熷疄璁℃暟 ===
            s['reps']        = _rehab.counter.reps
            s['sets']        = _rehab.sets_done
            s['bad_count']   = _rehab.bad_count
            s['total_bad_count'] = getattr(_rehab, 'total_bad_count', _rehab.bad_count)
            s['bad_form']    = _rehab.bad_form_this_rep
            s['completed']   = getattr(_rehab, 'completed', False)
            s['paused']      = _training_paused
        else:
            s['started'] = False
            s['paused']  = False
        s['depth_calib'] = {
            'offset_x': _depth_kp_offset_x,
            'offset_y': _depth_kp_offset_y,
            'scale_x': _depth_kp_scale_x,
            'scale_y': _depth_kp_scale_y,
            'mirror_x': _depth_kp_mirror_x,
        }
        # 添加温湿度数据
        s['temperature'] = _latest_temp
        s['humidity'] = _latest_humidity
        s['temp_warning'] = (_latest_temp is not None and
                            _latest_temp > _temp_warning_threshold)
    return jsonify(s)


@app.route('/pause_toggle', methods=['POST'])
def pause_toggle():
    global _training_paused
    _training_paused = not _training_paused
    state = '已暂停' if _training_paused else '已继续'
    _log_train(f'训练{state}')
    return jsonify({'ok': True, 'paused': _training_paused})


@app.route('/env_status')
def env_status():
    """环境监控状态接口"""
    with _lock:
        temp = _latest_temp
        hum = _latest_humidity

    return jsonify({
        'temperature': temp,
        'humidity': hum,
        'temp_warning': temp is not None and temp > _temp_warning_threshold,
        'temp_threshold': _temp_warning_threshold,
        'sht40_available': _SHT40_AVAILABLE
    })


@app.route('/health')
def health():
    """健康检查端点（用于快速测试连通性）"""
    return "OK", 200, {'Content-Type': 'text/plain'}


@app.route('/stop', methods=['POST'])
def stop_training():
    global _rehab, _training_paused
    with _lock:
        had = _rehab is not None
        _rehab = None
        _training_paused = False
        _latest_status.clear()
    if had:
        _log_train('训练已结束')
    return jsonify({'ok': True, 'stopped': had})


# === 移除 /manual_good 和 /manual_bad 路由 ===
# 纯自动计数不需要手动按钮

@app.route('/set_depth_angle', methods=['POST'])
def set_depth_angle():
    global _depth_azimuth, _depth_elevation
    try:
        az = float(request.form.get('az', _depth_azimuth))
        el = float(request.form.get('el', _depth_elevation))
        _depth_azimuth   = max(-90.0, min(90.0, az))
        _depth_elevation = max(0.0,   min(89.0, el))
        return jsonify({'ok': True, 'az': _depth_azimuth, 'el': _depth_elevation})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/set_depth_calib', methods=['POST'])
def set_depth_calib():
    global _depth_kp_offset_x, _depth_kp_offset_y, _depth_kp_scale_x, _depth_kp_scale_y, _depth_kp_mirror_x
    try:
        if 'offset_x' in request.form:
            _depth_kp_offset_x = max(-320.0, min(320.0, float(request.form['offset_x'])))
        if 'offset_y' in request.form:
            _depth_kp_offset_y = max(-240.0, min(240.0, float(request.form['offset_y'])))
        if 'scale_x' in request.form:
            _depth_kp_scale_x = max(0.5, min(1.5, float(request.form['scale_x'])))
        if 'scale_y' in request.form:
            _depth_kp_scale_y = max(0.5, min(1.5, float(request.form['scale_y'])))
        if 'mirror_x' in request.form:
            _depth_kp_mirror_x = request.form['mirror_x'].lower() in ('1', 'true', 'yes', 'on')
        return jsonify({
            'ok': True,
            'offset_x': _depth_kp_offset_x,
            'offset_y': _depth_kp_offset_y,
            'scale_x': _depth_kp_scale_x,
            'scale_y': _depth_kp_scale_y,
            'mirror_x': _depth_kp_mirror_x,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/voice_status')
def voice_status():
    with _voice_lock:
        active = _voice_active
        log = list(_voice_log[-6:])
    return jsonify({'active': active, 'log': log})


@app.route('/train_log')
def train_log():
    with _voice_lock:
        log = list(_train_log[-10:])
    return jsonify({'log': log})


@app.route('/speak_text', methods=['POST'])
def speak_text():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'ok': False, 'error': '空文本'}), 400
    threading.Thread(target=speak, args=(text,), daemon=True).start()
    return jsonify({'ok': True})


@app.route('/toggle_pc_skel_red', methods=['POST'])
def toggle_pc_skel_red():
    global _pc_skel_red
    _pc_skel_red = not _pc_skel_red
    return jsonify({'ok': True, 'red': _pc_skel_red})


def _gen_frames():
    while True:
        with _lock:
            frame = _latest_frame_jpeg
        if frame:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        time.sleep(0.033)


def _gen_depth():
    while True:
        with _lock:
            frame = _latest_depth_jpeg
        if frame:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/depth_feed')
def depth_feed():
    return Response(_gen_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 阿里云 Token

def _save_history(exercise_name, sets, total_reps, bad):
    record = {
        'time':       datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'exercise':   exercise_name,
        'sets':       sets,
        'total_reps': total_reps,
        'bad':        bad,
    }
    try:
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
        except Exception:
            data = []
        data.append(record)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f'[History] saved: {record}')
    except Exception as e:
        print(f'[History] error: {e}')

# Chat

def _sync_time():
    for server in ['ntp.aliyun.com', 'pool.ntp.org']:
        try:
            ret = subprocess.run(['ntpdate', '-u', server],
                                 capture_output=True, timeout=10)
            if ret.returncode == 0:
                print(f'[Time] NTP sync OK via {server}')
                return
        except Exception:
            continue
    print('[Time] NTP sync failed, continuing anyway')

# TTS / ASR

def _get_aliyun_token():
    if _token_cache['token'] and time.time() < _token_cache['expire'] - 300:
        return _token_cache['token']
    if not ALIYUN_ACCESS_KEY or not ALIYUN_ACCESS_SECRET:
        print('[Token] ERROR: Missing Aliyun credentials')
        return None
    from aliyunsdkcore.client import AcsClient
    from aliyunsdkcore.request import CommonRequest
    client = AcsClient(ALIYUN_ACCESS_KEY, ALIYUN_ACCESS_SECRET, 'cn-shanghai')
    req = CommonRequest()
    req.set_method('POST')
    req.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    req.set_version('2019-02-28')
    req.set_action_name('CreateToken')
    resp = json.loads(client.do_action_with_exception(req))
    token = resp['Token']['Id']
    _token_cache['token'] = token
    _token_cache['expire'] = time.time() + 24 * 3600
    print('[Token] OK')
    return token

# 语音对话主循环线程

def _chat(user_text):
    global _chat_history
    _chat_history.append({'role': 'user', 'content': user_text})
    if len(_chat_history) > MAX_HISTORY * 2:
        _chat_history = _chat_history[-(MAX_HISTORY * 2):]
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + _chat_history
    if not ARK_API_KEY:
        return '抱歉，API密钥未配置。'
    try:
        resp = requests.post(
            'https://ark.cn-beijing.volces.com/api/v3/responses',
            headers={'Authorization': f'Bearer {ARK_API_KEY}',
                     'Content-Type': 'application/json'},
            json={'model': CHAT_MODEL, 'input': messages},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Responses API：在 output 中查找 type=message 的项，取 content[0].text。
        reply = ''
        for item in data.get('output', []):
            if item.get('type') == 'message':
                for c in item.get('content', []):
                    if c.get('type') == 'output_text' and c.get('text'):
                        reply = c['text']
                        break
                if reply:
                    break
        if not reply:
            # 兜底：尝试 chat/completions 格式。
            reply = data.get('choices', [{}])[0].get('message', {}).get('content', '')
    except Exception as e:
        print(f'[Chat] error: {e}')
        reply = '抱歉，我现在无法回答。'
    if not reply:
        reply = '抱歉，我没有找到相关信息。'
    _chat_history.append({'role': 'assistant', 'content': reply})
    return reply

# 主入口

def _stereo_to_mono(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    if ch == 1:
        return frames
    samples = struct.unpack(f'<{len(frames)//2}h', frames)
    # 右声道才有麦克风信号。
    mono = samples[1::ch]
    return struct.pack(f'<{len(mono)}h', *mono)


def _tts_stream_play(text):
    """流式 TTS：音频块边合成边喂给 aplay，第一个字最快出声。"""
    import nls
    token = _get_aliyun_token()
    if not token:
        return False

    # aplay 直接读 stdin 原始 PCM（16k 单声道 16bit），无需等整段合成
    # 播放设备根据蓝牙连接状态动态选择（有蓝牙走 pulse，否则板载）
    with _bt_lock:
        play_dev = _bt_state['playback']
    proc = subprocess.Popen(
        ['aplay', '-D', play_dev, '-t', 'raw',
         '-f', 'S16_LE', '-r', '16000', '-c', '1'],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    done = threading.Event()
    got_data = {'any': False}

    def on_data(data, *a):
        got_data['any'] = True
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass  # aplay 已退出，忽略

    def on_completed(msg, *a): done.set()
    def on_error(msg, *a):
        print(f'[TTS] error: {msg[:100]}')
        done.set()

    tts = nls.NlsSpeechSynthesizer(
        token=token, appkey=ALIYUN_APPKEY,
        on_data=on_data, on_completed=on_completed, on_error=on_error,
    )
    tts.start(text=text, voice='aixia', aformat='pcm', sample_rate=16000)
    done.wait(timeout=15)

    try:
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return got_data['any']


def speak(text):
    """TTS 流式播放（线程安全），使用 Event 替代忙等。"""
    print(f'[speak] {text}')
    if not _speak_lock.acquire(blocking=False):
        print('[speak] busy, skipping')
        return
    _speak_done.clear()
    _speaking_event.set()
    try:
        _tts_stream_play(text)
    except Exception as e:
        print(f'[speak] error: {e}')
    finally:
        _speaking_event.clear()
        _speak_done.set()  # 通知 VAD 可以继续。
        _speak_lock.release()


def _record_vad(wav_path):
    while _speaking_event.is_set():
        _speak_done.wait(timeout=0.5)
    # 录音设备/声道根据蓝牙状态动态选择：
    #   板载：双声道，取右声道（麦克风在右声道）
    #   蓝牙HFP：单声道，全部采样都是麦克风
    with _bt_lock:
        cap_dev = _bt_state['capture']
        cap_ch = _bt_state['capture_ch']
    bytes_per_frame = 2 * cap_ch
    chunk_bytes = VAD_CHUNK_FRAMES * bytes_per_frame
    max_chunks = int(VAD_MAX_SECS * RECORD_RATE / VAD_CHUNK_FRAMES)
    silence_limit = int(VAD_SILENCE_SECS * RECORD_RATE / VAD_CHUNK_FRAMES)

    proc = subprocess.Popen(
        ['arecord', '-D', cap_dev, '-f', 'S16_LE',
         '-r', str(RECORD_RATE), '-c', str(cap_ch)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    print(f'[VAD] Waiting for speech... (dev={cap_dev}, ch={cap_ch})')
    speech_frames = []
    recording = False
    silence_chunks = 0
    total_speech_chunks = 0
    try:
        for _ in range(max_chunks + 100):
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            samples = np.frombuffer(chunk, dtype=np.int16)
            # 板载双声道取右声道；蓝牙单声道取全部
            mic = samples[1::cap_ch] if cap_ch > 1 else samples
            if not len(mic):
                continue
            rms = float(np.sqrt(np.mean(mic.astype(np.float32) ** 2)))
            if not recording:
                if rms > VAD_SPEECH_THRESH:
                    recording = True
                    silence_chunks = 0
                    speech_frames.append(chunk)
                    print(f'[VAD] Speech detected (rms={rms:.0f})')
            else:
                speech_frames.append(chunk)
                if rms < VAD_SILENCE_THRESH:
                    silence_chunks += 1
                    if silence_chunks >= silence_limit:
                        print(f'[VAD] Silence, stopping (rms={rms:.0f}, sil={silence_chunks}/{silence_limit})')
                        break
                else:
                    silence_chunks = 0
                    total_speech_chunks += 1
                    if total_speech_chunks >= max_chunks:
                        print(f'[VAD] Max length reached (rms={rms:.0f})')
                        break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if not speech_frames:
        return False
    speech_secs = len(speech_frames) * VAD_CHUNK_FRAMES / RECORD_RATE
    if speech_secs < VAD_MIN_SECS:
        return False
    raw = b''.join(speech_frames)
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(cap_ch); wf.setsampwidth(2)
        wf.setframerate(RECORD_RATE); wf.writeframes(raw)
    print(f'[VAD] Recorded {speech_secs:.1f}s')
    return True


def _asr_wav(wav_path):
    import nls
    token = _get_aliyun_token()
    if not token:
        return ''
    final_text = []
    done = threading.Event()
    started = threading.Event()

    def on_start(msg, *a): started.set()
    def on_result_changed(msg, *a):
        try:
            t = json.loads(msg).get('payload', {}).get('result', '')
            if t: print(f'[ASR] partial: {t}')
        except Exception: pass
    def on_completed(msg, *a):
        try:
            t = json.loads(msg).get('payload', {}).get('result', '')
            if t: final_text.append(t)
        except Exception: pass
        done.set()
    def on_error(msg, *a):
        print(f'[ASR] error: {msg[:150]}')
        started.set(); done.set()

    audio_data = _stereo_to_mono(wav_path)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    audio_data = np.clip(audio_np * 6.0, -32768, 32767).astype(np.int16).tobytes()
    sr = nls.NlsSpeechRecognizer(
        token=token, appkey=ALIYUN_APPKEY,
        on_start=on_start, on_result_changed=on_result_changed,
        on_completed=on_completed, on_error=on_error,
    )
    sr.start(aformat='pcm', sample_rate=RECORD_RATE, enable_punctuation_prediction=True)
    started.wait(timeout=10)
    chunk = 3200
    for i in range(0, len(audio_data), chunk):
        sr.send_audio(audio_data[i:i+chunk])
        time.sleep(0.05)
    sr.stop()
    done.wait(timeout=10)
    return ''.join(final_text).strip()

# 语音对话主循环线程

WAKE_WORDS   = ['你好小明', '小明', '你好，小明']
WAKE_TIMEOUT = 30  # 唤醒后 30 秒无输入自动回到待机。

def _log_voice(role, text):
    with _voice_lock:
        _voice_log.append({'role': role, 'text': text})

def _log_train(text):
    with _voice_lock:
        _train_log.append({'text': text})

def _parse_training_intent(text):
    text = text.lower()

    exercise_map = {
        '左肘': 'left_elbow', '左手肘': 'left_elbow', '左胳膊': 'left_elbow', '左臂': 'left_elbow',
        '右肘': 'right_elbow', '右手肘': 'right_elbow', '右胳膊': 'right_elbow', '右臂': 'right_elbow',
        '左膝': 'left_knee', '左膝盖': 'left_knee', '左腿': 'left_knee',
        '右膝': 'right_knee', '右膝盖': 'right_knee', '右腿': 'right_knee',
    }

    if not re.search(r'(训练|练|开始|做)', text):
        return None

    exercise_key = None
    for spoken, key in exercise_map.items():
        if spoken in text:
            exercise_key = key
            break

    if not exercise_key:
        return None

    reps = 10
    m = re.search(r'(\d+)\s*次', text)
    if m:
        reps = int(m.group(1))

    sets = 3
    m = re.search(r'(\d+)\s*组', text)
    if m:
        sets = int(m.group(1))

    return {'action': 'start_training', 'exercise': exercise_key, 'reps': reps, 'sets': sets}

def _parse_control_intent(text):
    text = text.lower()

    if '暂停' in text or '停一下' in text:
        return {'action': 'pause'}
    if '继续' in text or ('开始' in text and '训练' not in text):
        return {'action': 'resume'}
    if '结束训练' in text or '停止训练' in text or '不练了' in text:
        return {'action': 'stop'}
    if re.search(r'(练了|做了|完成了).*(多少|几)', text) or '进度' in text:
        return {'action': 'query'}

    return None

def _do_pause_toggle():
    global _training_paused
    _training_paused = not _training_paused
    state = '已暂停' if _training_paused else '已继续'
    _log_train(f'训练{state}')
    return _training_paused

def _do_stop_training():
    global _rehab, _training_paused
    with _lock:
        had = _rehab is not None
        _rehab = None
        _training_paused = False
        _latest_status.clear()
    if had:
        _log_train('训练已结束')
    return had

def _do_query_status():
    with _lock:
        if _rehab is not None:
            return {'started': True, 'sets': _rehab.sets_done, 'reps': _rehab.counter.reps}
    return {'started': False, 'sets': 0, 'reps': 0}

def _do_start_training(exercise_key, target_reps, target_sets):
    global _rehab, _training_paused
    with _lock:
        _rehab = RehabEngine(exercise_key=exercise_key,
                             target_reps=target_reps, target_sets=target_sets)
    _training_paused = False
    print(f'[训练] 开始: {EXERCISES[exercise_key][0]} {target_reps}次x{target_sets}组')


def _voice_loop():
    global _voice_active
    print('[Voice] started')
    while True:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
        try:
            if not _record_vad(wav_path):
                continue
            text = _asr_wav(wav_path)
            if not text:
                print('[ASR] no result')
                continue
            print(f'[ASR] {text}')

            if not any(w in text for w in WAKE_WORDS):
                continue

            print('[Voice] 唤醒，进入对话模式')
            with _voice_lock:
                _voice_active = True
            _log_voice('sys', '已唤醒')
            speak('我在，请说')
            last_active = time.time()

            while time.time() - last_active < WAKE_TIMEOUT:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f2:
                    wav2 = f2.name
                try:
                    if not _record_vad(wav2):
                        continue
                    user_text = _asr_wav(wav2)
                    if not user_text:
                        print('[ASR] no result')
                        continue
                    print(f'[User] {user_text}')
                    _log_voice('user', user_text)

                    # 退出对话
                    if any(w in user_text for w in ['再见', '退出', '结束对话']):
                        _log_voice('sys', '对话结束')
                        speak('好的，再见')
                        break

                    # 优先级1: 控制意图（暂停/继续/结束训练/查询）
                    control_intent = _parse_control_intent(user_text)
                    if control_intent:
                        action = control_intent['action']
                        try:
                            if action == 'pause':
                                paused = _do_pause_toggle()
                                speak('已暂停' if paused else '继续训练')
                                _log_voice('bot', '已暂停训练' if paused else '继续训练')
                            elif action == 'resume':
                                paused = _do_pause_toggle()
                                speak('继续训练' if not paused else '已暂停')
                                _log_voice('bot', '继续训练' if not paused else '已暂停')
                            elif action == 'stop':
                                _do_stop_training()
                                speak('训练已结束')
                                _log_voice('bot', '训练已结束')
                            elif action == 'query':
                                data = _do_query_status()
                                msg = f"已完成{data.get('sets', 0)}组，本组{data.get('reps', 0)}次"
                                speak(msg)
                                _log_voice('bot', msg)
                        except Exception as e:
                            print(f'[Voice] 控制失败: {e}')
                            speak('操作失败')
                        last_active = time.time()
                        continue

                    # 优先级2: 训练意图（正则识别）
                    training_intent = _parse_training_intent(user_text)
                    if training_intent:
                        ex_key = training_intent['exercise']
                        reps = training_intent['reps']
                        sets = training_intent['sets']
                        ex_name = EXERCISES.get(ex_key, ('未知', ''))[0]
                        try:
                            _do_start_training(ex_key, reps, sets)
                            msg = f'好的，开始{ex_name}训练，每组{reps}次，共{sets}组'
                            speak(msg)
                            _log_voice('bot', msg)
                        except Exception as e:
                            print(f'[Voice] 启动训练失败: {e}')
                            speak('启动失败')
                        last_active = time.time()
                        continue

                    # 优先级3: Prompt工程（模型返回JSON）
                    reply = _chat(user_text)
                    if reply.strip().startswith('{'):
                        try:
                            cmd = json.loads(reply)
                            if cmd.get('action') == 'start_training':
                                ex_key = cmd.get('exercise')
                                reps = cmd.get('reps', 10)
                                sets = cmd.get('sets', 3)
                                confirm = cmd.get('confirm', '开始训练')
                                _do_start_training(ex_key, reps, sets)
                                speak(confirm)
                                _log_voice('bot', confirm)
                                last_active = time.time()
                                continue
                        except (json.JSONDecodeError, Exception):
                            pass

                    # 优先级4: 普通对话
                    _log_voice('bot', reply)
                    speak(reply)
                    last_active = time.time()
                except Exception as e:
                    print(f'[Voice] error: {e}')
                finally:
                    if os.path.exists(wav2):
                        os.unlink(wav2)
            with _voice_lock:
                _voice_active = False
            print('[Voice] 对话结束，回到待机')
        except Exception as e:
            print(f'[Voice] error: {e}')
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)


# 主入口

def _self_check():
    """启动自检：检查外部命令、Python 模块、密钥配置。
    缺失项清晰打印，但不阻断启动（能降级的降级）。"""
    import importlib.util
    import shutil

    print('=' * 56)
    print('[自检] 系统依赖检查')
    print('-' * 56)

    # 1. 外部命令
    cmds = {
        'aplay':        ('语音播放',   '缺失则无法播放 TTS 语音'),
        'arecord':      ('语音录音',   '缺失则无法录音，语音助手不可用'),
        'ntpdate':      ('网络校时',   '缺失则时间可能不准，不影响主功能'),
        'bluetoothctl': ('蓝牙管理',   '缺失则蓝牙耳机功能不可用'),
        'pactl':        ('音频路由',   '缺失则蓝牙音频路由不可用'),
    }
    for cmd, (feat, warn) in cmds.items():
        if shutil.which(cmd):
            print(f'  [OK]   {cmd:<14} ({feat})')
        else:
            print(f'  [缺失] {cmd:<14} ({feat}) -> {warn}')

    # 2. Python 可选模块
    mods = {
        'nls':          '阿里云语音 SDK，缺失则 TTS/ASR 不可用',
        'aliyunsdkcore': '阿里云 SDK 核心，缺失则无法获取语音 token',
        'rknnlite':     'RKNN NPU 加速，缺失则自动退回 ONNX CPU',
    }
    for mod, warn in mods.items():
        if importlib.util.find_spec(mod) is not None:
            print(f'  [OK]   模块 {mod:<14}')
        else:
            print(f'  [缺失] 模块 {mod:<14} -> {warn}')

    # 3. 密钥配置
    keys = {
        'ARK_API_KEY':          ARK_API_KEY,
        'ALIYUN_APPKEY':        ALIYUN_APPKEY,
        'ALIYUN_ACCESS_KEY':    ALIYUN_ACCESS_KEY,
        'ALIYUN_ACCESS_SECRET': ALIYUN_ACCESS_SECRET,
    }
    missing_keys = [k for k, v in keys.items() if not v]
    if missing_keys:
        print(f'  [警告] 未配置密钥: {", ".join(missing_keys)}')
        print(f'         语音功能将不可用。请在 secrets.env 中填写。')
    else:
        print('  [OK]   密钥已配置')

    # 4. 模型文件
    for label, path in (('RKNN', RKNN_MODEL), ('ONNX', ONNX_MODEL)):
        if os.path.exists(path):
            print(f'  [OK]   {label} 模型: {os.path.basename(path)}')
        else:
            print(f'  [缺失] {label} 模型: {path}')

    print('=' * 56)


def _print_network_info():
    """打印网络访问信息"""
    import socket
    hostname = socket.gethostname()

    print("\n" + "=" * 60)
    print("Web UI 访问地址:")
    print("-" * 60)

    # 获取所有 IP 地址
    try:
        addrs = socket.getaddrinfo(hostname, None)
        ips = set()
        for addr in addrs:
            ip = addr[4][0]
            if not ip.startswith('127.') and ':' not in ip:  # 排除 localhost 和 IPv6
                ips.add(ip)

        for ip in sorted(ips):
            print(f"  http://{ip}:{FLASK_PORT}/")

        if not ips:
            print(f"  http://127.0.0.1:{FLASK_PORT}/ (仅本地)")
    except:
        print(f"  http://127.0.0.1:{FLASK_PORT}/ (仅本地)")

    print("-" * 60)
    print("如无法访问，请检查:")
    print("  1. 设备与开发板是否在同一网络")
    print("  2. 防火墙是否开放 5001 端口")
    print("  3. 在开发板上测试: curl http://127.0.0.1:5001/health")
    print("=" * 60 + "\n")



def main():
    _self_check()
    _sync_time()

    # 注意：WiFi 连接和网络功能已移至 QT 界面管理
    # 用户可通过 QT 界面的 "WiFi 设置" 按钮连接网络

    _init_inference()
    _camera_init()

    threading.Thread(target=_processing_loop, daemon=True).start()
    threading.Thread(target=_voice_loop,      daemon=True).start()

    _print_network_info()
    print(f'[Flask] 推理模式: {_infer_mode}')
    print('[INFO] true_total.py - 纯自动计数模式')
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print('[Main] exit')
        _cam_stop.set()
        sys.exit(0)


if __name__ == '__main__':
    main()


