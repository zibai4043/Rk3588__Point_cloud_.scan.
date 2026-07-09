# 康复训练系统

基于深度摄像头和姿态识别的智能康复训练辅助系统。

## 演示视频

[点击查看演示视频](assets/videos/demo.mp4)

## 项目文档

[项目简介 PDF](docs/project-introduction.pdf)

## 三维模型

- [new_box.3dm](assets/models/new_box.3dm)
- [顶盖.3dm](assets/models/顶盖.3dm)

## 功能特性

- ✅ 实时姿态识别和动作计数
- ✅ 多种康复训练动作支持
- ✅ 深度图像分析和安全提示
- ✅ 温湿度环境监测
- ✅ 语音交互功能
- ✅ Web 界面实时展示

## 系统要求

### 硬件
- 瑞芯微 RK3588 开发板或类似 ARM 平台
- 深度摄像头（OpenNI2 兼容）
- RGB 摄像头
- SHT40 温湿度传感器（可选）
- 音频设备（语音功能）

### 软件
- Python 3.12+
- OpenNI2
- ONNX Runtime 或 RKNN Lite
- Flask
- OpenCV
- NumPy

## 快速开始

### 1. 安装依赖

```bash
pip3 install flask opencv-python numpy onnxruntime
```

### 2. 配置环境

复制配置文件模板：
```bash
cp secrets.env.example secrets.env
```

编辑 `secrets.env` 填入你的 API 密钥（可选，仅语音功能需要）。

### 3. 准备模型文件

确保以下模型文件存在：
- `models/pose_model.onnx` - 姿态识别模型（ONNX 格式）
- `models/pose_model.rknn` - 姿态识别模型（RKNN 格式，可选）

### 4. 运行程序

```bash
python3 rehab_system.py
```

访问 `http://localhost:5001` 查看界面。

## 配置说明

### 环境变量（secrets.env）

```bash
# 阿里云语音服务（可选）
ALIYUN_APPKEY=your_appkey
ALIYUN_ACCESS_KEY_ID=your_access_key_id
ALIYUN_ACCESS_KEY_SECRET=your_access_key_secret

# 其他配置
TEMP_WARNING_THRESHOLD=35.0
```

### 网络配置

如需开机自动连接 WiFi，参考 `docs/wifi_setup.md`。

## 支持的训练动作

- 左臂肘关节屈伸
- 右臂肘关节屈伸
- 深蹲
- 更多动作可自定义添加

## 项目结构

```
.
├── rehab_system.py          # 主程序
├── models/                  # 模型文件目录
│   ├── pose_model.onnx
│   └── pose_model.rknn
├── secrets.env.example      # 配置文件模板
├── requirements.txt         # Python 依赖
└── README.md
```

## 开发说明

系统采用 Flask 提供 Web 界面，后台线程处理摄像头采集和姿态识别。

核心模块：
- 姿态识别：支持 ONNX 和 RKNN 两种推理后端
- 动作计数：基于关键点角度和位置变化
- 深度分析：检测动作规范性
- 语音交互：阿里云 NLS 服务

## 许可证

MIT License

## 致谢

- ONNX Runtime
- OpenNI2
- 瑞芯微 RKNN
