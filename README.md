# FocusVision - 课堂专注度检测系统

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-orange)

FocusVision 是一个基于计算机视觉的课堂专注度检测系统，能够通过分析视频中学生的面部表情和头部姿态，实时评估和记录学生的专注状态，为教学评估和课堂管理提供数据支持。系统利用YOLO姿态检测和DeepFace表情分析，综合计算专注度评分，并通过Web界面展示实时分析结果与数据可视化。

## 🌟 主要功能

- **多种视频输入支持**：
  - 本地视频文件上传与处理
  - 摄像头实时处理
  - RTSP流处理
  - IP摄像头连接

- **专注度智能评估**：
  - 基于头部姿态分析（抬头/低头检测）
  - 面部表情情绪识别
  - 综合多维度因素的专注度评分

- **可视化分析报告**：
  - 实时处理进度展示
  - 专注度随时间变化曲线图
  - 情绪分布统计
  - 关键指标数据导出

- **便捷的Web界面**：
  - 直观的操作流程
  - 实时预览与进度反馈
  - 响应式设计，支持多种设备

## 📋 技术栈

- **后端框架**：Flask 3.1+
- **视觉处理**：
  - YOLO v11（人体姿态检测）
  - DeepFace 0.0.93（情绪分析）
  - OpenCV 4.12+（图像处理）
- **机器学习框架**：
  - PyTorch 2.5+
  - TensorFlow 2.19+
- **数据分析**：Pandas 2.3+, NumPy 2.0+
- **可视化**：Matplotlib 3.9+

## 💻 系统要求

- Python 3.9+
- CUDA 11.3+ 和 cuDNN 8.2+ (用于GPU加速)
- 足够的GPU资源（推荐NVIDIA显卡，8GB+ VRAM）
- 适合视频处理的系统内存（建议16GB+）
- Windows/Linux/MacOS系统

## 🚀 快速开始

### 安装

1. 克隆仓库：

```bash
git clone https://github.com/zpyc1oud/FocusVision.git
cd FocusVision
```

2. 创建并激活Conda环境：

```bash
conda env create -f environment.yml
conda activate FocusVision
```

3. 确保CUDA环境正确配置（GPU加速）：

```bash
python -c "import torch; print('GPU可用:', torch.cuda.is_available())"
```

### 运行应用
```bash
python app.py
```

启动后，在浏览器中访问 `http://127.0.0.1:5000` 即可使用系统。

## 📝 使用指南

### 文件上传模式

1. 在主页选择"文件上传"标签
2. 点击"选择视频文件"按钮上传课堂视频
3. 点击"开始分析"按钮开始处理
4. 在处理页面查看实时进度和预览
5. 处理完成后查看分析结果和报告

### 视频流模式

1. 在主页选择"视频流模式"标签
2. 选择流类型（本地摄像头、RTSP流或IP摄像头）
3. 输入必要的连接信息
4. 点击"开始分析"按钮
5. 实时查看分析结果和专注度评分

## 📊 结果解读

系统生成的专注度评分介于0-1之间：
- **0.8-1.0**: 非常专注
- **0.6-0.8**: 较为专注
- **0.4-0.6**: 一般专注
- **0.2-0.4**: 不太专注
- **0.0-0.2**: 非常不专注

系统还会生成情绪分布图表，包括：中性、快乐、悲伤、生气、害怕、厌恶、惊讶等类别。

## 🔧 高级配置

专注度检测器参数可在 `focus_vision/focus_detector.py` 中调整：
```python
# 创建检测器实例时可自定义参数
focus_detector = FocusDetector(
    model_path="yolo11n-pose.pt",  # 姿态检测模型路径
    focus_threshold=30,            # 头部角度阈值(度)
    emotion_weight=0.3,            # 表情权重比例
    use_gpu=True                   # 是否使用GPU加速
)
```

系统其他配置可在 `app.py` 中调整：
- 视频处理参数（帧率、输出分辨率等）
- 服务器配置（端口、上传限制等）
- 输出文件格式与存储路径