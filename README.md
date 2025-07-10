# FocusVision - 课堂专注度检测系统

基于YOLOv11姿态检测和DeepFace表情分析的课堂专注度监测应用，通过分析学生头部姿态和表情来全面评估专注度。

## 项目概述

FocusVision是一款为智慧教育领域设计的课堂专注度检测系统，它通过计算机视觉和人工智能技术，实时监测和分析学生的专注状态。系统能够：

- 识别学生的抬头/低头姿态并计算头部角度
- 检测学生的面部表情和情绪状态
- 结合姿态和表情分析计算专注度评分
- 生成专注度数据统计和可视化报告

通过这些分析，教师可以获得课堂整体专注度的实时反馈，从而调整教学策略和节奏，提高教学效果。

## 功能特点

- **多维度专注度评估**：结合头部姿态和表情分析全面评估专注度
- **实时可视化**：展示学生骨架、表情状态和专注度指标
- **数据记录**：记录每一帧所有学生的专注度数据并保存为CSV文件
- **结果输出**：保存可视化后的视频结果，方便事后分析
- **统计报告**：生成专注度和情绪变化统计图表
- **多种输入支持**：支持本地视频文件、摄像头、RTSP流和IP摄像头
- **友好的Web界面**：直观的用户界面，方便操作和查看结果
- **GPU加速支持**：检测到GPU时自动使用加速

## 技术架构

FocusVision采用了以下技术栈：

- **前端**：HTML/CSS/JavaScript，基于Flask提供的模板系统
- **后端**：Python Flask Web框架
- **计算机视觉**：
  - 使用Ultralytics YOLOv11进行人体姿态检测
  - 采用DeepFace进行面部表情分析
- **数据处理与可视化**：
  - Pandas用于数据处理和分析
  - Matplotlib用于生成图表和统计报告
- **并发处理**：使用多线程处理视频流，保持界面响应性

## 安装要求

- Python 3.8+
- OpenCV
- NumPy
- Pandas
- Ultralytics (YOLOv11)
- DeepFace
- Flask (Web界面)
- Matplotlib

## 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/zpyc1oud/FocusVision.git
cd FocusVision
```

2. 创建并激活Conda环境（推荐）：

```bash
conda env create -f environment.yml
conda activate focus-vision
```

或者使用pip安装依赖：

```bash
pip install -r requirements.txt
```

3. 下载YOLOv11姿态检测模型（如果尚未下载）：

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

## 使用方法

### Web界面使用

启动Web服务：

```bash
python app.py
```

然后在浏览器中访问 `http://localhost:5000` 使用图形界面上传视频或连接摄像头。

Web界面提供两种操作模式：
- **文件上传模式**：上传本地视频文件进行处理
- **视频流模式**：连接实时视频源（摄像头、RTSP流或IP摄像头）

### 命令行使用

```bash
python main.py --video 视频路径 [选项]
```

#### 参数说明：

- `--video`, `-v`: 输入视频文件路径（必需）
- `--output`, `-o`: 输出视频文件路径（默认：output/result.mp4）
- `--csv`, `-c`: 输出CSV数据文件路径（默认：output/focus_data.csv）
- `--threshold`, `-t`: 头部角度阈值，超过此角度视为不专注（默认：30度）
- `--model`, `-m`: YOLO模型路径（默认：yolo11n-pose.pt）

### 示例

```bash
python main.py --video assets/test.mp4 --output output/result.mp4 --csv output/focus_data.csv --threshold 25
```

### 在代码中使用

```python
from focus_vision.focus_detector import FocusDetector

# 创建专注度检测器
detector = FocusDetector(model_path="yolo11n-pose.pt", focus_threshold=30)

# 处理视频
stats = detector.process_video(
    video_path="assets/test.mp4",  # 输入视频路径
    output_path="output/result.mp4",  # 输出视频路径
    csv_path="output/focus_data.csv"  # 输出CSV数据路径
)

# 打印统计信息
print(f"总帧数: {stats['total_frames']}")
print(f"检测到的学生数: {stats['total_students']}")
print(f"整体平均专注度: {stats['overall_avg_focus']:.2f}")
```

## 输出说明

### 视频输出

处理后的视频将包含：
- 学生骨架可视化（显示关键点和骨骼连接）
- 面部表情分析结果和情绪标签
- 每个学生的专注度评分实时显示
- 头部角度和姿态状态指示

### CSV数据

CSV文件包含以下列：
- `frame`: 帧编号
- `timestamp`: 时间戳
- `student_id`: 学生ID
- `head_angle`: 头部角度（度数）
- `focus_score`: 专注度评分（0-1，1表示完全专注）
- `emotion`: 检测到的主要情绪
- `emotion_confidence`: 情绪检测置信度

### 统计报告

程序运行结束后会显示并保存：
- 专注度随时间变化的折线图
- 不同情绪类型的分布饼图
- 每个学生的平均专注度条形图
- 总体统计数据摘要

## 工作原理

FocusVision的核心工作原理如下：

1. **姿态检测**：使用YOLOv11姿态检测模型识别视频中的人物和17个关键点
2. **头部角度计算**：通过分析头部关键点（鼻子、眼睛、耳朵）与肩膀的相对位置，计算头部俯仰角
3. **表情分析**：使用DeepFace提取和分析面部区域，识别主要情绪类型和置信度
4. **专注度评估**：
   - 头部姿态评分：根据头部角度判断是否低头或抬头
   - 表情评分：不同情绪对应不同的专注度系数
   - 综合评分：加权结合姿态和表情评分
5. **数据记录与可视化**：记录每帧数据，生成统计报告和可视化效果

## 应用场景

FocusVision适用于多种教育场景：

- **课堂教学**：实时监测学生专注度，帮助教师调整教学节奏
- **远程教育**：分析在线学习者的参与程度和专注状态
- **教学研究**：收集和分析课堂专注度数据，研究教学方法有效性
- **自适应学习系统**：为智能教育平台提供学生状态反馈
- **特殊教育**：辅助分析特殊学习需求学生的专注模式

## 局限性与注意事项

- **视频质量要求**：视频清晰度和光线条件会显著影响检测效果
- **姿态检测限制**：人物面部和上半身必须在画面中清晰可见
- **情绪分析约束**：面部表情检测在某些角度和光线条件下准确度可能降低
- **计算资源**：处理高分辨率视频需要较强的计算能力，GPU可显著提升性能
- **隐私考虑**：使用时应考虑学生隐私，获得适当的使用许可

## 未来改进方向

- **多模态分析**：结合音频分析（如回答问题的声音）增强专注度评估
- **学生识别**：加入人脸识别功能，跟踪特定学生的专注度变化
- **长时间跟踪**：改进对象跟踪算法，解决学生移动、遮挡问题
- **深度学习改进**：训练专门针对课堂场景的专注度检测模型
- **移动端支持**：开发轻量级模型支持移动设备运行
- **更多数据分析**：添加更丰富的统计报告和数据可视化功能