# FocusVision - 课堂专注度检测系统

基于YOLOv11姿态检测的课堂专注度监测应用，通过分析学生头部姿态来评估专注度。

## 功能特点

- 通过检测学生抬头/低头姿态来评估专注度
- 实时可视化学生骨架和专注度指标
- 记录每一帧所有学生的专注度数据并保存到CSV文件
- 保存可视化后的视频结果
- 生成专注度统计报告

## 安装要求

- Python 3.8+
- OpenCV
- NumPy
- Pandas
- Ultralytics (YOLOv11)

## 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/FocusVision.git
cd FocusVision
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 下载YOLOv11姿态检测模型（如果尚未下载）：

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

## 使用方法

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
python main.py --video assests/test.mp4 --output output/result.mp4 --csv output/focus_data.csv --threshold 25
```

### 在代码中使用

```python
from focus_vision.focus_detector import FocusDetector

# 创建专注度检测器
detector = FocusDetector(model_path="yolo11n-pose.pt", focus_threshold=30)

# 处理视频
stats = detector.process_video(
    video_path="assests/test.mp4",  # 输入视频路径
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
- 学生骨架可视化
- 每个学生的专注度评分显示

### CSV数据

CSV文件包含以下列：
- `frame`: 帧编号
- `timestamp`: 时间戳
- `student_id`: 学生ID
- `head_angle`: 头部角度
- `focus_score`: 专注度评分（0-1，1表示完全专注）

### 统计报告

程序运行结束后会显示：
- 总帧数
- 检测到的学生数量
- 整体平均专注度
- 每个学生的平均专注度

## 工作原理

1. 使用YOLOv11姿态检测模型识别视频中的人物和关键点
2. 通过分析头部与肩膀的相对位置计算头部角度
3. 根据头部角度评估专注度（头部垂直时专注度最高）
4. 记录数据并生成可视化结果

## 注意事项

- 确保视频中人物面部和上半身清晰可见
- 光线条件会影响检测效果
- 处理高分辨率视频可能需要较长时间