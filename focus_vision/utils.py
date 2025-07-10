import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Any, Dict, Union, List

# --- 默认 COCO 17 个关键点的连接方式 ---
DEFAULT_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]
# 调整索引以匹配 COCO (索引从0开始)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
DEFAULT_SKELETON_COCO = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]


# --- 默认调色板 ---
DEFAULT_PALETTE = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 153, 255),
    (153, 204, 255), (255, 102, 255), (255, 51, 255), (102, 178, 255), (51, 153, 255),
    (255, 153, 153), (255, 102, 102), (255, 51, 51), (153, 255, 153), (102, 255, 102),
    (51, 255, 51), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255),
]

def estimate_pitch(nose, left_eye, right_eye, left_ear, right_ear):
    """
    估计人头部的俯仰角（pitch）
    
    该函数通过脸部关键点的相对位置计算头部俯仰角度，
    并使用左右耳的距离差来校正头部的偏航角影响。
    
    算法步骤：
    1. 计算眼睛中心点位置
    2. 计算鼻子到眼睛中心的向量
    3. 估计头部的偏航角（yaw）
    4. 校正水平分量
    5. 计算俯仰角度
    
    Args:
        nose (array-like): 鼻子关键点的坐标 [x, y]
        left_eye (array-like): 左眼关键点的坐标 [x, y]
        right_eye (array-like): 右眼关键点的坐标 [x, y]
        left_ear (array-like): 左耳关键点的坐标 [x, y]
        right_ear (array-like): 右耳关键点的坐标 [x, y]
        
    Returns:
        float: 头部俯仰角（以度为单位），正值表示低头，负值表示抬头
    """
    # 1. 眼睛中心
    eye_cx = (left_eye[0] + right_eye[0]) / 2
    eye_cy = (left_eye[1] + right_eye[1]) / 2

    # 2. 鼻子到眼睛中心的向量分量
    dx = nose[0] - eye_cx
    dy = nose[1] - eye_cy

    # 3. 计算鼻子到左右耳的距离
    d_left  = math.hypot(nose[0] - left_ear[0],  nose[1] - left_ear[1])
    d_right = math.hypot(nose[0] - right_ear[0], nose[1] - right_ear[1])

    # 4. 估计偏航：sin(yaw) ≈ (d_left - d_right) / (d_left + d_right)
    sin_yaw = (d_left - d_right) / (d_left + d_right + 1e-6)
    sin_yaw = max(-1, min(1, sin_yaw))
    yaw = math.asin(sin_yaw)

    # 5. 用偏航校正水平分量（投影缩放校正）
    dx_corr = dx / math.cos(yaw)

    # 6. 计算俯仰角（rad -> deg）
    pitch = math.degrees(math.atan2(dy, dx_corr))
    return pitch

def plot_pose_from_yolo_results(
    image: np.ndarray,
    yolo_keypoints: Optional[Any],
    yolo_boxes: Optional[Any] = None,
    score_threshold: float = 0.0,
    point_radius: int = 4,
    line_thickness: int = 2,
    box_thickness: int = 2,
    draw_connections: bool = True,
    skeleton: list = DEFAULT_SKELETON_COCO,
    palette: list = DEFAULT_PALETTE,
    draw_labels: bool = False,
    label_font_scale: float = 0.4,
    label_thickness: int = 1,
    draw_boxes: bool = False
) -> np.ndarray:
    """
    在图像上绘制姿态估计的关键点和骨架，输入基于 YOLOv8 Pose 输出格式。
    
    该函数用于可视化YOLO姿态检测模型的输出结果，将检测到的人体关键点和骨架
    绘制在原始图像上。主要用于专注度检测系统中的可视化展示部分。
    
    功能包括：
    1. 绘制人体骨架的关键点（如鼻子、眼睛、耳朵、肩膀等）
    2. 连接关键点形成骨架线条
    3. 可选绘制边界框和关键点标签
    4. 支持多人同时绘制，每个人使用不同颜色

    Args:
        image (np.ndarray): 输入图像 (OpenCV BGR 格式).
        yolo_keypoints (object or None): 包含关键点数据的对象。
                                         期望至少有 .xy (N, K, 2) 和 .conf (N, K) 属性。
                                         可以传入 None 如果没有检测到关键点。
        yolo_boxes (object or None, optional): 包含边界框数据的对象。
                                                期望至少有 .xyxy (N, 4) 属性。
                                                默认为 None。
        score_threshold (float, optional): 仅绘制置信度高于此阈值的关键点。默认为 0.0。
        point_radius (int, optional): 绘制关键点的半径。默认为 4。
        line_thickness (int, optional): 绘制骨架连接线的厚度。默认为 2。
        box_thickness (int, optional): 绘制边界框的厚度。默认为 2。
        draw_connections (bool, optional): 是否绘制骨架连接线。默认为 True。
        skeleton (list, optional): 定义关键点如何连接的列表，每个元素是 [idx1, idx2]。
                                   默认为 DEFAULT_SKELETON_COCO (COCO 17 关键点索引)。
        palette (list, optional): 用于绘制骨架连接线和点的颜色列表 (BGR)。默认为 DEFAULT_PALETTE。
        draw_labels (bool, optional): 是否在关键点旁边绘制其索引号。默认为 False。
        label_font_scale (float, optional): 关键点标签的字体大小。默认为 0.4。
        label_thickness (int, optional): 关键点标签的字体厚度。默认为 1。
        draw_boxes (bool, optional): 是否绘制边界框（如果提供了 yolo_boxes）。默认为 False。

    Returns:
        np.ndarray: 绘制了标注的图像副本。

    Raises:
        AttributeError: 如果传入的 yolo_keypoints 或 yolo_boxes 对象缺少必要的属性（如 .xy, .conf, .xyxy）。
        ValueError: 如果关键点和边界框的数量不匹配（如果两者都提供）。
    """
    output_image = image.copy()

    if yolo_keypoints is None or not hasattr(yolo_keypoints, 'xy') or not hasattr(yolo_keypoints, 'conf') or yolo_keypoints.xy is None or yolo_keypoints.conf is None:
        print("Info: No valid keypoint data provided or found.")
        if not draw_boxes or yolo_boxes is None or not hasattr(yolo_boxes, 'xyxy') or yolo_boxes.xyxy is None:
           return output_image
    else:
        kpts_xy = yolo_keypoints.xy      # (N, K, 2)
        kpts_conf = yolo_keypoints.conf  # (N, K)
        num_instances, num_keypoints = kpts_xy.shape[0], kpts_xy.shape[1]

        if num_instances == 0:
             print("Info: Keypoint data exists but contains 0 instances.")
             if not draw_boxes or yolo_boxes is None or not hasattr(yolo_boxes, 'xyxy') or yolo_boxes.xyxy is None:
                return output_image


    # 检查是否有边界框数据
    boxes_xyxy = None
    num_boxes = 0
    if draw_boxes and yolo_boxes is not None and hasattr(yolo_boxes, 'xyxy') and yolo_boxes.xyxy is not None:
        boxes_xyxy = yolo_boxes.xyxy     # (N, 4)
        num_boxes = boxes_xyxy.shape[0]
        # 校验数量是否匹配（如果同时有关节点和框）
        if 'num_instances' in locals() and num_boxes != num_instances:
            print(f"Warning: Mismatch between number of boxes ({num_boxes}) and keypoint instances ({num_instances}). Ensure results correspond correctly.")
            # 可以选择不绘制框，或者继续绘制但可能不匹配
            # boxes_xyxy = None # 取消绘制框
    elif draw_boxes:
         print("Info: draw_boxes is True, but no valid box data provided.")


    # 遍历每个检测到的实例 (人)
    # 如果只有框，则只循环框；如果有关节点，则循环关节点；如果都有，以关节点数量为准（假设它们匹配）
    loop_count = 0
    if 'num_instances' in locals():
        loop_count = num_instances
    elif boxes_xyxy is not None:
        loop_count = num_boxes


    for i in range(loop_count):
        instance_has_keypoints = 'kpts_xy' in locals() and i < num_instances
        instance_has_box = boxes_xyxy is not None and i < num_boxes

        # --- 绘制边界框 ---
        if instance_has_box:
            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
            # 使用实例索引来选择颜色，增加多样性
            box_color = palette[i % len(palette)]
            # box_color = (0, 255, 0) # 固定绿色
            cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, box_thickness)
            # 可选：添加置信度标签
            # if hasattr(yolo_boxes, 'conf') and yolo_boxes.conf is not None:
            #     box_conf = yolo_boxes.conf[i]
            #     label = f"Conf: {box_conf:.2f}"
            #     cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, box_thickness)


        # --- 绘制关键点和骨架 ---
        if instance_has_keypoints:
            instance_kpts_xy = kpts_xy[i]     # (K, 2)
            instance_kpts_conf = kpts_conf[i] # (K,)

            valid_kpt_indices_instance = [] # 当前实例的有效关键点索引
            drawn_kpt_coords = {}           # 存储绘制出的点坐标 {kpt_idx: (x, y)}

            # 1. 绘制关键点
            for k in range(num_keypoints):
                x, y = instance_kpts_xy[k]
                conf = instance_kpts_conf[k]

                if conf >= score_threshold and x > 0 and y > 0: # 检查置信度和有效性
                    center = (int(x), int(y))
                    point_color = palette[k % len(palette)] # 用关键点索引定颜色
                    cv2.circle(output_image, center, point_radius, point_color, -1) # 实心圆
                    valid_kpt_indices_instance.append(k)
                    drawn_kpt_coords[k] = center

                    # 绘制标签
                    if draw_labels:
                        label_color = (255, 255, 255) # 白色标签
                        cv2.putText(output_image, str(k), (center[0] + point_radius // 2, center[1] - point_radius // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_color, label_thickness, cv2.LINE_AA)

            # 2. 绘制骨架连接线
            if draw_connections and skeleton:
                for conn_idx, connection in enumerate(skeleton):
                    kpt_idx1, kpt_idx2 = connection
                    # 确保索引有效且对应的关键点都被绘制了
                    if kpt_idx1 in valid_kpt_indices_instance and kpt_idx2 in valid_kpt_indices_instance:
                        pt1 = drawn_kpt_coords[kpt_idx1]
                        pt2 = drawn_kpt_coords[kpt_idx2]
                        line_color = palette[conn_idx % len(palette)] # 用连接线索引定颜色
                        cv2.line(output_image, pt1, pt2, line_color, line_thickness, cv2.LINE_AA) # 抗锯齿

    return output_image

def plot_attention_over_time(
    data: Union[pd.DataFrame, Dict[str, List[float]]],
    output_path: str,
    x_col: str = "frame",
    y_col: str = "focus_score",
    student_id_col: str = "student_id",
    title: str = "专注度随时间变化",
    student_id: Optional[str] = None,
    include_emotions: bool = True,
    emotion_col: str = "emotion",
    figsize: tuple = (12, 8),
    dpi: int = 100
):
    """
    绘制专注度随时间变化的曲线图，并可选择性地显示情绪数据
    
    该函数用于可视化学生专注度的时间序列数据，支持以下功能：
    1. 绘制专注度随时间变化的折线图
    2. 可选地标记不同情绪状态的数据点
    3. 支持筛选单个学生或显示所有学生的数据
    4. 添加图例、网格线和标题
    5. 将图表保存为图像文件
    
    在专注度检测系统中，该函数用于生成最终的分析报告和可视化结果。

    Args:
        data: 专注度数据，可以是pandas DataFrame或包含列表的字典
              必须包含frame/timestamp、focus_score和student_id列
        output_path: 图表保存路径（PNG格式）
        x_col: X轴列名（默认为"frame"帧数）
        y_col: Y轴列名（默认为"focus_score"专注度分数）
        student_id_col: 学生ID列名
        title: 图表标题
        student_id: 要筛选的特定学生ID，None表示所有学生
        include_emotions: 是否在图表中包含情绪数据
        emotion_col: 情绪数据的列名
        figsize: 图表尺寸（宽度，高度）（英寸）
        dpi: 图表分辨率（每英寸点数）

    Returns:
        None: 图表将保存到指定路径
    """
    # 设置默认的Matplotlib样式
    plt.style.use('ggplot')
    
    # 确保输入数据是DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # 如果数据为空，创建一个简单的提示图
    if data.empty:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.text(0.5, 0.5, "No Available Data", ha='center', va='center', fontsize=20)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return

    # 创建图形
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 检查是否需要筛选特定学生
    if student_id is not None:
        if student_id_col in data.columns:
            data = data[data[student_id_col] == student_id]
        else:
            print(f"警告: 找不到'{student_id_col}'列，将使用所有数据")
    
    # 确保所需的列存在
    if x_col not in data.columns:
        print(f"错误: 找不到'{x_col}'列，使用索引作为X轴")
        data[x_col] = data.index
    
    if y_col not in data.columns:
        print(f"错误: 找不到'{y_col}'列，无法绘制专注度")
        return
    
    # 绘制专注度曲线
    if student_id_col in data.columns and len(data[student_id_col].unique()) > 1:
        # 多个学生，使用不同颜色
        for sid, group in data.groupby(student_id_col):
            label = f"Student {sid}" if sid is not None else "Unidentified"
            ax1.plot(group[x_col], group[y_col], '-o', markersize=4, linewidth=2, label=label)
    else:
        # 单个学生或没有学生ID列
        ax1.plot(data[x_col], data[y_col], '-o', markersize=4, linewidth=2, color='blue', label='专注度')
    
    # 设置Y轴范围为0-1，因为专注度分数通常在这个范围内
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Frame' if x_col == "frame" else x_col)
    ax1.set_ylabel('Focus Score')
    
    # 如果要包含情绪数据并且存在情绪列
    if include_emotions and emotion_col in data.columns:
        # 在右侧创建一个标记情绪的轴
        ax2 = ax1.twinx()
        
        # 情绪类别到数值的映射
        emotion_categories = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 
            'fear': 4, 'disgust': 5, 'surprise': 6, 'unknown': 7
        }
        
        # 情绪类别到颜色的映射
        emotion_colors = {
            'neutral': 'gray', 'happy': 'yellow', 'sad': 'blue',
            'angry': 'red', 'fear': 'purple', 'disgust': 'brown',
            'surprise': 'cyan', 'unknown': 'black'
        }
        
        # 转换情绪为数值
        data['emotion_value'] = data[emotion_col].apply(
            lambda x: emotion_categories.get(x, 7) if isinstance(x, str) else 7
        )
        
        # 如果有多个学生，为每个学生绘制情绪
        if student_id_col in data.columns and len(data[student_id_col].unique()) > 1:
            for sid, group in data.groupby(student_id_col):
                # 为每种情绪使用不同的标记和颜色
                for emotion, value in emotion_categories.items():
                    emotion_data = group[group[emotion_col] == emotion]
                    if not emotion_data.empty:
                        ax2.scatter(
                            emotion_data[x_col],
                            [value] * len(emotion_data),
                            marker='s',
                            color=emotion_colors.get(emotion, 'black'),
                            s=50,
                            alpha=0.7,
                            label=f"{emotion} (Student {sid})"
                        )
        else:
            # 单个学生或没有学生ID列
            for emotion, value in emotion_categories.items():
                emotion_data = data[data[emotion_col] == emotion]
                if not emotion_data.empty:
                    ax2.scatter(
                        emotion_data[x_col],
                        [value] * len(emotion_data),
                        marker='s',
                        color=emotion_colors.get(emotion, 'black'),
                        s=50,
                        alpha=0.7,
                        label=emotion
                    )
        
        # 设置右侧Y轴
        ax2.set_ylabel('Emotion Category')
        ax2.set_yticks(list(emotion_categories.values()))
        ax2.set_yticklabels(list(emotion_categories.keys()))
        ax2.grid(False)
    
    # 添加图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    if include_emotions and emotion_col in data.columns:
        handles2, labels2 = ax2.get_legend_handles_labels()
        # 合并图例并去重
        by_label = dict(zip(labels1 + labels2, handles1 + handles2))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    # 添加网格和标题
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path)
    plt.close()
    
    print(f"图表已保存至: {output_path}")