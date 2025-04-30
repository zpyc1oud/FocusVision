import math

import cv2
import numpy as np
from typing import Optional, Any

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