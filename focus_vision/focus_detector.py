import math

import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from ultralytics import YOLO
from focus_vision.utils import plot_pose_from_yolo_results

class FocusDetector:
    """
    课堂专注度检测器，基于YOLOv11姿态检测模型。
    通过分析学生头部姿态来评估专注度。
    """
    
    def __init__(self, model_path="yolo11n-pose.pt", focus_threshold=30):
        """
        初始化专注度检测器
        
        Args:
            model_path (str): YOLO模型路径
            focus_threshold (int): 头部角度阈值，超过此角度视为不专注（低头）
        """
        self.model = YOLO(model_path)
        self.focus_threshold = focus_threshold
        self.focus_data = []
        self.frame_count = 0

    def _calculate_head_angle(self, keypoints):
        """
        计算头部俯仰角（pitch），正值表示低头，负值表示抬头，
        并在任意朝向（侧脸、半侧）下通过耳朵不对称估计偏航进行校正。

        Args:
            keypoints: np.ndarray, COCO18 格式的关键点数组，shape=(N,2) 或 (N,3)
        Returns:
            float or None: 头部俯仰角度（度）
        """
        nose = keypoints[0, :2]
        left_ear = keypoints[3, :2]
        right_ear = keypoints[4, :2]
        left_shoulder = keypoints[5, :2]
        right_shoulder = keypoints[6, :2]

        # 3. 计算颈部中点
        neck = (left_shoulder + right_shoulder) / 2.0

        # 4. 原始头部向量（颈部 -> 鼻子）
        head_vec = nose - neck
        dx, dy = head_vec

        # 5. 用左右耳距离差估计偏航角 yaw
        d_left = np.linalg.norm(nose - left_ear)
        d_right = np.linalg.norm(nose - right_ear)
        sin_yaw = (d_left - d_right) / (d_left + d_right + 1e-6)
        sin_yaw = np.clip(sin_yaw, -1.0, 1.0)
        yaw = math.asin(sin_yaw)

        # 6. 用 cos(yaw) 校正水平分量
        dx_corr = dx / (math.cos(yaw) + 1e-6)
        vec_corr = np.array([dx_corr, dy])

        # 7. 计算 vec_corr 与垂直向上向量的夹角
        vertical = np.array([0.0, -1.0])
        dot_product = np.dot(vec_corr, vertical)
        norm_vec = np.linalg.norm(vec_corr)
        if norm_vec == 0:
            return None
        cos_angle = dot_product / (norm_vec * 1.0)
        angle_rad = math.acos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = math.degrees(angle_rad)

        # 8. 用叉乘决定正负（负值表示抬头，正值表示低头）
        cross_z = np.cross(np.append(vec_corr, 0.0), np.append(vertical, 0.0))[2]
        if cross_z < 0:
            angle_deg = -angle_deg

        return angle_deg

    def _assess_focus(self, angle):
        """
        根据头部角度评估专注度
        
        Args:
            angle: 头部角度
            
        Returns:
            float: 专注度评分 (0-1)，1表示完全专注
        """
        if angle is None:
            return 0.5  # 无法判断时返回中等专注度
            
        # 头部角度的绝对值越大，表示偏离垂直方向越多，专注度越低
        abs_angle = abs(angle)
        
        if abs_angle <= self.focus_threshold:
            # 在阈值范围内，专注度随角度线性下降
            focus_score = 1.0 - (abs_angle / self.focus_threshold) * 0.5
        else:
            # 超过阈值，专注度较低但仍有基础分
            focus_score = 0.5 * (1 - min(abs_angle - self.focus_threshold, 90) / 90)
            
        return focus_score
    
    def process_frame(self, frame):
        """
        处理单帧图像，检测姿态并评估专注度
        
        Args:
            frame: 输入图像帧
            
        Returns:
            tuple: (处理后的图像, 专注度数据字典)
        """
        self.frame_count += 1
        
        # 使用YOLO模型检测姿态
        results = self.model(frame, verbose=False)
        result = results[0]
        
        # 获取关键点
        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []
        
        # 获取边界框
        boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else None
        
        # 创建用于可视化的图像
        vis_img = frame.copy()
        
        # 当前帧的专注度数据
        frame_data = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'students': []
        }
        
        # 处理每个检测到的人
        for i, kpts in enumerate(keypoints):
            # 计算头部角度
            head_angle = self._calculate_head_angle(kpts)
            
            # 评估专注度
            focus_score = self._assess_focus(head_angle)
            
            # 记录数据
            student_data = {
                'id': i,
                'head_angle': head_angle if head_angle is not None else float('nan'),
                'focus_score': focus_score
            }
            frame_data['students'].append(student_data)
            
            # 在图像上标注专注度
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = map(int, boxes[i][:4])
                label = f"Focus: {focus_score:.2f}"
                cv2.putText(vis_img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 可视化姿态关键点和骨架
        if len(keypoints) > 0:
            vis_img = plot_pose_from_yolo_results(
                image=vis_img,
                yolo_keypoints=result.keypoints,
            )
        
        self.focus_data.append(frame_data)
        
        return vis_img, frame_data
    
    def process_video(self, video_path, output_path=None, csv_path=None):
        """
        处理视频文件，检测专注度并保存结果
        
        Args:
            video_path (str): 输入视频路径
            output_path (str, optional): 输出视频路径
            csv_path (str, optional): 输出CSV数据路径
            
        Returns:
            dict: 处理结果统计信息
        """
        # 重置数据
        self.focus_data = []
        self.frame_count = 0
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出视频
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        else:
            out = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            vis_frame, frame_data = self.process_frame(frame)
            
            if out:
                out.write(vis_frame)
            
            if self.frame_count % 10 == 0:
                print(f"处理进度: {self.frame_count}/{total_frames} ({self.frame_count/total_frames*100:.1f}%)")
        
        cap.release()
        if out:
            out.release()
        
        if csv_path:
            self._save_csv(csv_path)
        
        stats = self._calculate_statistics()
        return stats
    
    def _save_csv(self, csv_path):
        """
        将专注度数据保存为CSV文件
        
        Args:
            csv_path (str): CSV文件保存路径
        """
        # 创建一个扁平化的数据列表
        flat_data = []
        
        for frame_data in self.focus_data:
            frame_num = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            if not frame_data['students']:
                # 如果没有检测到学生，添加一条空记录
                flat_data.append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'student_id': None,
                    'head_angle': None,
                    'focus_score': None
                })
            else:
                # 为每个学生添加一条记录
                for student in frame_data['students']:
                    flat_data.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'student_id': student['id'],
                        'head_angle': student['head_angle'],
                        'focus_score': student['focus_score']
                    })
        
        df = pd.DataFrame(flat_data)
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV
        df.to_csv(str(csv_path), index=False)
        print(f"专注度数据已保存至: {csv_path}")
    
    def _calculate_statistics(self):
        """
        计算专注度统计信息
        
        Returns:
            dict: 统计信息
        """
        if not self.focus_data:
            return {}
        
        # 计算每个学生的平均专注度
        student_focus = {}
        for frame_data in self.focus_data:
            for student in frame_data['students']:
                student_id = student['id']
                focus_score = student['focus_score']
                
                if student_id not in student_focus:
                    student_focus[student_id] = []
                    
                student_focus[student_id].append(focus_score)
        
        # 计算平均值
        avg_focus = {}
        for student_id, scores in student_focus.items():
            avg_focus[student_id] = sum(scores) / len(scores)
        
        # 计算整体平均专注度
        all_scores = [score for scores in student_focus.values() for score in scores]
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return {
            'total_frames': self.frame_count,
            'total_students': len(student_focus),
            'student_avg_focus': avg_focus,
            'overall_avg_focus': overall_avg
        }