import math
import cv2
import numpy as np
import pandas as pd
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ultralytics import YOLO
from focus_vision.utils import plot_pose_from_yolo_results
from deepface import DeepFace

class FocusDetector:
    """
    课堂专注度检测器，基于YOLOv11姿态检测模型和DeepFace表情识别。
    
    该检测器通过两个主要维度评估学生的专注度：
    1. 头部姿态：通过计算头部俯仰角度检测学生是否保持抬头状态
    2. 面部表情：分析面部表情识别是否处于专注状态相关的情绪
    
    检测过程包括：
    - 使用YOLO模型识别人体骨架关键点
    - 计算头部与颈部的角度判断姿态
    - 使用DeepFace提取面部表情并识别情绪
    - 综合两个维度评估专注度
    
    最终可生成专注度随时间变化的数据，用于课堂教学分析和优化。
    """
    
    def __init__(self, model_path="yolo11n-pose.pt", focus_threshold=30, emotion_weight=0.3, use_gpu=False, model_pool_size=1, deepface_concurrency=1):
        """
        初始化专注度检测器
        
        Args:
            model_path (str): YOLO姿态检测模型路径，默认使用yolo11n-pose.pt轻量级模型
            focus_threshold (int): 头部角度阈值(度)，超过此角度视为不专注（低头），默认30度
            emotion_weight (float): 表情在专注度评估中的权重 (0-1)，默认0.3，表示表情因素占30%权重
            use_gpu (bool): 是否使用GPU加速推理，默认False。自动检测GPU可用性
            model_pool_size (int): 并行模型实例数量，默认1（不开并行）
            deepface_concurrency (int): DeepFace并发分析的最大并发数，默认1（串行，最安全）
        """
        # 设置模型设备
        self.device = "cuda" if use_gpu else "cpu"
        print(f"使用设备: {self.device} 进行专注度检测")
        
        # 加载YOLO模型池，指定设备
        self.model_pool_size = max(1, int(model_pool_size))
        self.models = [YOLO(model_path) for _ in range(self.model_pool_size)]
        if use_gpu:
            for m in self.models:
                m.to(self.device)
        # 兼容原先接口，保留主模型引用
        self.model = self.models[0]
        
        # 配置DeepFace (DeepFace会自动使用可用的GPU)
        self.use_gpu = use_gpu
        
        # 其他设置
        self.focus_threshold = focus_threshold
        self.emotion_weight = emotion_weight
        self.focus_data = []
        self.frame_count = 0
        self._state_lock = threading.Lock()
        # DeepFace并发控制：使用有界信号量允许有限并发，默认1为串行
        self.deepface_concurrency = max(1, int(deepface_concurrency))
        self._deepface_semaphore = threading.BoundedSemaphore(value=self.deepface_concurrency)
        # 并行执行器与模型分配队列
        self._executor = ThreadPoolExecutor(max_workers=self.model_pool_size)
        self._model_queue = queue.Queue()
        for idx in range(self.model_pool_size):
            self._model_queue.put(idx)
        
        # 表情与专注度的映射关系，表示各种表情对应的专注度系数（0-1）
        self.emotion_focus_map = {
            'neutral': 0.9,   # 中性表情通常表示专注
            'happy': 0.7,     # 快乐可能表示参与度高，但不一定最专注
            'sad': 0.5,       # 悲伤可能表示不够投入
            'angry': 0.4,     # 生气通常表示注意力分散
            'fear': 0.3,      # 害怕通常表示注意力被干扰
            'disgust': 0.3,   # 厌恶表示负面体验，专注度低
            'surprise': 0.6,  # 惊讶可能是积极参与的标志，但也可能是分心
        }
        
        # 表情颜色映射
        self.emotion_color_map = {
            'neutral': (255, 255, 255),  # 白色
            'happy': (0, 255, 255),      # 黄色
            'sad': (255, 0, 0),          # 蓝色
            'angry': (0, 0, 255),        # 红色
            'fear': (0, 140, 255),       # 橙色
            'disgust': (0, 0, 128),      # 深红色
            'surprise': (255, 255, 0),   # 青色
        }

    def _calculate_head_angle(self, keypoints):
        """
        计算头部俯仰角（pitch），正值表示低头，负值表示抬头。
        
        该算法通过以下步骤计算头部姿态：
        1. 提取关键点（鼻子、耳朵和肩膀）
        2. 计算颈部中点位置
        3. 计算颈部到鼻子的向量
        4. 使用左右耳距离差估计头部偏航角(yaw)
        5. 用偏航角校正头部向量的水平分量
        6. 计算校正后向量与垂直方向的夹角
        
        这种方法在侧脸和各种头部朝向下具有较好的鲁棒性，可以准确估计抬头/低头程度。

        Args:
            keypoints (np.ndarray): COCO18格式的关键点数组，shape=(N,2)或(N,3)
                索引说明：
                0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳,
                5: 左肩, 6: 右肩, 其他关键点不在本函数中使用
                
        Returns:
            float or None: 头部俯仰角度(度)，正值表示低头，负值表示抬头
                           如果关键点不足以计算角度，则返回None
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
        
        评估逻辑：
        1. 如果角度无法计算（关键点缺失），返回中等专注度0.5
        2. 在阈值范围内，专注度随角度增大线性下降：
           - 0度（完全垂直）对应专注度1.0
           - 阈值角度对应专注度0.5
        3. 超过阈值，专注度进一步下降，但保持基础分，避免极端值

        Args:
            angle (float or None): 头部角度，正值表示低头，负值表示抬头
            
        Returns:
            float: 专注度评分 (0-1)，1表示完全专注，0表示完全不专注
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
    
    def _analyze_emotion(self, face_img):
        """
        使用DeepFace分析面部表情
        
        该方法实现：
        1. 提取人脸区域的表情特征
        2. 识别主要情绪类型及置信度
        3. 根据情绪类型映射到专注度评分
        
        Args:
            face_img (np.ndarray): 面部图像区域，RGB格式的numpy数组
            
        Returns:
            tuple: (主要情绪类型(str), 情绪得分字典(dict), 专注度评分(float))
                  主要情绪类型：如'neutral', 'happy', 'sad'等
                  情绪得分字典：包含各情绪类型及其对应分数
                  专注度评分：0-1的值，表示基于表情的专注度
        """
        try:
            deepface_options = {
                'actions': ['emotion'],
                'enforce_detection': False,
                'silent': True,
                'detector_backend': 'yolov11n',  # 使用YOLO最小模型进行人脸检测
            }
            # 执行情绪分析（通过信号量限制并发，默认串行）
            self._deepface_semaphore.acquire()
            try:
                emotion_result = DeepFace.analyze(face_img, **deepface_options)
            finally:
                self._deepface_semaphore.release()
            
            # 在列表情况下取第一个结果
            if isinstance(emotion_result, list):
                emotion_result = emotion_result[0]
            
            # 获取情绪评分
            emotion_scores = emotion_result['emotion']
            dominant_emotion = emotion_result['dominant_emotion']
            
            # 计算情绪对应的专注度评分
            emotion_focus_score = self.emotion_focus_map.get(dominant_emotion, 0.5)
            
            return dominant_emotion, emotion_scores, emotion_focus_score

        except Exception as e:
            print(f"表情分析错误: {e}")
            return "unknown", {}, 0.5

    def _detect_faces_emotions(self, frame):
        """
        在整帧上使用DeepFace（yolov11n后端）检测所有人脸并分析情绪。

        返回值为列表，每个元素包含：
        - 'region': {'x','y','w','h'} 人脸框
        - 'dominant_emotion': 主要情绪
        - 'emotion': 各情绪置信度字典
        """
        try:
            options = {
                'actions': ['emotion'],
                'detector_backend': 'yolov11n',
                'enforce_detection': False,
                'silent': True,
                'align': True,
            }
            # 使用信号量控制DeepFace分析并发，避免会话竞态
            self._deepface_semaphore.acquire()
            try:
                res = DeepFace.analyze(frame, **options)
            finally:
                self._deepface_semaphore.release()

            # 规范化为列表[dict]
            if isinstance(res, dict):
                faces = [res]
            elif isinstance(res, list):
                # 可能是 [dict, dict, ...] 或 [[dict, ...]]
                if len(res) > 0 and isinstance(res[0], dict):
                    faces = res
                elif len(res) > 0 and isinstance(res[0], list):
                    faces = res[0]
                else:
                    faces = []
            else:
                faces = []

            return faces
        except Exception as e:
            print(f"整帧人脸情绪检测失败: {e}")
            return []

    @staticmethod
    def _iou(boxA, boxB):
        """计算两个框的IoU，框格式为(x1,y1,x2,y2)"""
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = areaA + areaB - inter + 1e-6
        return inter / union

    def _match_face_to_person(self, person_box, faces):
        """
        将DeepFace检测到的人脸与YOLO人物框进行匹配。

        优先策略：
        1) 选取中心点落在人物框内的人脸，按IoU最高取最佳
        2) 若无中心点在内，则选择IoU最高的人脸
        """
        px1, py1, px2, py2 = map(float, person_box[:4])
        best = None
        best_iou = 0.0
        best_inside = False

        for f in faces:
            r = f.get('region', {})
            fx, fy, fw, fh = r.get('x'), r.get('y'), r.get('w'), r.get('h')
            if fx is None or fy is None or fw is None or fh is None:
                continue
            fx1, fy1 = float(fx), float(fy)
            fx2, fy2 = fx1 + float(fw), fy1 + float(fh)
            cx, cy = fx1 + fw / 2.0, fy1 + fh / 2.0
            inside = (px1 <= cx <= px2) and (py1 <= cy <= py2)
            iou = self._iou((px1, py1, px2, py2), (fx1, fy1, fx2, fy2))

            if inside:
                if (not best_inside) or (iou > best_iou):
                    best = f
                    best_iou = iou
                    best_inside = True
            else:
                if (not best_inside) and (iou > best_iou):
                    best = f
                    best_iou = iou

        return best
    
    def _calculate_combined_focus(self, pose_focus, emotion_focus):
        """
        综合姿态和表情计算最终专注度
        
        使用加权平均方法结合两种专注度评分：
        - 姿态专注度：占比(1-emotion_weight)
        - 表情专注度：占比emotion_weight
        
        Args:
            pose_focus (float): 基于姿态的专注度评分 (0-1)
            emotion_focus (float): 基于表情的专注度评分 (0-1)
            
        Returns:
            float: 综合专注度评分 (0-1)
        """
        # 根据权重组合两种专注度
        combined_focus = (1 - self.emotion_weight) * pose_focus + self.emotion_weight * emotion_focus
        return combined_focus
    
    def process_frame(self, frame, model_override=None):
        """
        处理单帧图像，检测姿态并评估专注度
        
        处理步骤：
        1. 使用YOLO模型检测人体姿态和关键点
        2. 对每个检测到的人：
           a. 计算头部角度
           b. 评估基于姿态的专注度
           c. 提取面部区域进行表情分析
           d. 综合姿态和表情评估专注度
        3. 在图像上可视化结果
        4. 记录专注度数据
        
        Args:
            frame (np.ndarray): 输入图像帧，BGR格式
            
        Returns:
            tuple: (处理后的图像(np.ndarray), 专注度数据字典(dict))
                  专注度数据字典包含帧号、时间戳和每个检测到人的专注度信息
        """
        with self._state_lock:
            self.frame_count += 1
        
        # 使用YOLO模型检测姿态，根据配置指定设备
        model_to_use = model_override if model_override is not None else self.model
        results = model_to_use(frame, verbose=False, device=self.device)
        result = results[0]
        
        # 获取关键点
        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []
        
        # 获取边界框
        boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else None

        # 在整帧上用DeepFace检测所有人脸及其情绪
        faces_info = self._detect_faces_emotions(frame)
        
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
            
            # 评估基于姿态的专注度
            pose_focus_score = self._assess_focus(head_angle)
            
            # 匹配该人物对应的人脸框并获取情绪
            dominant_emotion = "unknown"
            emotion_scores = {}
            emotion_focus_score = 0.5

            matched_face = None
            if boxes is not None and i < len(boxes):
                matched_face = self._match_face_to_person(boxes[i], faces_info)

            if matched_face is not None:
                dominant_emotion = matched_face.get('dominant_emotion', 'unknown')
                emotion_scores = matched_face.get('emotion', {})
                emotion_focus_score = self.emotion_focus_map.get(dominant_emotion, 0.5)
            else:
                # 回退：用人物框裁剪区域进行单独分析（确保仍可得到情绪）
                print(f"未匹配到人脸框，回退到人物框分析，人物框索引 {i}")
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[i][:4])
                    face_h = y2 - y1
                    face_w = x2 - x1
                    y1 = max(0, y1 - int(face_h * 0.1))
                    x1 = max(0, x1 - int(face_w * 0.1))
                    x2 = min(frame.shape[1], x2 + int(face_w * 0.1))
                    face_img = frame[y1:y2, x1:x2]
                    if face_img is not None and face_img.size > 0:
                        dominant_emotion, emotion_scores, emotion_focus_score = self._analyze_emotion(face_img)
            
            # 计算综合专注度
            combined_focus_score = self._calculate_combined_focus(pose_focus_score, emotion_focus_score)
            
            # 记录数据
            student_data = {
                'id': i,
                'head_angle': head_angle if head_angle is not None else float('nan'),
                'pose_focus_score': pose_focus_score,
                'emotion': dominant_emotion,
                'emotion_scores': emotion_scores,
                'emotion_focus_score': emotion_focus_score,
                'focus_score': combined_focus_score
            }
            frame_data['students'].append(student_data)
            
            # 在图像上标注专注度和表情
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = map(int, boxes[i][:4])
                
                # 确定表情的颜色
                emotion_color = self.emotion_color_map.get(dominant_emotion, (255, 255, 255))
                
                # 绘制专注度标签
                focus_label = f"Focus: {combined_focus_score:.2f}"
                cv2.putText(vis_img, focus_label, (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 绘制表情标签
                emotion_label = f"Emotion: {dominant_emotion}"
                cv2.putText(vis_img, emotion_label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
        
        # 可视化姿态关键点和骨架
        if len(keypoints) > 0:
            vis_img = plot_pose_from_yolo_results(
                image=vis_img,
                yolo_keypoints=result.keypoints,
            )
        
        with self._state_lock:
            self.focus_data.append(frame_data)
        
        return vis_img, frame_data

    def process_frame_async(self, frame):
        """
        异步提交帧进行并行处理，返回Future。
        使用内部模型池动态分配可用模型实例，保证线程安全。
        """
        # 申请一个可用的模型索引
        model_idx = self._model_queue.get()

        def _worker(frm, idx):
            try:
                model = self.models[idx]
                return self.process_frame(frm, model_override=model)
            except Exception as e:
                # 发生错误时返回占位结果，避免阻塞
                placeholder = frm
                err_frame_data = {
                    'frame': -1,
                    'timestamp': time.time(),
                    'students': [],
                    'error': str(e)
                }
                return placeholder, err_frame_data
            finally:
                # 模型归还到队列
                self._model_queue.put(idx)

        return self._executor.submit(_worker, frame, model_idx)

    def shutdown(self):
        """关闭线程池并进行资源清理"""
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            pass
    
    def process_video(self, video_path, output_path=None, csv_path=None, process_interval=1):
        """
        处理整个视频，检测所有帧中的专注度
        
        处理流程：
        1. 打开视频并读取帧
        2. 每隔process_interval帧处理一帧
        3. 对每帧调用process_frame进行专注度检测
        4. 记录专注度数据
        5. 如果指定了输出路径，将处理后的视频保存
        6. 如果指定了CSV路径，将专注度数据保存为CSV
        7. 返回处理统计信息
        
        Args:
            video_path (str): 输入视频文件路径
            output_path (str, optional): 输出视频文件路径，不指定则不保存
            csv_path (str, optional): 输出CSV数据文件路径，不指定则不保存
            process_interval (int, optional): 处理帧间隔，默认为1（处理每一帧）
                                              设置大于1的值可以加速处理
            
        Returns:
            dict: 处理统计信息，包括总帧数、学生数量、平均专注度等
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
        
        # 性能测量变量
        start_time = time.time()
        frame_counter = 0
        processing_times = []
        
        # 报告处理设备信息
        device_info = f"GPU ({self.device})" if self.use_gpu else "CPU"
        print(f"使用 {device_info} 处理视频，处理间隔: {process_interval} 帧")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根据间隔决定是否处理此帧
            if frame_counter % process_interval == 0:
                # 测量每帧处理时间
                frame_start_time = time.time()
                
                vis_frame, frame_data = self.process_frame(frame)
                
                # 记录处理时间
                frame_processing_time = time.time() - frame_start_time
                processing_times.append(frame_processing_time)
                
                if out:
                    out.write(vis_frame)
            else:
                # 对于不处理的帧，只写入原始帧
                if out:
                    out.write(frame)
            
            frame_counter += 1
            
            if frame_counter % 10 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_counter / elapsed
                avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
                
                print(f"处理进度: {frame_counter}/{total_frames} ({frame_counter/total_frames*100:.1f}%) | " 
                      f"平均FPS: {avg_fps:.2f} | 平均处理时间: {avg_processing_time*1000:.1f}ms/帧")
        
        # 计算和报告最终性能
        total_time = time.time() - start_time
        avg_fps = frame_counter / total_time
        avg_processing_ms = sum(processing_times) / len(processing_times) * 1000 if processing_times else 0
        
        print(f"\n处理完成:")
        print(f"总帧数: {frame_counter}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均速度: {avg_fps:.2f} FPS")
        print(f"平均处理时间: {avg_processing_ms:.1f}ms/帧")
        print(f"处理设备: {device_info}")
        
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
        
        将收集的所有帧的专注度数据整合成DataFrame并保存为CSV。
        生成的CSV文件包含以下列：
        - frame: 帧号
        - timestamp: 时间戳
        - student_id: 学生ID
        - head_angle: 头部角度
        - focus_score: 专注度评分
        - emotion: 检测到的情绪
        - emotion_confidence: 情绪检测置信度
        
        Args:
            csv_path (str): CSV文件保存路径
            
        Returns:
            pd.DataFrame: 保存的数据DataFrame
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
                    'emotion': None,
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
                        'emotion': student['emotion'],
                        'pose_focus_score': student.get('pose_focus_score', 0.5),
                        'emotion_focus_score': student.get('emotion_focus_score', 0.5),
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
        计算专注度数据统计信息
        
        统计计算内容：
        1. 总帧数和检测到的总学生数
        2. 整体平均专注度
        3. 每个学生的平均专注度
        4. 情绪分布情况
        
        Returns:
            dict: 包含以下键的统计信息字典
                - total_frames: 总帧数
                - total_students: 检测到的学生总数
                - overall_avg_focus: 整体平均专注度
                - student_avg_focus: 每个学生的平均专注度字典
                - emotion_distribution: 情绪分布计数字典
        """
        if not self.focus_data:
            return {}
        
        # 计算每个学生的平均专注度
        student_focus = {}
        student_emotions = {}
        
        for frame_data in self.focus_data:
            for student in frame_data['students']:
                student_id = student['id']
                focus_score = student['focus_score']
                emotion = student.get('emotion', 'unknown')
                
                if student_id not in student_focus:
                    student_focus[student_id] = []
                    student_emotions[student_id] = {}
                    
                student_focus[student_id].append(focus_score)
                
                # 记录情绪出现次数
                if emotion not in student_emotions[student_id]:
                    student_emotions[student_id][emotion] = 0
                student_emotions[student_id][emotion] += 1
        
        # 计算平均值
        avg_focus = {}
        dominant_emotions = {}
        
        for student_id, scores in student_focus.items():
            avg_focus[student_id] = sum(scores) / len(scores)
            
            # 找出主要情绪
            emotions = student_emotions[student_id]
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "unknown"
            dominant_emotions[student_id] = dominant_emotion
        
        # 计算整体平均专注度
        all_scores = [score for scores in student_focus.values() for score in scores]
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # 统计所有情绪数据
        all_emotions = {}
        for student_emotions_dict in student_emotions.values():
            for emotion, count in student_emotions_dict.items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = 0
                all_emotions[emotion] += count
        
        return {
            'total_frames': self.frame_count,
            'total_students': len(student_focus),
            'student_avg_focus': avg_focus,
            'student_dominant_emotions': dominant_emotions,
            'overall_avg_focus': overall_avg,
            'overall_emotions': all_emotions
        }
        
    def _append_csv_data(self, frame_data, frame_count):
        """
        将单帧的专注度数据添加到内部数据列表
        
        转换frame_data格式为内部存储格式，便于后续生成CSV。
        每个学生的数据会单独记录一行，包含完整的帧号、时间戳等信息。
        
        Args:
            frame_data (dict): 当前帧的专注度数据
            frame_count (int): 帧计数器
            
        Returns:
            None: 数据直接添加到self.focus_data
        """
        # 实现与app.py中的流处理模式兼容
        self.focus_data.append(frame_data)