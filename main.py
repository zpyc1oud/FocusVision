import argparse
import os
import time
from focus_vision.focus_detector import FocusDetector

def main():
    """
    课堂专注度检测应用程序入口点
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="课堂专注度检测系统")
    parser.add_argument(
        "--video", "-v", 
        type=str, 
        required=True,
        help="输入视频文件路径"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="output/result.mp4",
        help="输出视频文件路径 (默认: output/result.mp4)"
    )
    parser.add_argument(
        "--csv", "-c", 
        type=str, 
        default="output/focus_data.csv",
        help="输出CSV数据文件路径 (默认: output/focus_data.csv)"
    )
    parser.add_argument(
        "--threshold", "-t", 
        type=float, 
        default=30.0,
        help="头部角度阈值，超过此角度视为不专注 (默认: 30度)"
    )
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default="yolo11n-pose.pt",
        help="YOLO模型路径 (默认: yolo11n-pose.pt)"
    )
    
    args = parser.parse_args()
    
    # 检查输入视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 输入视频文件不存在: {args.video}")
        return 1
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_dir = os.path.dirname(args.csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    print(f"正在初始化专注度检测器...")
    print(f"使用模型: {args.model}")
    print(f"专注度阈值: {args.threshold}度")
    
    # 创建专注度检测器
    detector = FocusDetector(
        model_path=args.model,
        focus_threshold=args.threshold
    )
    
    # 处理视频
    print(f"\n开始处理视频: {args.video}")
    print(f"输出视频将保存至: {args.output}")
    print(f"专注度数据将保存至: {args.csv}")
    
    start_time = time.time()
    
    try:
        stats = detector.process_video(
            video_path=args.video,
            output_path=args.output,
            csv_path=args.csv
        )
        
        # 打印统计信息
        print("\n处理完成!")
        print(f"总帧数: {stats['total_frames']}")
        print(f"检测到的学生数: {stats['total_students']}")
        print(f"整体平均专注度: {stats['overall_avg_focus']:.2f}")
        
        # 打印每个学生的平均专注度
        print("\n各学生平均专注度:")
        for student_id, avg_focus in stats['student_avg_focus'].items():
            print(f"  学生 {student_id}: {avg_focus:.2f}")
            
        print(f"\n处理时间: {time.time() - start_time:.2f}秒")
        return 0
        
    except Exception as e:
        print(f"处理视频时出错: {e}")
        return 1

# 示例用法
def example():
    """
    示例：如何使用FocusDetector类处理视频
    """
    # 创建专注度检测器
    detector = FocusDetector(model_path="yolov8x-pose.pt", focus_threshold=30)
    
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

if __name__ == "__main__":
    example()