import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import os
import json
import logging
from pathlib import Path
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fault_detection.log'),
        logging.StreamHandler()
    ]
)

class FaultDetector:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"使用设备: {self.device}")
        
        try:
            self.model = YOLO(model_path)
            logging.info(f"成功加载模型: {model_path}")
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            raise
        
        # 故障等级定义
        self.severity_levels = {
            "broken_wire": {"level": "严重", "color": (0, 0, 255)},      # 红色
            "insulator_damage": {"level": "严重", "color": (0, 0, 255)}, # 红色
            "corrosion": {"level": "中等", "color": (0, 165, 255)},      # 橙色
            "wire_wear": {"level": "中等", "color": (0, 165, 255)},      # 橙色
            "foreign_object": {"level": "轻微", "color": (0, 255, 0)},   # 绿色
            "vegetation": {"level": "轻微", "color": (0, 255, 0)}        # 绿色
        }
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 确保图像尺寸合适
        if max(image.shape[:2]) > 1920:  # 如果图像太大，进行缩放
            scale = 1920 / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale)
        return image
    
    def detect_faults(self, image_path):
        """故障检测主函数"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 预处理图像
        image = self.preprocess_image(image)
        
        # 进行检测
        results = self.model(image, conf=self.conf_thresh, iou=self.iou_thresh)[0]
        
        # 处理检测结果
        detections = []
        for box in results.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            bbox = box.xyxy[0].tolist()
            class_name = results.names[class_id]
            
            # 获取故障信息
            fault_info = self.severity_levels.get(class_name, {
                "level": "未知",
                "color": (128, 128, 128)  # 灰色
            })
            
            detection = {
                "fault_type": class_name,
                "confidence": conf,
                "bbox": bbox,
                "severity": fault_info["level"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "location": {
                    "x_center": (bbox[0] + bbox[2]) / 2,
                    "y_center": (bbox[1] + bbox[3]) / 2,
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1]
                }
            }
            detections.append(detection)
            
            # 在图像上绘制检测框
            cv2.rectangle(image, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), 
                        fault_info["color"], 2)
            
            # 添加标签
            label = f"{class_name}: {conf:.2f} ({fault_info['level']})"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_ymin = max(bbox[1], label_size[1] + 10)
            cv2.rectangle(image,
                        (int(bbox[0]), int(label_ymin - label_size[1] - 10)),
                        (int(bbox[0] + label_size[0]), int(label_ymin)),
                        fault_info["color"], -1)
            cv2.putText(image, label,
                       (int(bbox[0]), int(label_ymin - 7)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image, detections
    
    def save_results(self, image, detections, output_dir, original_filename):
        """保存检测结果"""
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 使用原始文件名作为基础
        base_name = Path(original_filename).stem
        
        # 保存标注后的图像
        image_path = output_dir / f"{base_name}_detection_{timestamp}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # 添加统计信息
        stats = {
            "total_faults": len(detections),
            "severity_count": {
                "严重": len([d for d in detections if d["severity"] == "严重"]),
                "中等": len([d for d in detections if d["severity"] == "中等"]),
                "轻微": len([d for d in detections if d["severity"] == "轻微"])
            },
            "fault_types": {}
        }
        
        for detection in detections:
            fault_type = detection["fault_type"]
            if fault_type not in stats["fault_types"]:
                stats["fault_types"][fault_type] = 0
            stats["fault_types"][fault_type] += 1
        
        # 保存检测结果
        result = {
            "image_info": {
                "filename": original_filename,
                "timestamp": timestamp,
                "resolution": image.shape[:2]
            },
            "detections": detections,
            "statistics": stats
        }
        
        result_path = output_dir / f"{base_name}_results_{timestamp}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        return image_path, result_path

def main():
    try:
        # 配置参数
        model_path = "runs/detect/train/weights/best.pt"
        image_dir = "test_images"
        output_dir = "detection_results"
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 初始化检测器
        detector = FaultDetector(model_path)
        logging.info("故障检测器初始化完成")
        
        # 确保输入目录存在
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"输入目录不存在: {image_dir}")
        
        # 处理目录中的所有图像
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            logging.warning(f"在 {image_dir} 中没有找到图像文件")
            return
        
        total_faults = 0
        processed_images = 0
        
        for image_name in image_files:
            image_path = os.path.join(image_dir, image_name)
            
            try:
                # 进行检测
                image, detections = detector.detect_faults(image_path)
                
                # 保存结果
                img_path, json_path = detector.save_results(
                    image, detections, output_dir, image_name
                )
                
                total_faults += len(detections)
                processed_images += 1
                
                logging.info(f"处理完成: {image_name}")
                logging.info(f"检测到 {len(detections)} 个故障")
                logging.info(f"结果保存在: {json_path}\n")
                
            except Exception as e:
                logging.error(f"处理 {image_name} 时出错: {str(e)}")
        
        # 输出总结
        if processed_images > 0:
            logging.info(f"\n检测总结:")
            logging.info(f"处理图像数: {processed_images}")
            logging.info(f"检测到的总故障数: {total_faults}")
            logging.info(f"平均每张图像故障数: {total_faults/processed_images:.2f}")
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 