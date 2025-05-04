import os
os.environ['ULTRALYTICS_FONT_PATH'] = 'Arial.ttf'  # 设置字体路径环境变量
os.environ['ULTRALYTICS_SKIP_FONT_DOWNLOAD'] = '1'  # 跳过字体下载

import torch
import yaml
from ultralytics import YOLO  # 导入YOLO模型
from QtFusion.path import abs_path
import matplotlib
matplotlib.use('TkAgg')


def evaluate_model(model, data_path):
    # 模型评估
    results = model.val(data=data_path)
    
    # 打印评估指标
    metrics = results.results_dict
    print("\n=== 模型评估结果 ===")
    print(f"mAP@0.5: {metrics.get('metrics/mAP50', 0):.3f}")
    print(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95', 0):.3f}")
    print(f"精确率: {metrics.get('metrics/precision', 0):.3f}")
    print(f"召回率: {metrics.get('metrics/recall', 0):.3f}")
    
    return metrics

if __name__ == '__main__':  # 确保该模块被直接运行时才执行以下代码
    # directory = "D:\\vvss\\codeseg\\code1\\datasets\\data\\test\\yuan\\defect"
    # file_names = os.listdir(directory)
    # for file_name in file_names:
    #     print(file_name)
    workers = 1
    batch = 4  # 适当等修改Batchsize，根据电脑等显存/内存设置，如果爆显存可以调低
    device = "0" if torch.cuda.is_available() else "cpu"

    data_path = abs_path(f'datasets/data/data.yaml', path_type='current')  # 数据集的yaml的绝对路径

    unix_style_path = data_path.replace(os.sep, '/')
    # 获取目录路径
    directory_path = os.path.dirname(unix_style_path)
    
    # 读取YAML文件，保持原有顺序，使用UTF-8编码
    with open(data_path, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    # 修改path项
    if 'train' in data and 'valid' in data and 'test' in data:
        data['train'] = directory_path + '/train'
        data['valid'] = directory_path + '/valid'
        data['test'] = directory_path + '/test'

        # 将修改后的数据写回YAML文件，使用UTF-8编码
        with open(data_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    # 注意！不同模型大小不同，对设备等要求不同，如果要求较高的模型【报错】则换其他模型测试即可（例如yolov8-seg.yaml、yolov8-seg-goldyolo.yaml、yolov8-seg-C2f-Faster.yaml、yolov8-seg-C2f-DiverseBranchBlock.yaml、yolov8-seg-efficientViT.yaml等）
    # model = YOLO(r"D:\vvss\codeseg\50+种YOLOv8算法改进源码大全和调试加载训练教程（非必要）\改进YOLOv8模型配置文件\yolov8-seg-C2f-Faster.yaml").load("./weights/yolov8s-seg.pt")
    model = YOLO(model='./yolov8-seg.yaml', task='segment').load('./weights/yolov8s-seg.pt')  # 加载预训练的YOLOv8模型
    #model = YOLO(model='yolov8n.yaml', task='detect').load('yolov8n.pt') # 小模型，快速验证
    results = model.train(  # 开始训练模型
        data=data_path,  # 指定训练数据的配置文件路径
        device=device,  # 自动选择进行训练
        workers=workers,  # 指定使用2个工作进程加载数据
        imgsz=640,  # 指定输入图像的大小为640x640
        epochs=200,  # 增加训练轮次
        batch=batch,  # 指定每个批次的大小为8
        patience=50,  # 添加早停策略
        cos_lr=True,  # 使用余弦退火学习率
        augment=True,  # 启用更多数据增强
        mixup=0.1,    # 添加mixup增强
        mosaic=1.0,   # 使用mosaic增强
        save=True,  # 保存最佳模型
        save_period=10,  # 每10个epoch保存一次
        plots=True,  # 生成训练过程图
        val=True,  # 启用验证
        split=0.1,  # 从训练集中划分10%作为验证集
    )

    # 训练完成后进行模型评估
    print("\n开始模型评估...")
    metrics = evaluate_model(model, data_path)
    
    # 保存评估结果，使用UTF-8编码
    save_dir = os.path.join('runs', 'detect', 'train')
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 故障检测系统评估报告 ===\n")
        f.write(f"mAP@0.5: {metrics.get('metrics/mAP50', 0):.3f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95', 0):.3f}\n")
        f.write(f"精确率: {metrics.get('metrics/precision', 0):.3f}\n")
        f.write(f"召回率: {metrics.get('metrics/recall', 0):.3f}\n")
    
    print(f"\n评估结果已保存到: {os.path.join(save_dir, 'evaluation_results.txt')}")
