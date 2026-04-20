基于yolo11模型，对数据集进行训练，得到一个关于车辆与行人检测的最佳参数

一、实验目的与要求

• 了解YOLO系列目标检测算法的发展历程，掌握YOLOv11的网络结构与核心改进

• 学习在Anaconda中创建独立虚拟环境，并完成Ultralytics框架的安装与配置

• 掌握通过Roboflow获取小规模标注数据集的方法，无需手工标注即可完成数据准备

• 能够在CPU/GPU环境下对YOLOv11进行微调训练，理解训练流程与超参数含义

• 学会使用训练好的权重对图片和视频进行推理，解读检测结果与评估指标

二、实验环境与配置

硬件环境

• CPU: AMD Ryzen 9 7945HX

• GPU: NVIDIA RTX 4060 Laptop GPU (8GB显存)

• 内存: 16GB

软件环境

• 操作系统: Windows 11

• Python: 3.10

• 深度学习框架: Ultralytics

• 包管理工具: Anaconda

三、实验步骤

1. 创建虚拟环境

conda create -n yolo11 python=3.10 -y
conda activate yolo11


2. 安装所需库

pip install ultralytics
pip install roboflow


3. 数据集准备

从Roboflow下载行人车辆检测数据集：
1. 访问https://universe.roboflow.com/search?q=vehicle+pedestrian+detection
2. 搜索"vehicle pedestrian detection"数据集
3. 选择YOLOv11格式下载
4. 下载完成后数据集结构如下：

dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml


4. 训练脚本

train.py
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_yolo():
    model = YOLO('best.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=416,
        batch=4,
        device='cuda',
        workers=2,
        project='runs/detect',
        name='yolo11_vp',
        val=True,
    )
    print("训练完成! 权重保存于:", results.save_dir)

if __name__ == '__main__':
    train_yolo()


5. 推理脚本

infer.py
from ultralytics import YOLO
import os

os.environ['ULTRALYTICS_DISABLE_DOWNLOAD'] = '1'

model = YOLO("runs/detect/runs/detect/yolo11_vp3/weights/best.pt")

results = model.predict(
    source='test5.jpg',
    conf=0.25,
    iou=0.45,
    save=True,
    device='0',
)

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls]
        print(f"检测到：{name}，置信度：{conf:.2f}")


四、数据处理与分析

训练指标说明

• mAP@0.5: IoU ≥ 0.5时的平均检测精度，反映目标检测的整体准确率

• mAP@0.5:0.95: 在IoU阈值0.5~0.95范围内取平均，评估更严格，反映检测的精确性和鲁棒性

四次训练结果对比

训练1：初始版本（行人+车辆二分类）

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 98 489 0.802 0.520

Pedestrian 72 235 0.696 0.360

Vehicle 87 254 0.909 0.679

分析: 二分类任务中，Vehicle类的检测精度明显高于Pedestrian类，整体mAP@0.5为0.802，验证了基础检测流程的有效性。

训练2：超参数优化（行人+车辆二分类）

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 98 489 0.812 0.533

Pedestrian 72 235 0.722 0.383

Vehicle 87 254 0.902 0.682

分析: 优化后整体mAP@0.5提升至0.812，Pedestrian类精度有显著提升，Vehicle类精度略有下降但mAP@0.5:0.95提升。

训练3：多类别扩展

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 125 227 0.449 0.322

Ambulance 25 32 0.599 0.510

Bus 15 23 0.524 0.396

Car 45 119 0.419 0.288

Motorcycle 21 23 0.489 0.273

Truck 19 30 0.213 0.145

分析: 扩展为5类后整体精度显著下降，Truck类因样本量最少（19张）精度最低，Ambulance类精度最高。

训练4：回归二分类并优化

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 98 489 0.833 0.538

Pedestrian 72 235 0.745 0.382

Vehicle 87 254 0.921 0.694

分析: 回归二分类并进行优化后，整体mAP@0.5提升至0.833，达到最高水平，Vehicle类mAP@0.5达到0.921。

训练汇总对比

训练阶段 任务类型 all mAP@0.5 all mAP@0.5:0.95

训练1 行人+车辆二分类 0.802 0.520

训练2 行人+车辆二分类（优化） 0.812 0.533

训练3 多类别（5类） 0.449 0.322

训练4 行人+车辆（最终优化） 0.833 0.538

五、实验结果与验证

推理示例

!test7.jpg
图：推理结果示例（深蓝框为车辆，浅蓝框为行人）

GPU与CPU性能对比

• GPU训练: RTX 4060训练50轮约1-2小时

• CPU训练: AMD Ryzen 9 7945HX训练1轮时间约GPU的3倍

• 推理速度: GPU实时推理，满足实际应用需求

六、实验结论

1. 硬件性能对比: 实验验证了GPU在深度学习训练中的显著优势。RTX 4060 GPU训练50轮仅需1-2小时，而相同任务在AMD Ryzen 9 7945HX CPU上训练1轮时间约为GPU的3倍，体现了GPU并行计算的强大性能。

2. 二分类任务优化效果: 通过四次迭代训练，最终在行人+车辆二分类任务上取得了最优结果，整体mAP@0.5达到0.833，mAP@0.5:0.95达到0.538，验证了超参数优化和训练策略调整的有效性。

3. 多类别检测挑战: 扩展为5类目标检测后，整体精度显著下降（mAP@0.5降至0.449），其中Truck类因样本量不足（仅19张图像）精度最低（mAP@0.5=0.213），体现了数据集平衡对多类别检测的重要性。

4. 类别特异性差异: Vehicle类始终表现最优（mAP@0.5最高0.921），说明模型对规则、高对比度目标的识别能力更强；而Pedestrian类识别精度相对较低，可能与行人姿态多样、遮挡频繁等因素有关。

5. 数据集质量影响: 实验中发现数据集图像较模糊且为灰度图像，这可能是导致行人检测精度不高的原因之一。高质量的数据集是提升模型性能的基础。

6. 实际应用可行性: 在RTX 4060消费级GPU上能够完成中等规模目标检测模型的训练和推理，推理速度满足实时性要求，为实际部署提供了可行性验证。

七、参考文献

1. Ultralytics YOLOv11 Documentation
2. Roboflow Dataset Platform
3. J. Redmon et al. "You Only Look Once: Unified, Real-Time Object Detection"

八、GitHub仓库

项目地址：https://github.com/LinZhengJie888/Pedestrian-vehicle-detection.git

实验人: LinZhengJie  
完成时间: 2024年12月  
实验环境: Windows 11 + RTX 4060 Laptop GPU


