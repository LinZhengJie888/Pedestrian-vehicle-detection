基于yolo11模型，对数据集进行训练，得到一个关于车辆与行人检测的最佳参数

📁 实验概览

🎯 实验目的

1. 了解YOLO系列目标检测算法的发展历程，掌握YOLOv11的网络结构与核心改进。
2. 学习在Anaconda中创建独立虚拟环境，并完成Ultralytics框架的安装与配置。
3. 掌握通过Roboflow获取小规模标注数据集的方法，无需手工标注即可完成数据准备。
4. 能够在CPU环境下对YOLOv11n进行轻量级微调训练，理解训练流程与超参数含义。
5. 学会使用训练好的权重对图片和视频进行推理，解读检测结果与评估指标。

⚙️ 环境配置

1. 创建并激活虚拟环境
    conda create -n yolo11 python=3.10 -y
    conda activate yolo11
    

2. 安装Ultralytics
    pip install ultralytics
    
    安装后验证：
    python -c "from ultralytics import YOLO; print('安装成功')"
    

3. 安装Roboflow SDK（用于下载数据集）
    pip install roboflow
    

📂 数据集准备

数据集来源于Roboflow公开的小型数据集（约500张，含行人与车辆标注），可直接用于训练。

下载步骤：
1. 访问 https://universe.roboflow.com/search?q=vehicle+pedestrian+detection，注册并登录免费账号。
2. 搜索“vehicle pedestrian detection”，选择一个YOLO格式（YOLOv11）的公开数据集，点击Download。
3. 格式选择YOLOv11，下载方式选择“show download code”，复制生成的Python代码片段。
4. 将代码粘贴到download_data.py并运行，数据集会自动下载到本地并生成data.yaml。

目录结构（示例）：

数据集文件夹/

 ├── train/
 
 ├── val/
 
 ├── test/
 
 └── data.yaml


🚀 训练与推理代码

1. 训练脚本 train.py

from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_yolo():
    # 加载YOLOv11 nano预训练权重（首次运行自动下载，约5MB）
    model = YOLO('yolo11n.pt')

    # 开始训练
    results = model.train(
        data='data.yaml',        # 数据集配置文件路径
        epochs=50,                # 训练轮次
        imgsz=416,                # 输入图像尺寸
        batch=4,                  # 批次大小（根据显存调整）
        device='cuda',            # 使用GPU（如RTX 4060）
        workers=2,                # Windows下可先设为0避免多进程报错
        project='runs/detect',    # 项目保存目录
        name='yolo11_vp',         # 实验名称
        val=True                  # 训练时验证
    )
    print("训练完成！权重保存于：", results.save_dir)

if __name__ == '__main__':
    train_yolo()


2. 推理脚本 infer.py

from ultralytics import YOLO
import os

os.environ['ULTRALYTICS_DISABLE_DOWNLOAD'] = '1'

model = YOLO("runs/detect/yolo11_vp/weights/best.pt")

results = model.predict(
    source='test.jpg',
    conf=0.5,          # 置信度阈值
    iou=0.45,          # NMS去重阈值
    save=True,         # 保存标注结果图
    device='0'         # 指定设备为GPU（0表示第0号GPU，可改为'cpu'）
)

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls]
        print(f"检测到: {name}, 置信度: {conf:.2f}")


📈 实验结果与分析

实验共进行四次迭代训练，具体结果如下：

训练1：初始版本（行人+车辆二分类）

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 98 489 0.802 0.520

Pedestrian 72 235 0.696 0.360

Vehicle 87 254 0.909 0.679

分析：车辆（Vehicle）类的检测精度高于行人（Pedestrian），验证了基础检测流程的有效性。

训练2：超参数优化（行人+车辆二分类）

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 98 489 0.812 0.533

Pedestrian 72 235 0.722 0.383

Vehicle 87 254 0.902 0.682

分析：优化后整体mAP@0.5从0.802提升至0.812，Pedestrian类提升较明显。

训练3：多类别扩展（含救护车、公交车、汽车、摩托车、卡车）

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 125 227 0.449 0.322

Ambulance 25 32 0.599 0.510

Bus 15 23 0.524 0.396

Car 45 119 0.419 0.288

Motorcycle 21 23 0.489 0.273

Truck 19 30 0.213 0.145

分析：多类别任务难度显著增加，整体mAP下降；Ambulance类精度最高，Truck类最低（样本量不足）。

训练4：多类别+行人车辆（混合任务，进一步优化）

类别 图像数 实例数 mAP@0.5 mAP@0.5:0.95

all 98 489 0.833 0.538

Pedestrian 72 235 0.745 0.382

Vehicle 87 254 0.921 0.694

分析：回归二分类并进一步优化后，整体和各类别的mAP均达到最高，验证了迭代优化的有效性。

汇总对比

训练阶段 任务类型 all mAP@0.5 all mAP@0.5:0.95

训练1 行人+车辆二分类 0.802 0.520

训练2 行人+车辆二分类（优化） 0.812 0.533

训练3 多类别（5类） 0.449 0.322

训练4 行人+车辆（混合/优化） 0.833 0.538

✅ 实验结论

1. 在配备RTX 4060 Laptop GPU的硬件环境下，通过四次迭代训练验证了YOLOv11模型对不同复杂度目标检测任务的适应能力。
2. 在二分类任务中，Vehicle类始终表现出最高的检测精度（mAP@0.5最高可达0.921），验证了模型对规则、高对比度目标的稳定识别能力。
3. 更换为多类别数据集后，整体mAP@0.5从0.802下降至0.449，其中Truck类因样本量不足导致mAP@0.5仅为0.213，体现了数据集类别平衡对模型性能的关键影响。
4. 通过“基础训练 → 超参数调优 → 二次优化”的迭代过程，最终模型的整体mAP@0.5提升至0.833，mAP@0.5:0.95提升至0.538，证明了在固定硬件条件下通过训练策略优化可以有效挖掘模型潜力。
5. 推理结果示例显示检测框能够准确覆盖车辆目标（置信度0.83），但行人识别存在偏差，分析可能源于数据集中图像模糊、全为灰度图，目标不够清晰。
6. 实验验证了在主流消费级GPU硬件上进行中等规模目标检测模型训练的可行性，推理速度满足实时性要求，为同类应用场景下的模型部署提供了实践参考。





