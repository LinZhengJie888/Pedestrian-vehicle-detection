from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_yolo():
    # 加载 YOLOv11 nano 预训练权重（首次运行自动下载，约 5MB）
    model = YOLO('best.pt')

    # 开始训练
    results = model.train(
        data='data.yaml',       # 数据集配置文件路径（相对/绝对都可）
        epochs=50,              # 训练轮次
        imgsz=416,              # 输入图像尺寸
        batch=4,                # 批次大小（根据显存调整，4060 8G 设 4 没问题）
        device='cuda',          # 使用 GPU（你的 RTX 4060）
        workers=2,              # Windows 下先设 0，避免多进程数据加载报错
        project='runs/detect',  # 项目保存目录
        name='yolo11_vp',       # 实验名称
        val=True,               # 训练时验证
    )
    print("训练完成! 权重保存于:", results.save_dir)

if __name__ == '__main__':
    train_yolo()
