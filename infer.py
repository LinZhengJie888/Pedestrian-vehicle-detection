from ultralytics import YOLO
import os

# 禁用自动下载，避免网络请求和 SSL 错误
os.environ['ULTRALYTICS_DISABLE_DOWNLOAD'] = '1'

# 加载训练好的权重
model = YOLO("runs/detect/runs/detect/yolo11_vp2/weights/best.pt")

# 对图片进行推理（将 source处 替换为你自己的图片）
results = model.predict(
    source='test7.jpg',
    conf=0.25,           # 置信度阈值，低于此值的检测框会被过滤
    iou=0.45,            # NMS 去重阈值
    save=True,           # 保存标注结果图
    device='0',        # 指定设备为 CPU（可改为 0 表示使用第 0 号 GPU）
)

# 打印检测到的目标
for r in results:
    for box in r.boxes:
        cls  = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls]
        print(f"检测到：{name}，置信度：{conf:.2f}")