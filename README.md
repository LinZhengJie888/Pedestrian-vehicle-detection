# Pedestrian-vehicle-detection-
基于yolo11模型，对数据集进行训练，得到一个关于车辆与行人检测的最佳参数

% ============================================================
%   一、实验目的与要求
% ============================================================
\begin{labbox}{一、实验目的与要求：}{}

\begin{itemize}[leftmargin=2em, itemsep=0.3em]
    \item 了解 YOLO 系列目标检测算法的发展历程，掌握 YOLOv11 的网络结构与核心改进。
    \item 学习在 Anaconda 中创建独立虚拟环境，并完成 Ultralytics 框架的安装与配置。
    \item 掌握通过 Roboflow 获取小规模标注数据集的方法，无需手工标注即可完成数据准备。
    \item 能够在 CPU 环境下对 YOLOv11n 进行轻量级微调训练，理解训练流程与超参数含义。
    \item 学会使用训练好的权重对图片和视频进行推理，解读检测结果与评估指标。
\end{itemize}

\end{labbox}


% ============================================================
%   二、实验方法与步骤
% ============================================================
\begin{labbox}{二、实验方法与步骤：}{}

\begin{enumerate}[leftmargin=2em, itemsep=0.5em]

    \item \textbf{创建并激活虚拟环境}

    打开 Anaconda Prompt，依次执行：
\begin{lstlisting}
conda create -n yolo11 python=3.10 -y
conda activate yolo11
\end{lstlisting}

    \item \textbf{安装 Ultralytics}
\begin{lstlisting}
pip install ultralytics
\end{lstlisting}

    安装完成后验证：
\begin{lstlisting}
python -c "from ultralytics import YOLO; print('安装成功')"
\end{lstlisting}

    \item \textbf{从 Roboflow 下载数据集}

    本实验使用 Roboflow 上的公开小型数据集（约 500 张，含行人与车辆标注），
    无需 GPU，下载后即可直接用于训练。步骤如下：
\begin{center}
    \includegraphics[width=0.8\linewidth]{image1.png}  % ← 替换文件名
    \par\vspace{0.2em}
    {\small 图1\quad Roboflow 下载数据集}
\end{center}
\begin{center}
    \includegraphics[width=0.8\linewidth]{image2.png}  % ← 替换文件名
    \par\vspace{0.2em}
    {\small 图2\quad YOLOv11格式}
\end{center}
\begin{center}
    \includegraphics[width=0.8\linewidth]{image3.png}  % ← 替换文件名
    \par\vspace{0.2em}
    {\small 图3\quad 下载好啦}
\end{center}

    \begin{enumerate}[label=\alph*., leftmargin=2em, itemsep=0.2em]
        \item 访问 \url{https://universe.roboflow.com/search?q=vehicle+pedestrian+detection}，注册免费账号并登录。
        \item 在搜索框中搜索 \textbf{"vehicle pedestrian detection"}，
              选择一个 YOLO 格式（YOLOv11）的公开数据集，点击 Download。
        \item 格式选择 \textbf{YOLOv11}，下载方式选择 \textbf{show download code}，
              复制生成的 Python 代码片段。
        \item 安装 Roboflow SDK 并运行下载代码：

\begin{lstlisting}
pip install roboflow
\end{lstlisting}
        \item 将复制的代码粘贴到 \texttt{download\_data.py} 并运行，
              数据集会自动下载到本地并生成 \texttt{data.yaml}。
    \end{enumerate}
\begin{center}
    \includegraphics[width=0.8\linewidth]{image4.png}  % ← 替换文件名
    \par\vspace{0.2em}
    {\small 图4\quad 下载好的数据集目录结构}
\end{center}
    \item \textbf{确认数据集目录结构}

    下载完成后目录结构如下（路径以实际下载位置为准）：
\begin{lstlisting}
dataset/
  train/
    images/    # 训练图片
    labels/    # 对应 YOLO 格式标注 (.txt)
  valid/
    images/    # 验证图片
    labels/
  data.yaml    # 自动生成，包含路径和类别信息
\end{lstlisting}

    \item \textbf{编写并运行训练脚本}

    新建 \texttt{train.py}，内容见第三节，然后运行：
\begin{lstlisting}
python train.py
\end{lstlisting}

    \textbf{注意：} 无 GPU 的电脑训练 10 个 epoch 约需 30--60 分钟，请耐心等待。

    \item \textbf{推理验证}

    训练完成后运行推理脚本：
\begin{lstlisting}
python infer.py
\end{lstlisting}

    检测结果图片保存在 \texttt{runs/detect/} 目录下。

\end{enumerate}

\end{labbox}

% ============================================================
%   三、实验过程及内容
% ============================================================
\begin{labbox}{三、实验过程及内容：}{}
 
\textbf{（1）训练脚本 \texttt{train.py}}
 
\begin{lstlisting}
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 设置环境变量，避免某些系统库冲突

def train_yolo():
    # 加载 YOLOv11 nano 预训练权重（首次运行自动下载，约 5MB）
    model = YOLO('best.pt') 

    # 开始训练
    results = model.train(
        data='data.yaml',       # 数据集配置文件路径（相对/绝对都可）
        epochs=50,              # 训练轮次
        imgsz=416,              # 输入图像尺寸
        batch=4,                # 批次大小（根据自身显存调整，本机采用 4060 8G 设 4 没问题）
        device='cuda',          # 使用 GPU（RTX 4060）
        workers=2,              # Windows 下可先设 0，避免多进程数据加载报错
        project='runs/detect',  # 项目保存目录
        name='yolo11_vp',       # 实验名称
        val=True,               # 训练时验证
    )
    print("训练完成! 权重保存于:", results.save_dir)

if __name__ == '__main__':
    train_yolo()
\end{lstlisting}
 
\textbf{（2）推理脚本 \texttt{infer.py}}
 
\begin{lstlisting}
from ultralytics import YOLO
import os

# 禁用自动下载，避免网络请求和 SSL 错误
os.environ['ULTRALYTICS_DISABLE_DOWNLOAD'] = '1'

# 加载训练好的权重
model = YOLO("runs/detect/runs/detect/yolo11_vp3/weights/best.pt")

# 对图片进行推理（将 source处 替换为你自己的图片）
results = model.predict(
    source='test5.jpg',
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
\end{lstlisting}
 
\textbf{（3）YOLOv11 核心改进简介}
 
与 YOLOv8 相比，YOLOv11 主要改进如下：
\begin{itemize}[leftmargin=2em, itemsep=0.2em]
    \item \textbf{C3k2 模块}：改进跨阶段部分网络，在保持精度的前提下减少约 22\% 参数量，更适合 CPU 推理。
    \item \textbf{C2PSA 注意力机制}：在颈部引入位置敏感注意力，增强对小目标（如远距离行人）的特征提取能力。
    \item \textbf{统一任务接口}：检测、分割、姿态估计共用同一套训练代码，切换任务只需更换模型文件。
\end{itemize}
 
\vspace{0.3cm}
\textbf{（4）插入截图说明}
 
% ============================================================
% 【如何插入图片】
%
% 第一步：在 Overleaf 左侧文件栏点击上传按钮，上传你的截图
%         （支持 .jpg / .png / .pdf 格式）
%
% 第二步：把下面的 result.png 替换为你上传的文件名
%
% 注意：方框内不能用 \begin{figure}，直接用下面的写法即可
% ============================================================
 
% 单张图片（居中，宽度为方框宽度的 80%）：
\begin{center}
    \includegraphics[width=0.8\linewidth]{test7.jpg}  % ← 替换文件名
    \par\vspace{0.2em}
    {\small 图1\quad 推理结果示例（深蓝框为车辆，浅蓝框为行人）}
\end{center}
 
% 两张图并排（各占 48%）：
% \begin{center}
%     \includegraphics[width=0.48\linewidth]{before.png}
%     \hfill
%     \includegraphics[width=0.48\linewidth]{after.png}
%     \par\vspace{0.2em}
%     {\small 图2\quad 训练前（左）与训练后（右）对比}
% \end{center}
 
\end{labbox}
 
 
% ============================================================
%   四、数据处理与分析
% ============================================================
\begin{labbox}{四、数据处理与分析：}{}

本次实验，我一共采取了四次迭代训练，针对YOLOv11四次迭代训练（前两次为行人+车辆二分类，第三次更换为多类别数据集），结合训练日志的最终结果，分阶段分析训练效果：

\vspace{0.2cm}
\textbf{各指标含义：}
\begin{itemize}[leftmargin=2em, itemsep=0.2em]
    \item \textbf{mAP@0.5}：IoU $\geq$ 0.5 时的平均检测精度，反映目标检测的整体准确率。
    \item \textbf{mAP@0.5:0.95}：在 IoU 阈值0.5~0.95范围内取平均，评估更严格，反映检测的精确性和鲁棒性。
\end{itemize}

\vspace{0.3cm}
\textbf{训练1：初始版本（行人+车辆二分类）}
\begin{center}
\begin{tabular}{|c|c|c|c|c|}
    \hline
    \textbf{类别} & \textbf{图像数} & \textbf{实例数} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} \\
    \hline
    all      & 98  & 489 & 0.802 & 0.52  \\
    Pedestrian & 72  & 235 & 0.696 & 0.36  \\
    Vehicle  & 87  & 254 & 0.909 & 0.679 \\
    \hline
\end{tabular}
\end{center}
\textbf{分析：} 二分类任务中，Vehicle类的mAP@0.5（0.909）和mAP@0.5:0.95（0.679）均高于整体水平和Pedestrian类，说明车辆目标的检测精度更高；整体mAP@0.5为0.802，验证了基础检测流程的有效性。


\vspace{0.3cm}
\textbf{训练2：超参数优化（行人+车辆二分类）}
\begin{center}
\begin{tabular}{|c|c|c|c|c|}
    \hline
    \textbf{类别} & \textbf{图像数} & \textbf{实例数} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} \\
    \hline
    all      & 98  & 489 & 0.812 & 0.533 \\
    Pedestrian & 72  & 235 & 0.722 & 0.383 \\
    Vehicle  & 87  & 254 & 0.902 & 0.682 \\
    \hline
\end{tabular}
\end{center}
\textbf{分析：} 优化后，整体mAP@0.5从0.802提升至0.812；Pedestrian类的mAP@0.5从0.696提升至0.722，mAP@0.5:0.95从0.36提升至0.383；Vehicle类的mAP@0.5略有下降（0.909→0.902），但mAP@0.5:0.95从0.679提升至0.682，整体二分类性能略有提升。


\vspace{0.3cm}
\textbf{训练3：多类别扩展（含救护车、公交车、汽车、摩托车、卡车）}
\begin{center}
\begin{tabular}{|c|c|c|c|c|}
    \hline
    \textbf{类别} & \textbf{图像数} & \textbf{实例数} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} \\
    \hline
    all      & 125 & 227 & 0.449 & 0.322 \\
    Ambulance & 25  & 32  & 0.599 & 0.51  \\
    Bus      & 15  & 23  & 0.524 & 0.396 \\
    Car      & 45  & 119 & 0.419 & 0.288 \\
    Motorcycle & 21 & 23  & 0.489 & 0.273 \\
    Truck    & 19  & 30  & 0.213 & 0.145 \\
    \hline
\end{tabular}
\end{center}
\textbf{分析：} 更换为多类别数据集后，整体mAP@0.5（0.449）和mAP@0.5:0.95（0.322）显著低于二分类任务，说明多类别检测的难度更高；Ambulance类的mAP@0.5（0.599）和mAP@0.5:0.95（0.51）在多类别中最高，Truck类的mAP@0.5（0.213）和mAP@0.5:0.95（0.145）最低，类别间检测精度差异明显。


\vspace{0.3cm}
\textbf{训练4：多类别+行人车辆（混合任务）}
\begin{center}
\begin{tabular}{|c|c|c|c|c|}
    \hline
    \textbf{类别} & \textbf{图像数} & \textbf{实例数} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} \\
    \hline
    all      & 98  & 489 & 0.833 & 0.538 \\
    Pedestrian & 72  & 235 & 0.745 & 0.382 \\
    Vehicle  & 87  & 254 & 0.921 & 0.694 \\
    \hline
\end{tabular}
\end{center}
\textbf{分析：} 混合任务（回归二分类并优化）中，整体mAP@0.5（0.833）高于训练1和训练2；Pedestrian类mAP@0.5（0.745）和mAP@0.5:0.95（0.382）均有提升；Vehicle类mAP@0.5（0.921）和mAP@0.5:0.95（0.694）显著提升，整体检测性能优于前两次二分类训练。


\vspace{0.3cm}
\textbf{汇总对比（四类训练最终结果）}
\begin{center}
\begin{tabular}{|c|c|c|c|c|}
    \hline
    \textbf{训练阶段} & \textbf{任务类型} & \textbf{all mAP@0.5} & \textbf{all mAP@0.5:0.95} \\
    \hline
    训练1 & 行人+车辆二分类 & 0.802 & 0.52  \\
    训练2 & 行人+车辆二分类（优化） & 0.812 & 0.533 \\
    训练3 & 多类别（5类） & 0.449 & 0.322\\
    训练4 & 行人+车辆（混合/优化） & 0.833 & 0.538 \\
    \hline
\end{tabular}
\end{center}
\textbf{分析：} 二分类任务（训练1、2、4）的整体mAP@0.5均高于多类别任务（训练3），说明类别数量增加会降低检测难度上限；训练4在二分类基础上进一步优化，整体和Vehicle类、Pedestrian类的mAP均有提升，验证了迭代优化的有效性；训练3的多类别任务中，不同类别检测精度差异大，Truck类精度最低，Ambulance类最高。

\end{labbox}
 
 
% ============================================================
%   五、实验结论
% ============================================================
\begin{labbox}{五、实验结论：}{}
    \textbf{注：} github链接：https://github.com/LinZhengJie888/Pedestrian-vehicle-detection.git
\begin{enumerate}[leftmargin=2em, itemsep=0.4em]
    \item 本实验在配备 RTX 4060 Laptop GPU 的硬件环境下，通过四次迭代训练验证了 YOLOv11 模型对不同复杂度目标检测任务的适应能力，完成了从二分类到多类别的系统性实验探索，在未正式开始实验时通过CPU（AMD Ryzen 9 7945HX）对相同的数据集进行一轮训练的时间大概是采用GPU的三倍左右，这一对比实验深刻的反映了GPU在训练数据上的优势之处。

    \item 实验结果表明，在二分类任务（行人+车辆）中，Vehicle 类始终表现出最高的检测精度，mAP@0.5 最高可达 0.921，验证了模型对规则、高对比度目标的稳定识别能力。

    \item 更换为多类别数据集后，整体 mAP@0.5 从 0.802 下降至 0.449，其中 Truck 类因样本量不足导致 mAP@0.5 仅为 0.213，体现了数据集类别平衡对模型性能的关键影响。

    \item 通过“基础训练—超参数调优—二次优化”的迭代过程，最终模型的整体 mAP@0.5 提升至 0.833，mAP@0.5:0.95 提升至 0.538，证明了在固定硬件条件下通过训练策略优化可以有效挖掘模型潜力。

    \item 结合推理结果示例（车辆检测框标注“Vehicle 0.83”），验证了模型在真实场景中的实际应用效果：检测框能够准确覆盖目标，置信度较高，背景干扰未对车辆检测产生显著影响，但同时图中的行人并未精准的识别，通过不断地试验和查询资料，最终确定了导致这一实验偏差的原因可能在于选取的数据集并不太合适，通过观察数据集发现，数据集的图像比较模糊，且全为灰度图像，数据集中的行人和车辆并不明确清晰，这可能就是导致最后实验推理结果不太符合预期的原因所在。

    \item 实验验证了在主流消费级 GPU 硬件上进行中等规模目标检测模型训练的可行性，推理速度满足实时性要求，为同类应用场景下的模型部署提供了实践参考。

\end{enumerate}

\end{labbox}
