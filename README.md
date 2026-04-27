# 🚗 疲劳驾驶检测系统

基于 MediaPipe + OpenCV + CNN 的实时疲劳驾驶检测系统，支持 Web 前端界面操作。

## 功能特点

- **实时人脸检测**：基于 MediaPipe Face Mesh 的 468 个面部关键点检测
- **眼睛状态分析**：通过 EAR（眼睛纵横比）检测闭眼状态
- **嘴巴状态分析**：通过 MAR（嘴巴纵横比）检测打哈欠
- **头部姿态估计**：通过 PnP 算法估计俯仰角/偏航角/翻滚角，检测低头
- **三种检测模式**：
  - 📏 规则模式：基于阈值判断
  - 🧠 深度学习模式：基于 CNN 模型分类
  - 🔀 混合模式：规则 + 深度学习结合
- **Web 前端界面**：浏览器操作，实时视频流 + 数据面板 + 系统日志
- **阈值可调**：前端实时调整 EAR/MAR/角度阈值和帧数
- **疲劳警告**：检测到疲劳时红色闪烁警告

## 项目结构

```
├── detectors/          # 检测器模块
│   ├── face_detector.py      # 人脸检测
│   ├── eye_analyzer.py       # 眼睛分析
│   ├── mouth_analyzer.py     # 嘴巴分析
│   └── head_pose_analyzer.py # 头部姿态分析
├── evaluators/         # 疲劳评估
│   └── fatigue_evaluator.py
├── classifiers/        # 深度学习分类器
│   └── dl_classifier.py
├── display/            # 显示渲染
│   └── renderer.py
├── models/             # 数据模型 & 训练模型
│   ├── data_models.py
│   └── trained/              # CNN 模型存放目录
├── training/           # 模型训练脚本
│   └── train_cnn.py
├── calibration/        # 阈值校准
│   └── threshold_calibrator.py
├── web/                # Web 前端
│   ├── templates/index.html
│   └── static/
├── tests/              # 测试用例
├── web_app.py          # Flask Web 服务
├── main.py             # 命令行入口
├── start.py            # 一键启动脚本
└── 启动检测系统.bat     # Windows 双击启动
```

## 快速开始

### 安装依赖

```bash
pip install flask opencv-python mediapipe numpy scikit-learn
```

如需使用深度学习模式，还需安装：
```bash
pip install tensorflow
```

### 启动系统

**方式一**：双击 `启动检测系统.bat`（Windows）

**方式二**：命令行运行
```bash
python start.py
```

浏览器会自动打开 `http://localhost:5000`

### 训练 CNN 模型

```bash
# 训练眼部模型
python training/train_cnn.py --dataset_type eye --dataset_path <数据集路径>

# 训练嘴部模型
python training/train_cnn.py --dataset_type mouth --dataset_path <数据集路径>
```

训练好的模型放到 `models/trained/` 目录下。

## 技术栈

- Python 3.8+
- MediaPipe（人脸关键点检测）
- OpenCV（图像处理）
- TensorFlow/Keras（CNN 模型）
- Flask（Web 服务）
- NumPy / scikit-learn
DMS系统是驾驶疲劳检测系统（Driver Monitor System），主要功能包括：疲劳检测、分心
检测、表情识别、危险动作识别、视线追踪等
最终yolo8训练好的pt文件存放在
/dms-driver-monitoring-system/workdir目录下
现在需求是扩展功能 在前端页面添加按钮，点击后可以加载yolo8模型进行检测，并在视频流中显示检测结果。
