# 🚗 疲劳驾驶检测系统

实时疲劳驾驶检测系统，支持 Web 前端界面操作。

## 功能特点

- **实时人脸检测**：基于 MediaPipe Face Mesh 的 468 个面部关键点检测
- **眼睛状态分析**：通过 EAR（眼睛纵横比）检测闭眼状态
- **嘴巴状态分析**：通过 MAR（嘴巴纵横比）检测打哈欠
- **头部姿态估计**：通过 PnP 算法估计俯仰角/偏航角/翻滚角，检测低头
- **检测模式**：
  - 📏 规则模式：基于阈值判断
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
├── dms-driver-monitoring-system/ # YOLOv8 训练与产物目录
│   ├── dms_yolov8_runnable_lowmem.py
│   └── workdir/final_model.pt
├── web_app.py          # Flask Web 服务
├── main.py             # 命令行入口
├── start.py            # 一键启动脚本
├── 规则模式详解.md        # 规则模式说明
└── 启动检测系统.bat      # Windows 双击启动
```

 <div class="rule-title">规则模式阈值提醒
                        <div class="rule-item" id="ruleEar">👁️ 闭眼：EAR < 0.20（连续48帧）</div>
                        <div class="rule-item" id="ruleMar">👄 哈欠：MAR > 0.75（连续25帧）</div>
                        <div class="rule-item" id="rulePitch">🧑 低头：|Pitch| > 25.0°（连续50帧）</div>
                        <div class="rule-note">头部姿态由 PnP 算法估计 Pitch / Yaw / Roll</div>

## 快速开始

### 安装依赖

```bash
pip install flask opencv-python mediapipe numpy scikit-learn
```

### 启动系统

**方式一**：双击 `启动检测系统.bat`（Windows）

**方式二**：命令行运行
```bash
python start.py
```

浏览器会自动打开 `http://localhost:5000`


## 技术栈

- Python 3.8+
- MediaPipe（人脸关键点检测）
- TensorFlow/Keras（CNN 模型）
- Flask（Web 服务）
- NumPy / scikit-learn
DMS系统是驾驶疲劳检测系统（Driver Monitor System），主要功能包括：疲劳检测、分心
检测、表情识别、危险动作识别、视线追踪等
最终yolo8训练好的pt文件存放在
/dms-driver-monitoring-system/workdir目录下
现在需求是扩展功能 在前端页面添加按钮，点击后可以加载yolo8模型进行检测，并在视频流中显示检测结果。


yolov8目前训练的分类有
open  eye
Closed Eye
Cigarette
Phone
Seatbelt

尽可能优化
yolov8启动以后   前端在遇到视频下面的情况请对应提示用户显示
DMS主要预警功能：

闭眼预警：闭眼检测分为两个等级：闭眼0.8秒触发1级预警，闭眼2秒触发2级预警
低头预警：检测到低头超过超过1.0秒，触发报警
打哈欠预警：检测到打哈欠超过0.8秒，触发报警
打电话预警：检测到驾驶员有打电话行为时，触发报警
抽烟预警：检测到驾驶员有抽烟行为时，触发报警
左顾右盼预警：检测到驾驶员视线偏移超过1.0秒，触发报警
遮挡镜头预警：通过画面检测功能发现一定范围内的黑屏，触发报警
驾驶座无人预警：检测到驾驶座位上没有驾驶员时触发报警


当前代码里 DMS 的具体阈值定义在 [`self._dms_thresholds`](web_app.py:67)：

- 闭眼 1 级：`0.8s`（[`eye_l1`](web_app.py:69)）
- 闭眼 2 级：`2.0s`（[`eye_l2`](web_app.py:70)）
- 低头：`1.0s`（[`head_down`](web_app.py:71)）
- 哈欠：`0.8s`（[`yawn`](web_app.py:72)）
- 左顾右盼：`1.0s`（[`look_away`](web_app.py:73)）
- 打电话：`0.3s`（[`phone`](web_app.py:74)）
- 抽烟：`0.3s`（[`smoke`](web_app.py:75)）
- 驾驶座无人：`1.0s`（[`no_driver`](web_app.py:76)）
- 遮挡镜头：`0.6s`（[`occlusion`](web_app.py:77)）
- 视线偏移角度（yaw 绝对值）：`25.0°`（[`yaw_abs`](web_app.py:78)）

遮挡镜头的画面统计阈值在 [`self._occlusion_cfg`](web_app.py:80)：
- 灰度均值 ≤ `35.0`（[`dark_mean`](web_app.py:81)）
- 灰度标准差 ≤ `12.0`（[`low_var_std`](web_app.py:82)）

这些阈值最终在 [`_build_dms_alerts()`](web_app.py:169) 中通过 [`_elapsed()`](web_app.py:162) 的持续时长比较触发告警。

规则模式下  基于阈值判断 具体的阈值  规则展示在前端提醒用户   优化显示
 **眼睛状态分析**：通过 EAR（眼睛纵横比）检测闭眼状态
- **嘴巴状态分析**：通过 MAR（嘴巴纵横比）检测打哈欠
- **头部姿态估计**：通过 PnP 算法估计俯仰角/偏航角/翻滚角，检测低头

系统日志中重要的日志 比如危险预警红字和黄字部分可以直接展示在显眼的部分