# 疲劳驾驶检测系统实现详解（关键代码分析 + 数学公式）

本文档面向开发者，详细说明本项目从视频输入到告警输出的关键实现路径、核心公式、阈值规则与工程设计要点。

---

## 1. 系统总体架构

系统主流程由 Web 服务驱动，核心在后台循环中完成：

1. 读取摄像头帧
2. MediaPipe 提取人脸关键点
3. 计算 EAR / MAR / 头姿欧拉角
4. 规则引擎判断疲劳与原因
5. 可选 YOLOv8 叠加做 DMS 告警
6. 渲染图像并通过 MJPEG 持续推流
7. 前端轮询实时数值 + 状态 + 日志

主入口与处理循环位于：

- `WebDetectionSystem`：`web_app.py`
- `_process_loop()`：持续推理主循环
- `/video_feed`：MJPEG 视频流接口
- `/api/data`：实时状态数据接口

---

## 2. 数据模型（模块之间如何传递数据）

统一的数据结构定义在 `models/data_models.py`，核心对象：

- `FaceLandmarks`：关键点容器（眼、嘴、头姿点、全量点）
- `EyeResult`：EAR 与闭眼判定
- `MouthResult`：MAR 与哈欠判定
- `PoseResult`：pitch/yaw/roll 与低头判定
- `FatigueStatus`：最终疲劳结论与原因列表

这种分层好处是：

- 检测层与业务层解耦
- 前端接口稳定
- 单元测试容易做（可直接构造结果对象注入）

---

## 3. 人脸关键点检测（MediaPipe Face Mesh）

实现文件：`detectors/face_detector.py`

### 3.1 核心调用

- 初始化 FaceMesh：`FaceDetector.__init__`
- 单帧推理：`self._face_mesh.process(rgb_frame)`

### 3.2 关键点坐标变换

MediaPipe 输出为归一化坐标 `(x, y)`（相对于图像宽高），代码中转为像素坐标：

\[
(x_{px}, y_{px}) = (x \cdot W,\; y \cdot H)
\]

这样后续几何计算（距离、PnP）都在像素空间完成。

### 3.3 与业务计算直接相关的关键点索引

- 眼睛：`LEFT_EYE_INDICES` / `RIGHT_EYE_INDICES`
- 嘴巴：`MOUTH_INDICES`
- 头姿：`HEAD_POSE_INDICES`

这些索引来自 Face Mesh 的点位语义映射，避免处理全量点时的额外开销。

---

## 4. EAR（眼睛纵横比）计算与闭眼检测

实现文件：`detectors/eye_analyzer.py`

### 4.1 EAR 定义

对于单眼 6 个点 \(p_1...p_6\)：

\[
EAR = \frac{\|p_2-p_6\| + \|p_3-p_5\|}{2\cdot\|p_1-p_4\|}
\]

其中：

- 分子是两条竖向开合距离
- 分母是水平眼裂距离

### 4.2 双眼融合

左右眼分别计算后取平均：

\[
EAR_{avg} = \frac{EAR_{left}+EAR_{right}}{2}
\]

### 4.3 规则判定

- 闭眼瞬时条件：`EAR_avg < ear_threshold`
- 连续帧计数：闭眼则 `counter += 1`，否则清零
- 疲劳触发：`counter >= consec_frames`

这是一种经典的“时序去抖”策略，可抑制眨眼造成的误报。

---

## 5. MAR（嘴巴纵横比）计算与哈欠检测

实现文件：`detectors/mouth_analyzer.py`

### 5.1 MAR 定义

设关键点：上唇 `upper`、下唇 `lower`、嘴角 `left/right`、内唇 `upper_inner/lower_inner`。

\[
MAR = \frac{\|upper\_inner-lower\_inner\| + \|upper-lower\|}{2\cdot\|left-right\|}
\]

### 5.2 规则判定

- 哈欠瞬时条件：`MAR > mar_threshold`
- 连续帧策略与 EAR 相同
- 触发条件：`counter >= consec_frames`

---

## 6. 头部姿态估计（PnP）与低头检测

实现文件：`detectors/head_pose_analyzer.py`

### 6.1 问题建模

通过 6 个 2D 人脸点与标准 3D 人脸模型点建立 PnP：

\[
s\begin{bmatrix}u\\v\\1\end{bmatrix}
= K\,[R|t]\,\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
\]

其中：

- \((X,Y,Z)\)：3D 模型点
- \((u,v)\)：图像点
- \(K\)：相机内参矩阵
- \(R,t\)：旋转与平移
- \(s\)：尺度因子

代码中使用 `cv2.solvePnP(..., SOLVEPNP_ITERATIVE)` 求解。

### 6.2 相机内参近似

本项目采用简单近似：

\[
f = \max(H,W),\quad c_x=W/2,\quad c_y=H/2
\]

\[
K=\begin{bmatrix}
f&0&c_x\\
0&f&c_y\\
0&0&1
\end{bmatrix}
\]

畸变系数设为 0（`dist_coeffs = 0`），工程上可工作，但精度受相机标定质量影响。

### 6.3 旋转表示转换

- `solvePnP` 输出旋转向量 `rvec`
- `cv2.Rodrigues(rvec)` 转旋转矩阵 `R`
- 再由 `R` 分解欧拉角（pitch/yaw/roll）

代码采用 ZYX 分解并做奇异位姿分支保护。

### 6.4 低头判定

\[
|pitch| > pitch\_threshold
\]

同样结合连续帧计数避免瞬时抖动误报。

---

## 7. 综合疲劳决策（规则 / DL / 混合）

实现文件：`evaluators/fatigue_evaluator.py`

### 7.1 模式说明

- `rule`：仅使用几何规则（闭眼/哈欠/低头）
- `dl`：仅使用分类器结果（若 `dl_result` 缺失则回退规则）
- `hybrid`：规则与 DL 做 OR 融合

### 7.2 规则模式逻辑

若以下任一为真则疲劳：

- `eye_result.is_fatigued`
- `mouth_result.is_fatigued`
- `pose_result.is_fatigued`

输出 `reasons` 供前端和日志展示。

---

## 8. YOLOv8 叠加 DMS 告警机制

实现文件：`web_app.py`

### 8.1 启停与懒加载

- `/api/yolo` 调用 `set_yolo_enabled()`
- 首次启用时 `_load_yolo_model()` 懒加载 `.pt`

### 8.2 融合思路

主循环中当 YOLO 开启：

1. `yolo_model.predict(...)` 得到检测框类别
2. 组装 `yolo_labels`
3. 调用 `_build_dms_alerts(...)`
4. 写入 `dms_alerts` 与 `is_dms_alert`

### 8.3 时间阈值触发（核心）

每类风险维护一个 `*_since` 时间戳：

- active 且首次出现：记录开始时间
- active 结束：清空时间戳
- 通过 `elapsed = now - since` 比较阈值秒数触发告警

例如：

\[
\text{if } elapsed_{eye\_closed} \ge 2.0s \Rightarrow \text{闭眼二级预警}
\]

相比纯帧计数，这种秒级方式对不同 FPS 更鲁棒。

### 8.4 遮挡镜头近似判定

当无 YOLO 对应类时，使用灰度统计：

- 均值：\(\mu = mean(I)\)
- 标准差：\(\sigma = std(I)\)

满足：

\[
\mu \le dark\_mean \quad \text{且} \quad \sigma \le low\_var\_std
\]

判定为“黑屏/遮挡”近似成立。

---

## 9. 连续帧阈值与时间关系（实践中经常被问）

对于 EAR / MAR / 低头这类“连续帧触发”规则：

\[
T \approx \frac{N}{FPS}
\]

- \(N\)：连续帧阈值
- \(FPS\)：实时帧率
- \(T\)：对应持续时间

示例：

- `N=48`，`FPS=24` 时约 `2.0s`
- `N=48`，`FPS=30` 时约 `1.6s`

因此若设备帧率变化明显，推荐改为“秒级阈值”或做 FPS 自适应。

---

## 10. 并发与接口设计

实现文件：`web_app.py`

### 10.1 线程模型

- 主 Flask 线程：处理 HTTP API
- 后台检测线程：`_process_loop`
- 共享状态：`_latest_frame` / `_latest_data`，使用 `threading.Lock` 保护

### 10.2 视频推流

`/video_feed` 使用 `multipart/x-mixed-replace` 输出 JPEG 帧流，浏览器 `<img>` 即可实时播放。

### 10.3 日志系统

- `_add_log(level, message)` 收集日志
- `/api/logs?since=N` 增量拉取
- 前端可高亮 warning/danger 并展示在显眼区域

---

## 11. 关键可调参数与调优建议

### 11.1 规则参数

- `ear_threshold`
- `mar_threshold`
- `pitch_threshold`
- `eye_consec_frames`
- `mouth_consec_frames`
- `head_consec_frames`

由 `/api/config` 动态下发并即时生效。

### 11.2 调优建议

1. 先固定光照和摄像头角度做基线测试
2. 用真实驾驶视频统计误报/漏报
3. 先调阈值，再调连续帧
4. 分白天/夜间建立不同参数集
5. 若追求跨设备一致性，优先改用秒级逻辑

---

## 12. 已知工程权衡

1. `solvePnP` 相机内参为近似值，绝对角度会有偏差
2. 遮挡与无人驾驶采用近似策略，复杂场景可能误判
3. 当前 DL 分支在主循环中预留接口，默认未注入模型推理
4. 实时系统受 CPU/GPU 与分辨率影响显著

---

## 13. 建议阅读顺序

1. `web_app.py`：先看主循环与 API
2. `detectors/face_detector.py`：理解关键点来源
3. `detectors/eye_analyzer.py` + `detectors/mouth_analyzer.py`
4. `detectors/head_pose_analyzer.py`
5. `evaluators/fatigue_evaluator.py`
6. `models/data_models.py`

按此顺序可以最快建立“输入帧 → 公式计算 → 规则决策 → 前端展示”的完整心智模型。