import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False
# ====================== 1. 加载数据 ======================
# CSV 文件（取消注释即可）
df = pd.read_csv("calibration_data.csv")

# ====================== 2. ROC 计算最优阈值函数 ======================
def get_best_threshold(y_true, y_score, direction="above"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_th = thresholds[best_idx]
    
    if direction == "below":
        best_th = -best_th  # EAR 越小越异常，需要取反
    
    return best_th, fpr, tpr, thresholds

# ====================== 3. 计算三个指标最优阈值 ======================
# 1) EAR 阈值（闭眼：越小越异常 → direction="below"）
y_ear = (df["label"] == "drowsy").astype(int)
best_ear, _, _, _ = get_best_threshold(y_ear, -df["EAR"], direction="below")

# 2) MAR 阈值（打哈欠：越大越异常 → direction="above"）
y_mar = (df["label"] == "yawn").astype(int)
best_mar, _, _, _ = get_best_threshold(y_mar, df["MAR"], direction="above")

# 3) Pitch 阈值（低头：绝对值越大越异常）
y_pitch = (df["label"] == "drowsy").astype(int)
best_pitch, _, _, _ = get_best_threshold(y_pitch, df["Pitch"].abs(), direction="above")

# ====================== 4. 输出最终结果 ======================
print("=" * 60)
print("          ROC 曲线最优阈值计算结果（可直接写入系统）")
print("=" * 60)
print(f"EAR 最优闭眼阈值   = {best_ear:.4f}")
print(f"MAR 最优哈欠阈值   = {best_mar:.4f}")
print(f"Pitch 最优低头阈值 = {best_pitch:.4f}")
print("=" * 60)

# ====================== 5. 绘制 ROC 曲线图（论文用）======================
plt.figure(figsize=(10, 4))

# EAR ROC
plt.subplot(1,3,1)
fpr_ear, tpr_ear, _ = roc_curve(y_ear, -df["EAR"])
plt.plot(fpr_ear, tpr_ear, label=f"EAR (th={best_ear:.3f})")
plt.title("EAR ROC (闭眼)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(True)
plt.legend()

# MAR ROC
plt.subplot(1,3,2)
fpr_mar, tpr_mar, _ = roc_curve(y_mar, df["MAR"])
plt.plot(fpr_mar, tpr_mar, label=f"MAR (th={best_mar:.3f})")
plt.title("MAR ROC (哈欠)")
plt.xlabel("FPR")
plt.grid(True)
plt.legend()

# Pitch ROC
plt.subplot(1,3,3)
fpr_pitch, tpr_pitch, _ = roc_curve(y_pitch, df["Pitch"].abs())
plt.plot(fpr_pitch, tpr_pitch, label=f"Pitch (th={best_pitch:.3f})")
plt.title("Pitch ROC (低头)")
plt.xlabel("FPR")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()