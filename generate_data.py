import pandas as pd
import numpy as np

# ====================== 真实分布 + 噪声 + 区间重叠 ======================
np.random.seed(42)

# 帧数（和你ROC代码保持一致）
n_normal = 100
n_drowsy = 60
n_yawn = 60

# -------------------------- EAR 分布 --------------------------
# 正常睁眼：均值 0.30，带噪声
ear_normal = np.random.normal(loc=0.30, scale=0.035, size=n_normal)

# 疲劳闭眼：围绕 0.20 波动，允许 0.16~0.24
ear_drowsy = np.random.normal(loc=0.195, scale=0.025, size=n_drowsy)

# 打哈欠时眼睛正常睁开
ear_yawn = np.random.normal(loc=0.30, scale=0.035, size=n_yawn)

# -------------------------- MAR 分布 --------------------------
# 正常闭嘴：均值 0.25
mar_normal = np.random.normal(loc=0.25, scale=0.06, size=n_normal)

# 疲劳时嘴巴正常
mar_drowsy = np.random.normal(loc=0.26, scale=0.06, size=n_drowsy)

# 打哈欠：均值 0.45，大于 0.35
mar_yawn = np.random.normal(loc=0.45, scale=0.08, size=n_yawn)

# -------------------------- Pitch 分布 --------------------------
# 正常姿态：0 左右
pitch_normal = np.random.normal(loc=0, scale=8, size=n_normal)

# 低头疲劳：均值 30°，大于 25°
pitch_drowsy = np.random.normal(loc=32, scale=7, size=n_drowsy)

# 打哈欠头部姿态正常
pitch_yawn = np.random.normal(loc=2, scale=7, size=n_yawn)

# ====================== 合并数据 ======================
data = {
    "EAR": np.concatenate([ear_normal, ear_drowsy, ear_yawn]),
    "MAR": np.concatenate([mar_normal, mar_drowsy, mar_yawn]),
    "Pitch": np.concatenate([pitch_normal, pitch_drowsy, pitch_yawn]),
    "label": ["normal"] * n_normal + ["drowsy"] * n_drowsy + ["yawn"] * n_yawn
}

df = pd.DataFrame(data)

# ====================== 加微小测量噪声（更真实） ======================
df["EAR"] += np.random.normal(0, 0.008, len(df))
df["MAR"] += np.random.normal(0, 0.015, len(df))
df["Pitch"] += np.random.normal(0, 0.8, len(df))

# ====================== 保存 ======================
df.to_csv("calibration_data.csv", index=False, encoding="utf-8")
print("✅ 成功生成 自然真实版 calibration_data.csv")
print("\n数据规则：")
print("• EAR 围绕 0.20 区分闭眼（drowsy）")
print("• MAR 围绕 0.35 区分打哈欠（yawn）")
print("• Pitch 围绕 25° 区分低头（drowsy）")
print("• 带测量噪声 + 区间重叠 → ROC 曲线自然顺滑")