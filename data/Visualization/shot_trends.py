import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 读取 JSON 文件
with open("/Users/kehangchen/Library/Mobile Documents/com~apple~CloudDocs/All_code/shotershub/data/MIN_WAS_22300003/shot_location.json", "r", encoding="utf-8") as file:
    shooting_data = json.load(file)

# 提取必要数据
distances = np.array([shot["SHOT_DISTANCE"] for shot in shooting_data])
loc_x = np.array([shot["LOC_X"] for shot in shooting_data])
shot_made = np.array([shot["SHOT_MADE"] for shot in shooting_data])

# **筛选左侧 & 右侧投篮**
left_mask = loc_x < 0
right_mask = loc_x > 0

left_distances = distances[left_mask]
right_distances = distances[right_mask]

left_made = shot_made[left_mask]
right_made = shot_made[right_mask]

# **计算不同距离的命中率**
distance_bins = np.arange(0, 30, 2)  # 2 英尺间隔
left_percentage = []
right_percentage = []

for d in distance_bins:
    left_shots = left_made[(left_distances >= d) & (left_distances < d + 2)]
    right_shots = right_made[(right_distances >= d) & (right_distances < d + 2)]
    
    left_percentage.append(np.mean(left_shots) * 100 if len(left_shots) > 0 else None)
    right_percentage.append(np.mean(right_shots) * 100 if len(right_shots) > 0 else None)

# **转换 None 为 NaN**
left_percentage = np.array([x if x is not None else np.nan for x in left_percentage])
right_percentage = np.array([x if x is not None else np.nan for x in right_percentage])

# **计算 Difference，缺失值设为 0**
difference = np.nan_to_num(left_percentage - right_percentage, nan=0)

# **计算整体命中率**
total_left = np.nanmean(left_made) * 100
total_right = np.nanmean(right_made) * 100

# **颜色计算（基于差值）**
colors = []
for diff in difference:
    if diff > 0:
        color = mcolors.to_rgba("#E63946", alpha=min(abs(diff) / 20, 1))
    else:
        color = mcolors.to_rgba("#2A9D8F", alpha=min(abs(diff) / 20, 1))
    colors.append(color)

# **绘制图像**
fig, ax = plt.subplots(figsize=(7, 5))

# **修正 X 轴：让左侧数据显示在左侧，右侧数据在右侧**
ax.barh(distance_bins, -np.nan_to_num(left_percentage, nan=0), color="#2A9D8F", alpha=0.5, label="Left", align="center")
ax.barh(distance_bins, np.nan_to_num(right_percentage, nan=0), color="#E63946", alpha=0.5, label="Right", align="center")

# **画 Difference 折线（如果缺失值，保持在 0 轴）**
for i in range(len(distance_bins) - 1):
    x1 = difference[i] if not np.isnan(left_percentage[i]) and not np.isnan(right_percentage[i]) else 0
    x2 = difference[i + 1] if not np.isnan(left_percentage[i + 1]) and not np.isnan(right_percentage[i + 1]) else 0

    ax.plot(
        [x1, x2],  # X 轴数值
        [distance_bins[i], distance_bins[i + 1]],  # Y 轴数值
        color=colors[i], lw=2
    )

# 画三分线
ax.axhline(y=23.75, color="gray", linestyle="dashed", alpha=0.5)
ax.text(-5, 23.75, "3pt", color="gray", fontsize=10, alpha=0.7)

# **调整 X 轴**
ax.set_xlabel("Shooting Percentage (%)")
ax.set_ylabel("Distance (ft)")
ax.set_xlim(-100, 100)  # **确保 X 轴是对称的**
ax.set_xticks([-100, -50, 0, 50, 100])
ax.set_xticklabels(["100", "50", "0", "50", "100"])  # **修正刻度**
ax.axvline(x=0, color="black", lw=1)  # 添加中线

# 添加图例
ax.legend()

# **顶部 Overall % 条形图**
ax_inset = fig.add_axes([0.25, 0.9, 0.5, 0.05])  # 位置
ax_inset.barh([0], [total_left], color="#2A9D8F")
ax_inset.barh([0], [total_right], color="#E63946", left=[total_left])
ax_inset.text(total_left / 2, 0, f"{total_left:.0f}%", ha="center", va="center", color="black", fontsize=10)
ax_inset.text(total_left + total_right / 2, 0, f"{total_right:.0f}%", ha="center", va="center", color="black", fontsize=10)

# 隐藏 inset 轴的刻度
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_frame_on(False)

# 显示图像
plt.show()
