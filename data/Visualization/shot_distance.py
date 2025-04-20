import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter  # 新增百分比格式化模块

# 读取数据
with open("/Users/kehangchen/Library/Mobile Documents/com~apple~CloudDocs/All_code/shotershub/data/MIN_WAS_22300003/shot_location.json", "r", encoding="utf-8") as file:
    data = json.load(file)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # Mac 兼容
plt.rcParams["axes.unicode_minus"] = False
# 定义球队ID常量（根据实际数据修改）
HOME_TEAM_ID = 1610612748  # 迈阿密热火队（示例中未出现）
AWAY_TEAM_ID = 1610612764  # 华盛顿奇才队

# 划分主客队数据（修正逻辑）
home_team_shots = []
away_team_shots = []

for shot in data:
    if shot["TEAM_ID"] == HOME_TEAM_ID:
        home_team_shots.append(shot)
    elif shot["TEAM_ID"] == AWAY_TEAM_ID:
        away_team_shots.append(shot)

# 定义距离分段（每5英尺为一个区间）
max_distance = max(shot["SHOT_DISTANCE"] for shot in data)
bins = list(range(0, int(max_distance) + 6, 5))  # +6确保覆盖最大值
labels = [f"{i}-{i+5}ft" for i in bins[:-1]]  # 添加单位说明

def calculate_hit_rates(shots, bins):
    """计算指定投篮数据的分段命中率（跳过无数据区间）"""
    hit_rates = []
    bin_centers = []
    for i in range(len(bins)-1):
        lower = bins[i]
        upper = bins[i+1]
        
        made = 0
        attempts = 0
        
        for shot in shots:
            distance = shot["SHOT_DISTANCE"]
            if lower <= distance < upper:
                attempts += 1
                made += 1 if shot["SHOT_MADE"] else 0
        
        # 仅保留有数据的区间
        if attempts > 0:
            hit_rate = made / attempts
            hit_rates.append(hit_rate)
            bin_centers.append((lower + upper)/2)
    
    return bin_centers, hit_rates

# 计算命中率（过滤空区间）
home_bins, home_rates = calculate_hit_rates(home_team_shots, bins)
away_bins, away_rates = calculate_hit_rates(away_team_shots, bins)

# 可视化设置
plt.figure(figsize=(12, 6))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))  # Y轴百分比格式化

# 主队折线图（示例中无数据）
if len(home_rates) > 0:
    plt.plot(home_bins, home_rates, 
            marker='o', markersize=8, linestyle='-', linewidth=2,
            color='#E63946', label=f'主队')

# 客队折线图（实际有数据）
plt.plot(away_bins, away_rates,
        marker='s', markersize=8, linestyle='--', linewidth=2,
        color='#2A9D8F', label=f'客队')

# 美化设置
plt.xticks(bins[:-1], labels, rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("投篮距离区间", fontsize=12, labelpad=10)
plt.ylabel("命中率", fontsize=12, labelpad=10)
plt.title("主客队不同距离命中率对比", fontsize=14, pad=20)
plt.ylim(0, 1.05)  # 留出标注空间
plt.grid(True, alpha=0.3, linestyle=':')

# 动态标注数据点（自动跳过缺失值）
def annotate_points(x_values, y_values, color):
    for x, y in zip(x_values, y_values):
        plt.text(x, y+0.03, f'{y:.0%}', 
                ha='center', va='bottom',
                fontsize=9, color=color,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

if len(home_rates) > 0:
    annotate_points(home_bins, home_rates, '#E63946')
annotate_points(away_bins, away_rates, '#2A9D8F')

# 智能图例（仅显示有数据的队伍）
handles, labels = plt.gca().get_legend_handles_labels()
if len(home_rates) == 0:  # 如果主队无数据，移除图例条目
    handles = [h for h, l in zip(handles, labels) if l.startswith('客队')]
    labels = [l for l in labels if l.startswith('客队')]
plt.legend(handles, labels, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()