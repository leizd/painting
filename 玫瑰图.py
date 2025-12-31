import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import matplotlib.font_manager as fm

# ==================== 配置区域 ====================
DATA_FILE = "Problem1_Results_PuLP.csv"
OUTPUT_FOLDER = "visualization_results"
# 更改为更具视觉冲击力的 'plasma' (或者 'magma', 'inferno', 'Spectral_r')
ROSE_COLORMAP = 'plasma'
ROSE_TITLE = '24小时负荷分布 (优化配色)'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 字体设置
font_names = [f.name for f in fm.fontManager.ttflist]
if 'SimHei' in font_names:
    plt.rcParams['font.family'] = ['SimHei']
elif 'Microsoft YaHei' in font_names:
    plt.rcParams['font.family'] = ['Microsoft YaHei']
else:
    plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def create_improved_rose_chart(file_path, output_path):
    # 读取数据
    if not os.path.exists(file_path):
        print("文件不存在")
        return

    df = pd.read_csv(file_path)
    categories = df['Period'].astype(str).tolist()
    measurements = df['Load'].tolist()

    # 绘图准备
    item_num = len(categories)
    angle_distribution = np.linspace(0, 2 * pi, item_num, endpoint=False)
    bar_width = 2 * pi / item_num * 0.85  # 稍微加宽一点，减少缝隙

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    max_val = max(measurements)
    min_val = min(measurements)

    # 颜色映射
    norm = plt.Normalize(min_val, max_val)
    # 使用 plasma colormap
    colors = plt.cm.plasma(norm(measurements))

    # 绘制
    bars = ax.bar(angle_distribution, measurements, width=bar_width, bottom=0,
                  color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)

    # 设置方向和起始点
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # X轴标签
    ax.set_xticks(angle_distribution)
    ax.set_xticklabels(categories, fontsize=11, color='#444444')

    # 标签旋转优化
    for label, angle in zip(ax.get_xticklabels(), angle_distribution):
        label.set_horizontalalignment('center')
        angle_deg = angle * 180 / pi
        if 90 < angle_deg < 270:
            label.set_rotation(angle_deg + 90)
        else:
            label.set_rotation(angle_deg - 90)

    # Y轴设置
    ax.set_ylim(0, max_val * 1.1)
    ax.set_yticks(np.linspace(0, max_val, 5)[1:])
    ax.set_yticklabels([])  # 隐藏Y轴数值标签，让画面更干净，或者保留并设淡
    # 如果想保留Y轴数字，取消上面这行，使用下面这行：
    # ax.set_yticklabels([f'{int(y)}' for y in np.linspace(0, max_val, 5)[1:]], color='gray', fontsize=8)

    # 网格线优化
    ax.grid(True, alpha=0.15, color='black', linestyle='--')
    ax.spines['polar'].set_visible(False)  # 隐藏外圈圆线

    # 标题
    plt.title(ROSE_TITLE, fontsize=18, fontweight='bold', pad=30, color='#333333')

    # Colorbar 优化
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.outline.set_visible(False)
    cbar.set_label('Load (MW)', rotation=270, labelpad=15, fontsize=10, color='#555555')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已生成: {output_path}")


if __name__ == "__main__":
    create_improved_rose_chart(DATA_FILE, os.path.join(OUTPUT_FOLDER, "Improved_Rose_Chart.png"))