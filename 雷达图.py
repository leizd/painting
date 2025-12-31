import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd

# ===================== 01 环境准备与数据读取 =====================
# 从CSV文件读取数据
df = pd.read_csv('Problem1_Results_PuLP.csv')

# 提取Categories（角度轴）：Period 1-24
categories = df['Period'].astype(str).tolist()
num_vars = len(categories)

# 提取Group数据（径向轴）：选择具有代表性的机组
group_names = ['P_Unit1', 'P_Unit2', 'P_Unit8']
data_sets = {col: df[col].tolist() for col in group_names}

# 提取Size数据（动态大小）：使用 Load (系统负荷)
load_values = df['Load'].tolist()
# 归一化 Load 值以控制圆点大小 (min 20, max 200)
min_load, max_load = min(load_values), max(load_values)
min_size, max_size = 20, 250
marker_sizes = [min_size + (x - min_load) / (max_load - min_load) * (max_size - min_size)
                for x in load_values]

# ===================== 02 颜色配置 =====================
color_palette = ['#ED949A', '#B2A3DD', '#96CCEA', '#A4DDD3']
current_colors = {name: color_palette[i % len(color_palette)]
                  for i, name in enumerate(group_names)}

# ===================== 03 创建极坐标图 =====================
# 创建画布
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# 生成角度（从顶部开始，顺时针排列）
# np.pi/2 是顶部，减去 2*np.pi 表示顺时针旋转一圈
angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, num_vars, endpoint=False)

# ===================== 04 绘制参考网格 =====================
# 基础半径（中心留白）
base_radius = 20

# 1. 绘制中心留白圆
circle = plt.Circle((0, 0), base_radius, transform=ax.transData._b,
                    facecolor='white', edgecolor='black', linestyle='--',
                    linewidth=1, alpha=1.0, zorder=1)
ax.add_patch(circle)

# 2. 绘制参考圆圈 (Grid Lines)
# 计算最大半径范围
max_val = max([max(d) for d in data_sets.values()])
max_radius_for_lines = base_radius + max_val * 1.1  # 留出一点余量

# 中间参考线 (例如 50MW 处)
mid_val = 50
mid_radius = base_radius + mid_val
mid_circle = plt.Circle((0, 0), mid_radius, transform=ax.transData._b,
                        facecolor='none', edgecolor='lightgray', linestyle='--',
                        linewidth=0.8, alpha=1.0, zorder=0)
ax.add_patch(mid_circle)

# 外围边界线
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(theta, [max_radius_for_lines] * len(theta),
        color='black', linewidth=1.5, zorder=1)

# 3. 绘制径向线 (Spokes)
for angle in angles:
    ax.plot([angle, angle], [base_radius, max_radius_for_lines],
            color='lightgray', linestyle='--', linewidth=0.8, zorder=0)

# ===================== 05 绘制数据区域 =====================
for name in group_names:
    data = data_sets[name]
    color = current_colors[name]

    # 计算数据点半径 (Base + Value)
    line_radii = [base_radius + value for value in data]

    # 闭合线条 (将第一个点追加到末尾)
    closed_angles = np.append(angles, angles[0])
    closed_radii = np.append(line_radii, line_radii[0])

    # 绘制线条
    ax.plot(closed_angles, closed_radii, color=color, linewidth=2, zorder=2)

    # 填充区域
    ax.fill(closed_angles, closed_radii, color=color, alpha=0.15, zorder=1)

    # 绘制动态大小的散点
    # 注意：scatter 不需要闭合，直接使用原始 angles 和 radii
    # s 参数接收大小列表，实现动态大小
    ax.scatter(angles, line_radii, color=color, s=marker_sizes,
               alpha=0.9, edgecolors='white', linewidth=0.8, zorder=3)

# ===================== 06 添加标签 =====================
label_radius_category = max_radius_for_lines + 10

for i, (angle, label) in enumerate(zip(angles, categories)):
    # 计算角度用于旋转标签
    angle_deg = np.rad2deg(angle)
    # 调整角度到 0-360 范围
    normalized_angle = angle_deg % 360

    # 根据位置调整标签旋转方向，使其易读
    if 90 < normalized_angle < 270:
        rotation = normalized_angle + 180
    else:
        rotation = normalized_angle

    ax.text(angle, label_radius_category, label,
            ha='center', va='center', fontsize=11,
            fontfamily='Times New Roman', rotation=rotation - 90,
            weight='bold', color='black')

# ===================== 07 美化与保存 =====================
# 隐藏默认极坐标轴
ax.set_ylim(0, max_radius_for_lines + 15)
ax.spines['polar'].set_visible(False)
ax.grid(False)
ax.set_yticklabels([])
ax.set_xticks([])

# 添加图例
# 1. 颜色图例 (Units)
group_patches = [mpatches.Patch(color=current_colors[name], alpha=0.6, label=name)
                 for name in group_names]

fig.legend(handles=group_patches, loc='lower center',
           bbox_to_anchor=(0.5, 0.05), frameon=False, ncol=3,
           prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})

# 2. 大小图例 (Load) - 手动绘制
# 在图表右侧添加一个小区域画圆圈示例
legend_ax = fig.add_axes([0.85, 0.15, 0.1, 0.2])  # [left, bottom, width, height]
legend_ax.axis('off')
legend_ax.set_title('Load (MW)', fontsize=10, weight='bold', loc='center')

# 绘制三个示例圆圈：小、中、大
sample_loads = [min_load, (min_load + max_load) / 2, max_load]
sample_sizes = [min_size, (min_size + max_size) / 2, max_size]

for i, (size, load) in enumerate(zip(sample_sizes, sample_loads)):
    y_pos = i * 0.4 + 0.2
    legend_ax.scatter(0.5, y_pos, s=size, color='gray', alpha=0.5, edgecolors='none')
    legend_ax.text(1.2, y_pos, f'{int(load)}', va='center', fontsize=10)

legend_ax.set_xlim(0, 2)
legend_ax.set_ylim(0, 1.5)

# 保存与展示
plt.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.15)
plt.savefig('dynamic_radar_chart.png', dpi=600)
plt.show()