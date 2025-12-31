import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===================== 全局样式配置 =====================
# 设置全局字体，优先尝试 Times New Roman，如果不可用则使用默认无衬线字体
plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

# ===================== 数据加载 =====================
# 读取上传的CSV文件
input_file = 'Problem1_Results_PuLP.csv'
df = pd.read_csv(input_file)

# 映射数据列
# X轴：Period (转为字符串以作为标签)
datasets = df['Period'].astype(str).tolist()
# Y轴：P_Unit1 (机组1出力)
values = df['P_Unit1'].tolist()
# 大小控制：Load (系统负荷)
amounts = df['Load'].tolist()

# 生成x轴位置坐标
x_positions = np.arange(len(datasets))

# ===================== 颜色配置 =====================
# 棒棒糖颜色 (循环使用)
edge_colors = [
                  '#4791C1', '#4791C1', '#4791C1',
                  '#A0CDE4', '#A0CDE4', '#A0CDE4',
                  '#FAB37D', '#FAB37D', '#4791C1',
                  '#4791C1', '#4791C1', '#FAB37D',
                  '#A0CDE4', '#A0CDE4', '#FAB37D',
                  '#FAB37D', '#FAB37D', '#FAB37D'
              ] * 2  # 复制一次以覆盖24个数据点

# 背景矩形的填充颜色 (统一为白色)
background_colors = ['#FFFFFF']

# ===================== 创建图形和坐标轴 =====================
# 创建画布
fig = plt.figure(figsize=(14, 5))  # 稍微加宽以容纳24个点

# 添加自定义坐标轴 (left, bottom, width, height)
# 留出右侧空间给图例
ax = fig.add_axes((0.08, 0.15, 0.78, 0.75))

# ===================== 坐标轴样式设置 =====================
# 隐藏上轴和右轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 动态计算Y轴范围 (比最大值稍高一点)
max_val = max(values)
y_lim_max = max_val * 1.2
y_lim = (0, y_lim_max)

# 设置Y轴标签
ax.set_ylabel('Power Output (MW)', fontsize=16, labelpad=10, fontweight='bold')
ax.set_ylim(y_lim)

# 设置X轴
ax.set_xticks(x_positions)
ax.set_xlim(-0.8, len(datasets) - 0.2)
ax.set_xticklabels(datasets, rotation=0, ha='center', fontsize=12, fontweight='bold')
ax.set_xlabel('Period', fontsize=16, labelpad=10, fontweight='bold')

# 设置刻度样式
ax.tick_params(axis='both', width=1.5, length=4, labelsize=12)

# ===================== 动态大小参数配置 =====================
amount_min = min(amounts)
amount_max = max(amounts)

# 设置圆点大小范围
min_size = 50
max_size = 350
ring_size_scale = 1.3
line_width = 1.5
ring_width = 1.0
inner_scale = 0.65
label_y_offset = max_val * 0.08  # 根据数据范围动态调整标签偏移量

# ===================== 绘制动态大小棒棒糖 =====================
for i, (value, position, amount) in enumerate(zip(values, x_positions, amounts)):
    edge_color = edge_colors[i % len(edge_colors)]

    # 计算大小
    if amount_max > amount_min:
        inner_size = min_size + (max_size - min_size) * ((amount - amount_min) / (amount_max - amount_min))
    else:
        inner_size = min_size

    outer_size = inner_size * ring_size_scale

    # 1. 绘制杆
    ax.plot([position, position], [0, value],
            color=edge_color, linewidth=line_width, alpha=1, zorder=2)

    # 2. 绘制外圈 (空心)
    ax.scatter(position, value, s=outer_size,
               facecolor='white', edgecolor=edge_color,
               linewidth=ring_width, alpha=1.0, zorder=3)

    # 3. 绘制内圈 (实心)
    ax.scatter(position, value,
               s=inner_size * inner_scale,
               facecolor=edge_color, edgecolor='none',
               alpha=1.0, zorder=4)

    # 4. 绘制数值标签
    ax.text(position, value + label_y_offset,
            f'{value:.0f}',
            ha='center', va='center',
            fontsize=10, fontweight='bold', color='black', zorder=5)

# ==================== 添加大小图例 (右侧) ====================
legend_ax = fig.add_axes([0.86, 0.3, 0.1, 0.4])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis('off')

# 图例标题
legend_ax.text(0.5, 1.0, 'Load (MW)', fontsize=14, fontweight='bold', ha='center', va='bottom')

# 图例示例
legend_amounts = [amount_min, (amount_min + amount_max) / 2, amount_max]
legend_labels = [f'{int(x)}' for x in legend_amounts]
legend_y_positions = np.linspace(0.8, 0.2, len(legend_amounts))
legend_color = edge_colors[0]

for i, (amount, y_pos) in enumerate(zip(legend_amounts, legend_y_positions)):
    if amount_max > amount_min:
        s = min_size + (max_size - min_size) * ((amount - amount_min) / (amount_max - amount_min))
    else:
        s = min_size

    outer_s = s * ring_size_scale

    # 外圈
    legend_ax.scatter(0.3, y_pos, s=outer_s,
                      facecolor='none', edgecolor=legend_color,
                      linewidth=ring_width, zorder=3)
    # 内圈
    legend_ax.scatter(0.3, y_pos, s=s * inner_scale,
                      facecolor=legend_color, edgecolor='none', zorder=4)

    # 标签
    legend_ax.text(0.6, y_pos, legend_labels[i],
                   fontsize=12, fontweight='bold', ha='left', va='center')

# 保存结果
output_file = 'dynamic_lollipop_chart.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight')