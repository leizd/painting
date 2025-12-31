import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import os

# ==================== 全局样式配置 ====================
plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans']  # 优先使用Times New Roman
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

# ==================== 颜色配置 ====================
COLOR_SCHEMES = [
    '#87CEEB',  # 天蓝色
    '#ED949A'  # 柔红色
]

# ==================== 图形参数 ====================
FIG_CONFIG = {
    'figsize': (12, 8),  # 稍微加宽以适应标签
    'axes_position': (0.08, 0.15, 0.38, 0.75),  # 左子图位置
    'spines_linewidth': 1.5,
    'tick_length': 6,
    'tick_width': 1.5,
    'fontsize_labels': 14,
    'fontsize_title': 16,
    'fontsize_legend': 14,
    'y_tick_label_pad': 10  # 调整间距
}

# ==================== 透明度配置 ====================
VIOLIN_ALPHA = 0.3


# ==================== 数据加载与预处理函数 ====================
def load_and_transform_data(input_file):
    """
    读取原始CSV并转换为绘图所需的格式：
    Dataset (Unit), Group (AM/PM), Indicator (Metric), Value
    """
    raw_df = pd.read_csv(input_file)

    # 提取功率列
    p_cols = [c for c in raw_df.columns if c.startswith('P_Unit')]

    transformed_data = []

    for _, row in raw_df.iterrows():
        period = row['Period']
        # 定义分组：1-12为Group1(AM)，13-24为Group2(PM)
        group = 'Group1' if period <= 12 else 'Group2'

        for col in p_cols:
            unit_name = col.replace('P_', '')  # 简化名称，例如 Unit1
            value = row[col]

            transformed_data.append({
                'Dataset': unit_name,
                'Group': group,
                'Indicator': 'Power Output',
                'Value': value
            })

    df = pd.DataFrame(transformed_data)

    datasets = sorted(df['Dataset'].unique(), key=lambda x: int(x.replace('Unit', '')))  # 按数字排序
    indicators = df['Indicator'].unique()
    num_indicators = len(indicators)

    y_positions = np.arange(len(datasets))

    if num_indicators == 1:
        offsets = [0]
        widths = 0.5
    else:
        total_width = 0.7
        widths = total_width / num_indicators
        offsets = np.linspace(-total_width / 2 + widths / 2, total_width / 2 - widths / 2, num_indicators)

    return df, datasets, indicators, num_indicators, y_positions, offsets, widths


def prepare_data(df, datasets, indicators):
    """提取各组数据和统计信息"""
    data_dict = {'group1_data': {}, 'group2_data': {}, 'group1_stats': {}, 'group2_stats': {}}

    for indicator in indicators:
        data_dict['group1_data'][indicator] = []
        data_dict['group2_data'][indicator] = []
        data_dict['group1_stats'][indicator] = []
        data_dict['group2_stats'][indicator] = []

        for dataset in datasets:
            # Group1 数据
            g1_data = df[(df['Dataset'] == dataset) & (df['Group'] == 'Group1') & (df['Indicator'] == indicator)][
                'Value'].values
            data_dict['group1_data'][indicator].append(g1_data)

            if len(g1_data) > 0:
                data_dict['group1_stats'][indicator].append({
                    'median': np.median(g1_data),
                    'q1': np.percentile(g1_data, 25),
                    'q3': np.percentile(g1_data, 75),
                    'whislo': np.min(g1_data),
                    'whishi': np.max(g1_data)
                })
            else:
                data_dict['group1_stats'][indicator].append({'median': 0, 'q1': 0, 'q3': 0, 'whislo': 0, 'whishi': 0})

            # Group2 数据
            g2_data = df[(df['Dataset'] == dataset) & (df['Group'] == 'Group2') & (df['Indicator'] == indicator)][
                'Value'].values
            data_dict['group2_data'][indicator].append(g2_data)

            if len(g2_data) > 0:
                data_dict['group2_stats'][indicator].append({
                    'median': np.median(g2_data),
                    'q1': np.percentile(g2_data, 25),
                    'q3': np.percentile(g2_data, 75),
                    'whislo': np.min(g2_data),
                    'whishi': np.max(g2_data)
                })
            else:
                data_dict['group2_stats'][indicator].append({'median': 0, 'q1': 0, 'q3': 0, 'whislo': 0, 'whishi': 0})

    return data_dict


# ==================== 绘图组件函数 ====================
def create_half_violinplot(ax, data, positions, color, width, alpha=0.3):
    """绘制半小提琴图"""
    # 过滤空数据，防止报错
    valid_data = [d if len(d) > 0 else np.array([0]) for d in data]

    try:
        violin_parts = ax.violinplot(valid_data, positions=positions, vert=False,
                                     showmeans=False, showextrema=False, widths=width)

        for j, pc in enumerate(violin_parts['bodies']):
            if len(data[j]) == 0:  # 如果原始数据为空，隐藏
                pc.set_visible(False)
                continue

            path = pc.get_paths()[0]
            vertices = path.vertices
            center_y = positions[j]
            # 保留上半部分
            vertices[:, 1] = np.where(vertices[:, 1] > center_y, vertices[:, 1], center_y)
            path.vertices = vertices
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(alpha)
            pc.set_linewidth(1.5)

        return violin_parts
    except Exception as e:
        print(f"Warning in violin plot: {e}")
        return None


def draw_boxplot(ax, stats, pos, color):
    """绘制简化箱体图"""
    if stats['median'] == 0 and stats['whishi'] == 0: return  # 跳过空数据

    box_width = 0.12
    # 箱体
    ax.add_patch(plt.Rectangle((stats['q1'], pos - box_width / 2),
                               stats['q3'] - stats['q1'], box_width,
                               facecolor='white', edgecolor=color, linewidth=2, zorder=3))
    # 中位数线
    ax.plot([stats['median'], stats['median']],
            [pos - box_width * 0.3, pos + box_width * 0.3],
            color=color, linewidth=2.5, zorder=4)
    # 须线
    ax.plot([stats['q1'], stats['whislo']], [pos, pos], color=color, linewidth=1.5, zorder=2)
    ax.plot([stats['q3'], stats['whishi']], [pos, pos], color=color, linewidth=1.5, zorder=2)
    ax.plot([stats['whislo'], stats['whislo']], [pos - box_width / 3, pos + box_width / 3], color=color, linewidth=1.5)
    ax.plot([stats['whishi'], stats['whishi']], [pos - box_width / 3, pos + box_width / 3], color=color, linewidth=1.5)


def setup_axes(ax1, ax2, datasets, y_positions):
    """设置坐标轴样式"""
    # 数据范围是 0 - 110 MW，设置范围为 0 - 120
    x_max = 120
    x_ticks = [0, 30, 60, 90, 120]
    x_ticklabels = ['0', '30', '60', '90', '120']

    # ==================== Group1子图设置 (Period 1-12) ====================
    ax1.set_xlabel('Power Output (MW)', fontsize=FIG_CONFIG['fontsize_labels'],
                   labelpad=10, fontweight='bold')
    ax1.set_xlim(x_max, 0)  # 反转X轴

    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['right'].set_linewidth(FIG_CONFIG['spines_linewidth'])
    ax1.spines['bottom'].set_linewidth(FIG_CONFIG['spines_linewidth'])
    ax1.tick_params(axis='y', which='both', left=False, right=False)
    ax1.set_yticklabels([])

    ax1.set_yticks(y_positions)
    ax1.set_ylim(-0.6, len(datasets) - 0.4)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticklabels)
    ax1.tick_params(axis='x', width=FIG_CONFIG['tick_width'],
                    length=FIG_CONFIG['tick_length'], labelsize=FIG_CONFIG['fontsize_labels'])

    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    ax1.set_title('Period 1-12 (AM)', fontsize=FIG_CONFIG['fontsize_title'],
                  fontweight='bold', pad=15)

    # ==================== Group2子图设置 (Period 13-24) ====================
    ax2.set_xlabel('Power Output (MW)', fontsize=FIG_CONFIG['fontsize_labels'],
                   labelpad=10, fontweight='bold')
    ax2.set_xlim(0, x_max)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_linewidth(FIG_CONFIG['spines_linewidth'])
    ax2.spines['bottom'].set_linewidth(FIG_CONFIG['spines_linewidth'])
    ax2.tick_params(axis='y', which='both', left=False)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(datasets, va='center', ha='center',
                        fontsize=FIG_CONFIG['fontsize_labels'], fontweight='bold')
    ax2.set_ylim(-0.6, len(datasets) - 0.4)
    ax2.tick_params(axis='y', width=0, length=0, pad=FIG_CONFIG['y_tick_label_pad'])
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticklabels)
    ax2.tick_params(axis='x', width=FIG_CONFIG['tick_width'],
                    length=FIG_CONFIG['tick_length'], labelsize=FIG_CONFIG['fontsize_labels'])

    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    ax2.set_title('Period 13-24 (PM)', fontsize=FIG_CONFIG['fontsize_title'],
                  fontweight='bold', pad=15)


# ==================== 主绘图函数 ====================
def create_half_violin_with_box_and_strip(input_file, output_file):
    # 加载转换后的数据
    df, datasets, indicators, num_indicators, y_positions, offsets, widths = load_and_transform_data(input_file)
    data_dict = prepare_data(df, datasets, indicators)

    # 创建画布
    fig = plt.figure(figsize=FIG_CONFIG['figsize'])

    # 调整子图位置以适应中间的文字标签
    left_ax_pos = list(FIG_CONFIG['axes_position'])  # [0.08, 0.15, 0.38, 0.75]
    right_ax_pos = [left_ax_pos[0] + left_ax_pos[2] + 0.12, left_ax_pos[1], left_ax_pos[2], left_ax_pos[3]]

    ax1 = fig.add_axes(left_ax_pos)  # 左子图
    ax2 = fig.add_axes(right_ax_pos)  # 右子图

    setup_axes(ax1, ax2, datasets, y_positions)

    # 绘制左子图（Group1）
    for i, indicator in enumerate(indicators):
        color = COLOR_SCHEMES[0]  # 使用蓝色
        positions = y_positions + offsets[i]

        # 增加 jitter 防止数据重叠
        # 小提琴
        create_half_violinplot(ax1, data_dict['group1_data'][indicator], positions, color, widths, alpha=VIOLIN_ALPHA)

        for j, pos in enumerate(positions):
            # 箱体
            stats = data_dict['group1_stats'][indicator][j]
            draw_boxplot(ax1, stats, pos, color)

            # 散点
            data = data_dict['group1_data'][indicator][j]
            if len(data) > 0:
                # 添加垂直抖动
                jitter = np.random.uniform(-0.05, 0.05, size=len(data))
                ax1.scatter(data, [pos - widths * 0.35 + jitt for jitt in jitter],
                            color=color, alpha=0.6, s=15, zorder=2, edgecolors='none')

    # 绘制右子图（Group2）
    for i, indicator in enumerate(indicators):
        color = COLOR_SCHEMES[1]  # 使用红色
        positions = y_positions + offsets[i]

        create_half_violinplot(ax2, data_dict['group2_data'][indicator], positions, color, widths, alpha=VIOLIN_ALPHA)

        for j, pos in enumerate(positions):
            stats = data_dict['group2_stats'][indicator][j]
            draw_boxplot(ax2, stats, pos, color)

            data = data_dict['group2_data'][indicator][j]
            if len(data) > 0:
                jitter = np.random.uniform(-0.05, 0.05, size=len(data))
                ax2.scatter(data, [pos - widths * 0.35 + jitt for jitt in jitter],
                            color=color, alpha=0.6, s=15, zorder=2, edgecolors='none')

    # 保存
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()


# 运行
if __name__ == "__main__":
    input_file = 'Problem1_Results_PuLP.csv'
    output_file = 'half_violin_dynamic.png'
    create_half_violin_with_box_and_strip(input_file, output_file)