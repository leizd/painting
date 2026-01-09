import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# --- 1. 设置绘图参数 ---
# 尝试设置中文字体，如果本地有 SimHei 则会正常显示中文
# 如果没有 SimHei，Matplotlib 通常会回退到默认字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
except:
    pass
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.1,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False
})


# --- 2. 辅助函数 ---
def hex_to_rgb(hex_color):
    """将十六进制颜色转换为 RGB 归一化数值"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def create_chart(categories, labels, data, colors, output_path):
    """绘制横轴百分比堆叠条形图的核心函数"""
    data = np.array(data)

    # 归一化数据为百分比
    row_sums = data.sum(axis=1)
    # 防止除以0
    row_sums[row_sums == 0] = 1
    data = data / row_sums[:, np.newaxis] * 100

    # 创建画布，根据分类数量调整高度
    fig, ax = plt.subplots(figsize=(14, 12))
    bar_height = 0.6
    y_positions = np.arange(len(categories))
    left_edges = np.zeros(len(categories))
    segment_right_edges = np.zeros((len(categories), len(labels)))

    # 颜色处理
    if len(colors) < len(labels):
        colors = colors * (len(labels) // len(colors) + 1)
    colors = colors[:len(labels)]
    rgb_colors = [hex_to_rgb(color) for color in colors]

    bars_by_category = [[] for _ in range(len(categories))]

    # 绘制堆叠条形
    for i, (label, color, rgb_color) in enumerate(zip(labels, colors, rgb_colors)):
        values = data[:, i]
        bars = ax.barh(
            y_positions,
            values,
            height=bar_height,
            left=left_edges,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            label=label,
            zorder=5
        )

        # 添加数值标签
        for idx, (bar, val) in enumerate(zip(bars, values)):
            bars_by_category[idx].append(bar)
            if val > 3:  # 仅在占比大于3%时显示数值
                x_center = bar.get_x() + bar.get_width() / 2
                y_center = bar.get_y() + bar.get_height() / 2

                # 根据背景亮度选择文本颜色
                bg_color = np.array(rgb_color[:3])
                brightness = np.dot(bg_color, [0.299, 0.587, 0.114])
                text_color = 'white' if brightness < 0.6 else 'black'

                ax.text(
                    x_center,
                    y_center,
                    f'{val:.1f}%',
                    va='center',
                    ha='center',
                    color=text_color,
                    fontsize=9,
                    fontweight='bold',
                    zorder=10
                )
        segment_right_edges[:, i] = left_edges + values
        left_edges += values

    # 绘制分类间的连线
    if len(categories) > 1:
        for col_idx in range(len(labels)):
            for row_idx in range(len(categories) - 1):
                x1 = segment_right_edges[row_idx, col_idx]
                x2 = segment_right_edges[row_idx + 1, col_idx]

                if x1 > 0.1 and x2 > 0.1:
                    bar_top = bars_by_category[row_idx][col_idx]
                    bar_bottom = bars_by_category[row_idx + 1][col_idx]

                    # 这里的 y1, y2 计算是为了连接相邻条形的间隙
                    # y1 是上面条形的顶部（Matplotlib默认原点在左下，y向上增加）
                    # 但为了逻辑清晰，我们连接的是前一个条形的"下边缘"还是"上边缘"？
                    # 代码逻辑是：连接 row_idx 条形的顶部 和 row_idx+1 条形的底部
                    # 因为 barh 默认是从下往上画的（索引0在最下），所以 row_idx+1 在 row_idx 上方
                    # 我们这里由于后面会 invert_yaxis，视觉上是从上往下

                    y1 = bar_top.get_y() + bar_top.get_height()
                    y2 = bar_bottom.get_y()

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color='black',
                        linewidth=1.0,
                        alpha=0.7,
                        zorder=2,
                        clip_on=False,  # 允许线条画在坐标轴外（如果有必要）
                        solid_capstyle='round'
                    )

    # --- 3. 图表修饰 ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlabel('占比 / Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

    # 次要网格线
    if len(categories) > 1:
        minor_locations = np.arange(len(categories) - 1) + 0.5
        ax.set_yticks(minor_locations, minor=True)
        ax.grid(which='minor', axis='y', linestyle='--', alpha=0.5, color='gray', linewidth=0.5, zorder=0)

    # 图例设置
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(len(labels), 6),
        frameon=True,
        fontsize=11,
        title='机组 / Units',
        title_fontsize=12,
        framealpha=0.9,
        edgecolor='lightgray'
    )

    ax.set_title('各时段机组出力占比分布', fontsize=16, fontweight='bold', pad=40)

    # 反转Y轴，让 Period 1 在最上面
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图表已保存至: {output_path}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 读取数据
    # 请确保文件名与您的文件一致
    filename = 'Problem1_Results_PuLP.csv'
    try:
        df = pd.read_csv(filename)

        # 2. 准备绘图数据
        # 生成分类标签：Period 1, Period 2...
        categories = [f"Period {i}" for i in df['Period']]

        # 选择需要堆叠的列
        unit_cols = ['P_Unit1', 'P_Unit2', 'P_Unit5', 'P_Unit8', 'P_Unit11', 'P_Unit13']
        # 对应的图例标签
        labels = ['Unit 1', 'Unit 2', 'Unit 5', 'Unit 8', 'Unit 11', 'Unit 13']

        data = df[unit_cols].values

        # 3. 选择配色方案 (使用您代码中的配色2)
        colors = ["#AC64A9", "#4CC0A0", "#EE8D25", "#045C57", "#F58E91", "#B92725"]

        # 4. 执行绘图
        create_chart(categories, labels, data, colors, 'stacked_bar_chart_custom.png')

    except FileNotFoundError:
        print(f"未找到文件 {filename}，请检查路径。")
    except Exception as e:
        print(f"发生错误: {e}")