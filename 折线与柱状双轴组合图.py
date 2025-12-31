import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import math
import numpy as np

# 02 数据读取与预处理
# 读取数据 (请确保文件在当前目录下)
df_raw = pd.read_csv("Problem1_Results_PuLP.csv")
# 仅保留非 Status 的列，确保图形有波动
cols_to_keep = [col for col in df_raw.columns if not col.startswith('Status')]
df = df_raw[cols_to_keep].copy()


# 模拟用户代码中的去中文列名逻辑
def has_chinese(text):
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


cols = [f'column_{i + 1}' if has_chinese(c) else c for i, c in enumerate(df.columns)]
df.columns = cols

x_col = cols[0]  # Period
remain_cols = cols[1:]  # Load, P_Unit1, ...
group_n = len(remain_cols) // 2

# 03 颜色方案配置
color_schemes = [("#8080FF", "#FF8080"), ("#80FFFF", "#8080FF"), ("#FFFF80", "#8080FF"),
                 ("#FF8080", "#8080FF"), ("#80FF80", "#8080FF"), ("#8080FF", "#80FFFF"),
                 ("#FFFF80", "#80FFFF"), ("#FF8080", "#80FFFF"), ("#80FF80", "#8080FF")]


# 04 单图表绘制函数
def draw_one(x_s, bar_s, line_s, idx, clr_low, clr_high):
    fig = plt.figure(figsize=(6.7, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    cmap = LinearSegmentedColormap.from_list("", [clr_low, clr_high])
    x, bar, line = x_s.values, bar_s.values, line_s.values

    # 归一化处理 (防止分母为0)
    denom = line.max() - line.min()
    normalized = (line - line.min()) / denom if denom != 0 else np.zeros_like(line)

    # 绘制柱状图 (颜色由折线数值决定)
    ax.bar(range(len(bar)), bar, width=0.4, color=cmap(normalized))

    # 绘制折线图 (双轴)
    ax2 = ax.twinx()
    ax2.plot(range(len(line)), line, color='crimson',
             marker='o', markerfacecolor='none', markeredgewidth=1.5,
             markersize=7, linewidth=1)

    ax.axhline(0, color='gray', lw=1.2, zorder=0)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=0)
    ax.set_xlabel(x_s.name)
    ax.set_ylabel(bar_s.name)
    ax2.set_ylabel(line_s.name)
    ax.set_title(f'Group {idx}')

    # 颜色条
    norm = mpl.colors.Normalize(vmin=line.min(), vmax=line.max())
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(line_s.name, rotation=270, va='bottom')

    plt.close(fig)  # 关闭以释放内存，稍后在合并时重建
    return fig


# 05 多图表生成 (这里仅做逻辑占位，实际绘图直接在合并步骤完成或单独保存)
# 06 多图表合并展示
cols_per_row = 3
rows = math.ceil(group_n / cols_per_row)
fig_merge = plt.figure(figsize=(5.8 * cols_per_row, 4 * rows), dpi=300)

# 构建 GridSpec
ratios = []
for _ in range(cols_per_row): ratios.extend([20, 1])
gs = gridspec.GridSpec(rows, len(ratios), width_ratios=ratios, wspace=0.7, hspace=0.35)

axes, caxes = [], []
for r in range(rows):
    for i in range(0, len(ratios), 2):
        axes.append(fig_merge.add_subplot(gs[r, i]))
        caxes.append(fig_merge.add_subplot(gs[r, i + 1]))

# 循环绘制并填充到大图中
for g in range(group_n):
    ax_target = axes[g]
    cax_target = caxes[g]

    bar_col = remain_cols[g * 2]
    line_col = remain_cols[g * 2 + 1]
    clr = color_schemes[g % len(color_schemes)]

    # 获取数据
    x_vals = df[x_col].values
    bar_vals = df[bar_col].values
    line_vals = df[line_col].values

    # 绘图逻辑同 draw_one
    cmap = LinearSegmentedColormap.from_list("", clr)
    denom = line_vals.max() - line_vals.min()
    normalized = (line_vals - line_vals.min()) / denom if denom != 0 else np.zeros_like(line_vals)

    ax_target.bar(range(len(bar_vals)), bar_vals, width=0.4, color=cmap(normalized))

    ax_target2 = ax_target.twinx()
    ax_target2.plot(range(len(line_vals)), line_vals, color='crimson',
                    marker='o', markerfacecolor='none', markeredgewidth=1.5,
                    markersize=7, linewidth=1)

    # 标签与美化
    ax_target.set_xticks(range(len(x_vals)))
    ax_target.set_xticklabels(x_vals, rotation=0)
    ax_target.set_xlabel(x_col)
    ax_target.set_ylabel(bar_col)
    ax_target2.set_ylabel(line_col)
    ax_target.set_title(f'Group {g + 1}: {bar_col} vs {line_col}')

    # 颜色条
    norm = mpl.colors.Normalize(vmin=line_vals.min(), vmax=line_vals.max())
    cb = fig_merge.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax_target)
    cb.set_label(line_col, rotation=270, va='bottom')

    # 调整颜色条位置
    pos = cax_target.get_position()
    cax_target.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])

# 隐藏多余子图
for j in range(group_n, len(axes)):
    axes[j].set_visible(False)
    caxes[j].set_visible(False)

plt.savefig('merged_dual_axis_plot.png', dpi=300, bbox_inches='tight')
plt.show()