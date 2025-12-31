import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# 1. 数据加载与预处理
df_raw = pd.read_csv("Problem1_Results_PuLP.csv")


# 定义归一化函数，将任意三列转换为总和为100的三元数据
def normalize_ternary(df, cols):
    sub = df[cols].copy()
    sub['sum'] = sub.sum(axis=1)
    # 避免除以零
    sub = sub[sub['sum'] > 0]
    for c in cols:
        sub[c] = sub[c] / sub['sum'] * 100
    return sub


# 准备两组数据用于展示
data_dict = {}
# 组1: 主力机组
cols1 = ['P_Unit1', 'P_Unit2', 'P_Unit5']
if all(c in df_raw.columns for c in cols1):
    df1 = normalize_ternary(df_raw, cols1)
    df1['Size'] = df_raw.loc[df1.index, 'Load']  # 气泡大小 = 负荷
    df1['Color'] = df_raw.loc[df1.index, 'P_Unit8']  # 颜色 = Unit8出力
    # 重命名以便通用绘图
    data_dict['Main_Units'] = df1.rename(columns={'P_Unit1': 'Axis1', 'P_Unit2': 'Axis2', 'P_Unit5': 'Axis3'})

# 组2: 辅助机组
cols2 = ['P_Unit8', 'P_Unit11', 'P_Unit13']
if all(c in df_raw.columns for c in cols2):
    df2 = normalize_ternary(df_raw, cols2)
    df2['Size'] = df_raw.loc[df2.index, 'Load']
    df2['Color'] = df_raw.loc[df2.index, 'P_Unit1']
    data_dict['Aux_Units'] = df2.rename(columns={'P_Unit8': 'Axis1', 'P_Unit11': 'Axis2', 'P_Unit13': 'Axis3'})


# 2. 核心绘图函数 (Matplotlib原生实现三元图)
def plot_ternary_mpl(df, theme, ax):
    # 三元坐标变换逻辑 (Barycentric -> Cartesian)
    # A(Top), B(Right), C(Left)
    # 这里映射：Axis3 -> Top, Axis2 -> Right, Axis1 -> Left

    a = df['Axis1'].values  # 对应左侧分量
    b = df['Axis2'].values  # 对应右侧分量
    c = df['Axis3'].values  # 对应顶部分量

    # 坐标转换公式
    # x = b + c * 0.5 (水平位置)
    # y = c * sqrt(3)/2 (垂直位置)
    x = b + c * 0.5
    y = c * np.sqrt(3) / 2

    # 绘制等边三角形边界
    tri_path = [(0, 0), (100, 0), (50, 50 * np.sqrt(3)), (0, 0)]
    path = patches.Polygon(tri_path, closed=True, fill=False, edgecolor='black', lw=2)
    ax.add_patch(path)

    # 绘制网格线 (模拟 ternary 库效果)
    for i in range(1, 5):
        val = i * 20
        # 水平线 (Top分量固定)
        y_line = val * np.sqrt(3) / 2
        ax.plot([val / 2, 100 - val / 2], [y_line, y_line], 'k--', lw=0.5, alpha=0.3)
        # 左斜线 (Left分量固定)
        p1 = (100 - val, 0)
        p2 = ((100 - val) / 2, (100 - val) * np.sqrt(3) / 2)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', lw=0.5, alpha=0.3)
        # 右斜线 (Right分量固定)
        p1 = (val, 0)
        p2 = (50 + val / 2, (100 - val) * np.sqrt(3) / 2)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', lw=0.5, alpha=0.3)

    # 绘制散点
    cmap = LinearSegmentedColormap.from_list('custom', theme['cmap'], N=256)
    sc = ax.scatter(x, y, s=df['Size'], c=df['Color'], cmap=cmap, alpha=0.8, edgecolors='gray', linewidth=0.5)

    # 设置显示范围与隐藏坐标轴
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-15, 115)
    ax.set_ylim(-10, 100)

    # 添加标题与标签
    ax.set_title(theme['title'], fontsize=16, pad=20, fontweight='bold')

    # 动态添加轴标签 (旋转角度模拟真实三元图)
    ax.text(-5, 40, theme['label_l'], rotation=60, ha='right', va='center', fontsize=12, color=theme['color_l'])
    ax.text(105, 40, theme['label_r'], rotation=-60, ha='left', va='center', fontsize=12, color=theme['color_r'])
    ax.text(50, -8, theme['label_b'], ha='center', va='top', fontsize=12, color=theme['color_b'])

    # 装饰性箭头
    ax.arrow(0, 10, 10, 17.3, width=0.8, head_width=2, color=theme['color_l'])  # 左升
    ax.arrow(90, 27, 10, -17.3, width=0.8, head_width=2, color=theme['color_r'])  # 右降
    ax.arrow(35, -5, 30, 0, width=0.8, head_width=2, color=theme['color_b'])  # 底右

    return sc


# 3. 循环生成图表
color_schemes = [
    ["#287CB9", "#9FC9E1", "#E4ECF4", "#F7B494", "#DB4236"],
    ["#AC64A9", "#4CC0A0", "#EE8D25", "#045C57", "#F58E91", "#B92725"]
]

for i, (key, df_sub) in enumerate(data_dict.items()):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 配置主题
    theme = {
        'cmap': color_schemes[i % len(color_schemes)],
        'title': f'{key} Ternary Analysis',
        'label_l': df_sub.columns[0],  # 左轴变量
        'label_r': df_sub.columns[1],  # 右轴变量
        'label_b': df_sub.columns[2],  # 底轴变量
        'color_l': '#333333', 'color_r': '#333333', 'color_b': '#333333'
    }

    sc = plot_ternary_mpl(df_sub, theme, ax)

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Secondary Indicator', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()