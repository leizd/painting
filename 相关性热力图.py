import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import os

# 1. 数据加载与预处理
df = pd.read_csv("Problem1_Results_PuLP.csv")
# 确保数据为数值型
df = df.apply(pd.to_numeric, errors='coerce')
# 剔除标准差为0的常量列（这些列会导致相关系数为NaN）
df_clean = df.loc[:, df.std() > 0]
# 计算相关系数矩阵
corr = df_clean.corr()

# 2. 定义配色方案
palettes = [
    "RdBu_r", "PiYG", "PRGn", "BrBG", "PuOr",
    "RdYlBu", "RdYlGn", "Spectral", "coolwarm"
]

# 创建输出目录
output_dir = "correlation_plots"
os.makedirs(output_dir, exist_ok=True)

# 3. 生成综合对比图 (3x3)
fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=300)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # 遮罩上三角

for ax_idx, (ax, cmap_name) in enumerate(zip(axes.flat, palettes)):
    # --- 绘制下三角热力图 ---
    sns.heatmap(corr, mask=mask, cmap=cmap_name, center=0,
                square=True, cbar_kws={"shrink": 0.8},
                fmt=".2f", ax=ax, annot=False)  # 关闭自动标注，使用自定义标注

    # 获取配色对象，用于给文字和圆圈上色
    cmap_obj = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # --- 美化技巧 ---
    # A. 在对角线添加数值1
    for i in range(len(corr.columns)):
        ax.text(i + 0.5, i + 0.5, '1.00', ha='center', va='center',
                color='black', fontsize=10, fontweight='bold')

    # B. 在上三角添加带圆圈的相关系数值
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            # 根据值获取颜色
            color = cmap_obj(norm(value))

            # 添加数值文本
            ax.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

            # 添加空心圆突出显示
            circle = Circle((j + 0.5, i + 0.5), 0.4, fill=False,
                            edgecolor=color, linewidth=1.5)
            ax.add_patch(circle)

    # C. 添加水印和标题
    ax.text(0.95, 0.02, 'DataAnalysis', fontsize=10, color='gray',
            alpha=0.5, ha='right', va='bottom', transform=ax.transAxes)
    ax.set_title(f'Style: {cmap_name}', fontsize=12, fontweight='bold')

plt.tight_layout()
# 保存图片
plt.savefig(f'{output_dir}/corr_heatmap_all_styles.png', dpi=300, bbox_inches='tight')
plt.show()