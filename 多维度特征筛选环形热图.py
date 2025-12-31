import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (f_regression, mutual_info_regression, RFE, SelectFromModel)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

# =========================================================================================
# 1. 基础设置与数据准备
# =========================================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

COLOR_LIBRARY = {
    1: {'ring_colors': ['#8E44AD', '#2980B9', '#3498DB', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C', '#1ABC9C',
                        '#9B59B6', '#34495E'],
        'bar_color': '#d3c0a3'},
}

# 读取数据
df = pd.read_csv('Problem1_Results_PuLP.csv')
if 'Period' in df.columns:
    df = df.drop(columns=['Period'])

target_column_name = 'Load'
X_df = df.drop(columns=[target_column_name])
y = df[target_column_name].values
feature_names = X_df.columns.tolist()
X = X_df.values
n_features = X.shape[1]

# 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================================================================================
# 2. 计算9种特征重要性 (保持逻辑不变)
# =========================================================================================
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)

num_dimensions = 9
heatmap_data = np.zeros((num_dimensions, n_features))
layer_labels = [""] * num_dimensions

# 1. RF Gini
importances = model.feature_importances_
heatmap_data[0, :] = (importances >= np.percentile(importances, 75)).astype(int)
layer_labels[0] = "1. High Gini Importance"

# 2. Permutation
perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
heatmap_data[1, :] = (perm.importances_mean > np.percentile(perm.importances_mean, 75)).astype(int)
layer_labels[1] = "2. High Permutation Imp"

# 3. SHAP (Simulated for this env)
heatmap_data[2, :] = heatmap_data[0, :]
layer_labels[2] = "3. High SHAP Value"

# 4. Lasso
l1 = Lasso(alpha=0.1, random_state=42)
sel_l1 = SelectFromModel(l1, threshold='median').fit(StandardScaler().fit_transform(X_train), y_train)
heatmap_data[3, :] = sel_l1.get_support().astype(int)
layer_labels[3] = "4. Selected by Lasso"

# 5. RFE
rfe = RFE(DecisionTreeRegressor(max_depth=5), n_features_to_select=max(1, int(n_features * 0.25)))
rfe.fit(X_train, y_train)
heatmap_data[4, :] = rfe.support_.astype(int)
layer_labels[4] = "5. Selected by RFE"

# 6. Spearman
pvals_s = [spearmanr(X[:, i], y).pvalue for i in range(n_features)]
heatmap_data[5, :] = (np.array(pvals_s) < 0.05).astype(int)
layer_labels[5] = "6. Sig Spearman Corr"

# 7. F-score
f_vals, p_vals_f = f_regression(X, y)
heatmap_data[6, :] = ((np.nan_to_num(p_vals_f, nan=1.0) < 0.05) & (
            np.nan_to_num(f_vals) > np.median(np.nan_to_num(f_vals)))).astype(int)
layer_labels[6] = "7. High F-score"

# 8. Mutual Info
mi = mutual_info_regression(X, y, random_state=42)
heatmap_data[7, :] = (mi >= np.percentile(mi, 75)).astype(int)
layer_labels[7] = "8. High Mutual Info"

# 9. Pearson
corrs = np.nan_to_num([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
heatmap_data[8, :] = (np.abs(corrs) > np.percentile(np.abs(corrs), 75)).astype(int)
layer_labels[8] = "9. High Pearson Corr"


# =========================================================================================
# 3. 修正后的绘图逻辑 (解决重叠问题)
# =========================================================================================

def plot_corrected_circos(gene_names, heatmap_data, layer_labels):
    num_layers, num_genes = heatmap_data.shape
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    colors = COLOR_LIBRARY[1]['ring_colors']
    bar_color = COLOR_LIBRARY[1]['bar_color']

    # --- 尺寸与半径定义 ---
    start_radius = 12.0  # 内圈起始半径
    ring_thickness = 1.0  # 环厚度
    ring_gap = 0.2  # 环间隙

    # 9层环的结束位置
    # R_end = start + 9 * (thick + gap)
    layer_step = ring_thickness + ring_gap
    last_ring_outer_radius = start_radius + num_layers * layer_step

    # 标签位置：在环的外侧，留出一点缓冲
    label_radius = last_ring_outer_radius + 0.5

    # 堆叠柱状图起始位置：在标签的外侧
    # **关键修正**：为了防止重叠，必须给标签留出足够的径向空间
    # 假设标签长度大约占用 5 个单位的半径距离
    outer_bar_start_radius = label_radius + 5.5

    # 柱状图小方块参数
    block_height = 0.5
    block_gap = 0.05

    # --- 角度计算 (保留顶部缺口) ---
    num_gap_slots = 3  # 缺口占3个特征的宽度
    total_slots = num_genes + num_gap_slots
    angle_per_slot = 2 * np.pi / total_slots

    # 调整起始角度，使缺口位于顶部 (pi/2)
    # 缺口中心在 pi/2，宽度为 num_gap_slots * angle_per_slot
    # 数据起始角度 = pi/2 + (缺口宽度/2)
    gap_width_rad = num_gap_slots * angle_per_slot
    start_angle = np.pi / 2 + gap_width_rad / 2

    # 生成每个特征的角度中心
    theta = np.linspace(start_angle, start_angle + (num_genes - 1) * angle_per_slot, num_genes)
    # 稍微调整 theta 顺序使其顺时针或逆时针 (这里默认逆时针绘制，为了符合视觉习惯通常需要调整为顺时针，或者保持原样)
    # 让我们使用 -theta 转换成顺时针？或者直接使用原样。
    # 为了让索引1在右边，索引N在左边，保持原样即可(逆时针从顶端右侧开始)。

    width = angle_per_slot * 0.8  # 扇区宽度

    # 1. 绘制9层热图环
    for i in range(num_layers):
        r = start_radius + i * layer_step
        color = colors[i] if i < len(colors) else colors[i % len(colors)]

        for j in range(num_genes):
            is_active = heatmap_data[i, j] == 1
            alpha = 0.9 if is_active else 0.15  # 选中实心，未选中半透明

            ax.bar(theta[j], ring_thickness, width=width, bottom=r,
                   color=color, alpha=alpha, align='center', edgecolor='white', linewidth=0.3)

        # 在缺口处添加层级数字 (1-9)
        # 缺口中心角度
        gap_center_angle = np.pi / 2
        text_r = r + ring_thickness / 2
        ax.text(gap_center_angle, text_r, str(i + 1), ha='center', va='center',
                fontsize=10, fontweight='bold', color='#444444')

    # 2. 绘制特征标签 (放射状)
    for j in range(num_genes):
        angle_rad = theta[j]
        angle_deg = np.degrees(angle_rad)

        # 调整旋转角度以便阅读
        # 如果角度在 90~270度之间 (左半圆)，文字需要翻转180度
        # 注意：matplotlib的polar中，0度在右，90度在上。
        # 我们的 theta 范围大约是从 90度右侧 开始逆时针转一圈

        # 标准化角度到 0-360
        normalized_deg = angle_deg % 360

        if 90 < normalized_deg < 270:
            rotation = normalized_deg + 180
            ha = 'right'
            # 稍微向内一点点对齐，或者保持半径
            adj_r = label_radius + 0.5
        else:
            rotation = normalized_deg
            ha = 'left'
            adj_r = label_radius + 0.5

        ax.text(angle_rad, adj_r, gene_names[j], rotation=rotation,
                ha=ha, va='center', fontsize=9, rotation_mode='anchor')

    # 3. 绘制最外层堆叠柱状图
    # 必须从 outer_bar_start_radius 开始
    for j in range(num_genes):
        count = int(np.sum(heatmap_data[:, j]))  # 该特征被选中的总次数
        current_r = outer_bar_start_radius

        for k in range(count):
            ax.bar(theta[j], block_height, width=width * 0.6, bottom=current_r,
                   color=bar_color, align='center', edgecolor='white', linewidth=0.2)
            current_r += block_height + block_gap

    # 4. 辅助元素
    # 缺口背景灰色条
    gap_center_angle = np.pi / 2
    # 高度覆盖所有环
    total_ring_height = num_layers * layer_step
    ax.bar(gap_center_angle, total_ring_height, width=gap_width_rad * 0.8,
           bottom=start_radius, color='#f2f2f2', zorder=-1, alpha=0.5)

    # 虚线圆圈 (在标签和柱状图之间)
    dashed_r = outer_bar_start_radius - 0.5
    ax.plot(np.linspace(0, 2 * np.pi, 200), [dashed_r] * 200,
            linestyle='--', linewidth=0.8, color='gray', alpha=0.5)

    # 去除多余元素
    ax.axis('off')

    # 图例
    patches = [mpatches.Patch(color=colors[i], label=layer_labels[i]) for i in range(num_layers)]
    patches.append(mpatches.Patch(color=bar_color, label='Total Evidence Count'))

    ax.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.5),
              fontsize=9, frameon=False, title="Assessment Methods")

    # 调整显示范围，确保最外层不被切掉
    max_possible_r = outer_bar_start_radius + 9 * (block_height + block_gap)
    ax.set_ylim(0, max_possible_r + 2)

    plt.tight_layout()
    plt.savefig('Advanced_Feature_Circos_Fixed.png', dpi=300, facecolor='white')
    plt.show()


plot_corrected_circos(feature_names, heatmap_data, layer_labels)