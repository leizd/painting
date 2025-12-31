import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
from scipy.stats import gaussian_kde

# ==============================
# 1. 数据读取与预处理
# ==============================
df = pd.read_csv('Problem1_Results_PuLP.csv')

# 将宽表转换为长表 (Melt)，以便进行分组回归
unit_ids = [1, 2, 5, 8, 11, 13]
dfs = []
for u in unit_ids:
    cols = ['Period', 'Load', f'P_Unit{u}', f'Status_Unit{u}']
    temp = df[cols].copy()
    temp.columns = ['Period', 'Load', 'Power', 'Status']
    temp['Unit'] = f'Unit {u}'
    dfs.append(temp)

long_df = pd.concat(dfs, ignore_index=True)
# 仅保留运行中的机组数据
df_clean = long_df[long_df['Status'] == 1].copy()


# ==============================
# 2. 绘图函数定义 (基于提供的框架适配)
# ==============================

def plot_group_regression(ax, df_group, color, x_key, y_key, category_key, confidence=0.95, alpha_ci=0.15):
    """绘制散点图 + 回归线 + 置信区间"""
    # 散点
    ax.scatter(df_group[x_key], df_group[y_key], color=color, s=40, alpha=0.7, edgecolor='white', linewidth=0.5)

    if len(df_group) <= 1: return None

    X = df_group[[x_key]].values
    y = df_group[y_key].values
    n = len(X)

    # 线性回归
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_hat)

    # 生成拟合线数据
    xs = np.linspace(X.min(), X.max(), 100)
    ys = model.predict(xs.reshape(-1, 1))

    # 计算置信区间
    mse = np.sum((y - y_hat) ** 2) / (n - 2)
    x_var_sum = np.sum((X - X.mean()) ** 2)
    if x_var_sum > 0:
        se = np.sqrt(mse * (1 / n + (xs - X.mean()) ** 2 / x_var_sum))
        t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
        ax.fill_between(xs, ys - t_val * se, ys + t_val * se, color=color, alpha=alpha_ci, linewidth=0)

    # 绘制回归线
    ax.plot(xs, ys, color=color, linestyle="--", linewidth=2, alpha=0.8)
    return dict(group=df_group[category_key].iloc[0], r2=r2, slope=slope, intercept=intercept, color=color)


def margin_hist(ax_top, ax_right, grouped_data, x_lims, y_lims, x_key, y_key):
    """边缘图：柱状图"""
    x_bins, y_bins = 15, 15
    for label, df_sub, color in grouped_data:
        xc, xe = np.histogram(df_sub[x_key], bins=x_bins, range=x_lims)
        yc, ye = np.histogram(df_sub[y_key], bins=y_bins, range=y_lims)
        ax_top.bar((xe[:-1] + xe[1:]) / 2, xc, width=xe[1] - xe[0], color=color, alpha=0.5)
        ax_right.barh((ye[:-1] + ye[1:]) / 2, yc, height=ye[1] - ye[0], color=color, alpha=0.5)


def margin_density(ax_top, ax_right, grouped_data, x_lims, y_lims, x_key, y_key):
    """边缘图：密度曲线 KDE"""
    x_span = np.linspace(x_lims[0], x_lims[1], 200)
    y_span = np.linspace(y_lims[0], y_lims[1], 200)
    for label, df_sub, color in grouped_data:
        try:
            if df_sub[x_key].nunique() > 1:
                kde_x = gaussian_kde(df_sub[x_key])
                ax_top.plot(x_span, kde_x(x_span), color=color, linewidth=2)
                ax_top.fill_between(x_span, kde_x(x_span), color=color, alpha=0.3)
            if df_sub[y_key].nunique() > 1:
                kde_y = gaussian_kde(df_sub[y_key])
                ax_right.plot(kde_y(y_span), y_span, color=color, linewidth=2)
                ax_right.fill_betweenx(y_span, kde_y(y_span), color=color, alpha=0.3)
        except:
            pass


def margin_boxplot(ax_top, ax_right, grouped_data, x_lims, y_lims, x_key, y_key):
    """边缘图：箱线图"""
    box_w, gap = 0.4, 1.3
    count = len(grouped_data)
    for i, (label, df_sub, color) in enumerate(grouped_data):
        offset = (i - (count - 1) / 2) * box_w * gap
        bp_props = dict(boxprops=dict(facecolor=color, alpha=0.7), medianprops=dict(color='white'))
        ax_top.boxplot(df_sub[x_key], positions=[offset], widths=box_w * 0.8, vert=False, patch_artist=True, **bp_props)
        ax_right.boxplot(df_sub[y_key], positions=[offset], widths=box_w * 0.8, vert=True, patch_artist=True,
                         **bp_props)

    limit = max(1, count * box_w * gap / 2 + 1)
    ax_top.set_ylim(-limit, limit);
    ax_top.set_yticks([])
    ax_right.set_xlim(-limit, limit);
    ax_right.set_xticks([])


def margin_violin(ax_top, ax_right, grouped_data, x_lims, y_lims, x_key, y_key):
    """边缘图：小提琴图"""
    gap = 1.2
    count = len(grouped_data)
    for i, (label, df_sub, color) in enumerate(grouped_data):
        offset = (i - (count - 1) / 2) * gap
        if len(df_sub) > 1 and df_sub[x_key].nunique() > 0:
            v = ax_top.violinplot(df_sub[x_key], positions=[offset], vert=False, showmedians=True)
            for pc in v['bodies']: pc.set_facecolor(color); pc.set_alpha(0.6)
        if len(df_sub) > 1 and df_sub[y_key].nunique() > 0:
            v = ax_right.violinplot(df_sub[y_key], positions=[offset], vert=True, showmedians=True)
            for pc in v['bodies']: pc.set_facecolor(color); pc.set_alpha(0.6)

    limit = max(2, count * gap / 2 + 1)
    ax_top.set_ylim(-limit, limit);
    ax_top.set_yticks([])
    ax_right.set_xlim(-limit, limit);
    ax_right.set_xticks([])


# ==============================
# 3. 主程序循环生成四张图
# ==============================
variants = {1: ("Histogram", margin_hist), 2: ("Density KDE", margin_density),
            3: ("Boxplot", margin_boxplot), 4: ("Violinplot", margin_violin)}
x_key, y_key, category_key = 'Load', 'Power', 'Unit'
groups = df_clean[category_key].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

for v_id, (v_name, margin_func) in variants.items():
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    grouped_data_list = []
    for idx, g_name in enumerate(groups):
        df_sub = df_clean[df_clean[category_key] == g_name]
        color = colors[idx]
        grouped_data_list.append((g_name, df_sub, color))
        plot_group_regression(ax_main, df_sub, color, x_key, y_key, category_key)

    # 设置坐标轴范围
    ax_main.set_xlim(df_clean[x_key].min() * 0.9, df_clean[x_key].max() * 1.1)
    ax_main.set_ylim(df_clean[y_key].min() * 0.9, df_clean[y_key].max() * 1.1)

    # 绘制边缘图
    margin_func(ax_top, ax_right, grouped_data_list, ax_main.get_xlim(), ax_main.get_ylim(), x_key, y_key)

    # 添加图例和标签
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l) for l, d, c in
               grouped_data_list]
    ax_main.legend(handles=handles, title=category_key, loc='upper left')
    ax_main.set_xlabel(x_key);
    ax_main.set_ylabel(y_key)
    ax_top.set_title(f"Grouped Regression with {v_name}", fontsize=14)

    plt.show()