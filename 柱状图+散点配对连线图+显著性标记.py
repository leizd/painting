import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# 01 导入库并设置全局参数
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif'] # 优先使用Arial
plt.rcParams['font.size'] = 14

# 02 读取数据
# 使用您上传的文件
data = pd.read_csv('Problem1_Results_PuLP.csv')
# 选取 Unit 1 和 Unit 2 作为对比组
group1 = data['P_Unit1'].values
group2 = data['P_Unit2'].values
print(f"成功读取数据: Unit 1有{len(group1)}个样本, Unit 2有{len(group2)}个样本")

# 03 独立样本 t 检验与显著性标记转换
# 进行独立样本t检验
t_stat, p_value = ttest_ind(group1, group2, equal_var=False) # Unit 1方差较大，建议关闭方差齐性假设
print(f"Unit 1 vs Unit 2：t统计量 = {t_stat:.3f}，p值 = {p_value:.3f}")

def p_value_to_stars(p_value):
    if p_value < 0.0001:
        return '****'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

stars = p_value_to_stars(p_value)
print(f"显著性标记：{stars}")

# 04 绘制基础图
mean1, mean2 = np.mean(group1), np.mean(group2)
labels = ['Unit 1', 'Unit 2']
x_pos = np.arange(len(labels))
width = 0.6
bar_color = ['#9AC9C920', '#9AC9C9']
hatch_patterns = ['//', '']

plt.figure(figsize=(5, 6))

# 绘制柱状图
plt.bar(x_pos, [mean1, mean2], width,
        color=bar_color,
        edgecolor='#9AC9C9',
        linewidth=1.5,
        hatch=hatch_patterns,
        zorder=1)

# 绘制散点图
# 稍微调整透明度以免遮挡线条
plt.scatter([x_pos[0]] * len(group1), group1, color='black', alpha=0.6, s=30, zorder=3)
plt.scatter([x_pos[1]] * len(group2), group2, color='black', alpha=0.6, s=30, zorder=3)

# 连接配对数据点
for i in range(len(group1)):
    plt.plot([x_pos[0], x_pos[1]],
             [group1[i], group2[i]],
             color='gray', alpha=0.3, linewidth=0.8, zorder=2)

# 设置显著性标记的位置
y_max = max(np.max(group1), np.max(group2))
base_y = y_max * 1.05 # 动态设置高度

# 绘制显著性标记的横线和竖线
line_h = y_max * 0.02
plt.plot([x_pos[0], x_pos[0]], [base_y - line_h, base_y], color='black', linewidth=1.25)
plt.plot([x_pos[1], x_pos[1]], [base_y - line_h, base_y], color='black', linewidth=1.25)
plt.plot([x_pos[0], x_pos[1]], [base_y, base_y], color='black', linewidth=1.25)
plt.text(x=0.5, y=base_y + line_h, s=stars, ha='center', va='bottom', fontsize=16)

# 05 修改细节
plt.ylabel('Power Output (MW)', fontsize=16)
# 动态调整Y轴范围
plt.ylim(0, base_y * 1.15)
plt.xlim(-0.8, 1.8)

# 设置刻度
ax = plt.gca()
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=14, fontweight='bold')

# 隐藏上轴和右轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.tick_params(axis='both', which='major', width=1.5, labelsize=12)

# 调整布局
plt.tight_layout()

# 保存
plt.savefig('paired_plot_units.png', dpi=600)
plt.show()