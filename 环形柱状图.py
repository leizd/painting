# ===================== 01 环境准备 =====================
# 请确保已安装库: pip install pycirclize matplotlib numpy pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from pycirclize import Circos


# ===================== 00 数据预处理 (新增) =====================
# 将原始宽表转换为框架所需的 [Group, Label, Value] 长表格式
def prepare_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # 选择要展示的机组作为“组”
    target_units = ['P_Unit1', 'P_Unit2', 'P_Unit8']

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'Label', 'Value'])  # 写入表头

        for unit_col in target_units:
            # 简化组名，例如 P_Unit1 -> Unit1
            group_name = unit_col.replace('P_', '')
            for _, row in df.iterrows():
                label = str(row['Period'])  # 标签为时段
                value = row[unit_col]  # 数值为功率
                writer.writerow([group_name, label, value])

    print(f"数据已转换并保存至: {output_csv}")


# 执行数据准备
input_file = 'Problem1_Results_PuLP.csv'  # 您的原始文件
data_file = 'circos_data.csv'  # 转换后的中间文件
prepare_data(input_file, data_file)

# ===================== 03 全局参数配置 =====================
# 配置绘图参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

# 设置参数 (根据您的数据范围调整)
output_file = 'circos_plot.png'
vmin = 0.0  # 最小功率 0
vmax = 120.0  # 最大功率约 110，留出余量
baseline = 0.0
positive_color = '#DC6F74'  # 您的粉色参数
negative_color = '#6B63B2'
sector_colors = ["#F2F2F270", "#FFFFFF70"]  # 扇区背景交替
sector_gap = 10  # 增加间隙以便区分不同机组

# ===================== 04 加载并检查数据 =====================
groups = []
labels = []
values = []
with open(data_file, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    for row in reader:
        if len(row) >= 3:
            groups.append(row[0])
            labels.append(row[1])
            values.append(float(row[2]))

print(f"Successfully loaded data: {len(groups)} data points")
print(f"Value range: {min(values):.3f} to {max(values):.3f}")

# ===================== 05 创建扇区 =====================
# 步骤 1：提取组名
unique_groups = list(dict.fromkeys(groups))
num_groups = len(unique_groups)

# 步骤 2：计算扇区大小
group_counts = {g: groups.count(g) for g in unique_groups}
sectors = {g: group_counts[g] for g in unique_groups}

# 步骤 3：初始化 Circos 环形布局
# 调整 end 角度，留出缺口展示图例或标题，或者设为 360 形成闭环
# 这里根据 sector_gap 动态调整，保留一个大约 40 度的开口
space = [sector_gap] * num_groups
circos = Circos(sectors, space=space, start=0, end=360 - sector_gap * num_groups)

# 步骤 4：扇区交替配色
sector_colors_dict = {g: sector_colors[i % 2] for i, g in enumerate(unique_groups)}

# ===================== 06 创建轨道 =====================
for idx, sector in enumerate(circos.sectors):
    group_name = sector.name
    group_size = group_counts[group_name]

    # 5.1 提取当前组数据
    group_indices = [i for i, g in enumerate(groups) if g == group_name]
    group_labels = [labels[i] for i in group_indices]
    group_values = [values[i] for i in group_indices]

    # 5.2 计算坐标与颜色
    x = np.arange(sector.start, sector.end) + 0.5
    bar_colors = [positive_color if v > baseline else negative_color for v in group_values]

    # 5.3 创建主轨道 (绘制条形图)
    # 轨道范围设置为半径 40 到 90
    track = sector.add_track((40, 90), r_pad_ratio=0.1)
    track.axis(fc=sector_colors_dict[group_name], ec="none")
    track.grid()

    # 绘制基线
    full_x = np.linspace(sector.start, sector.end, 100)
    track.line(full_x, [baseline] * len(full_x), color='black', lw=0.5)

    # 绘制条形图
    track.bar(x, group_values, bottom=baseline, width=0.6,
              vmin=vmin, vmax=vmax, color=bar_colors)

    # 5.4 添加 x 轴标签 (Period)
    pos_list = [i + 0.5 for i in range(group_size)]
    # 每隔3个显示一个标签，避免拥挤
    shown_indices = [i for i in range(len(group_labels)) if (i + 1) % 3 == 0 or i == 0]
    shown_pos = [pos_list[i] for i in shown_indices]
    shown_labels = [group_labels[i] for i in shown_indices]

    track.xticks(
        shown_pos, shown_labels,
        outer=True, tick_length=0, label_margin=2,
        line_kws=dict(color="#000000", lw=0),
        label_orientation="vertical",
        text_kws=dict(size=8, color="#000000", weight='bold')
    )

    # -----------------------------------------------------------
    # 【修改点】：使用 r 参数代替 y 参数
    # 轨道外径是 90，所以我们将 r 设为 95，让文字显示在轨道外侧
    # -----------------------------------------------------------
    track.text(group_name, x=(sector.start + sector.end) / 2, r=95,
               size=14, weight='bold', color='black')

    # 5.5 统一 y 轴刻度 (仅在第一个扇区)
    if idx == 0:
        first_sector_y = [0, 30, 60, 90, 120]
        first_sector_y_labels = [str(val) for val in first_sector_y]
        track.yticks(first_sector_y, first_sector_y_labels, vmin=vmin, vmax=vmax,
                     side="left", line_kws=dict(color="black", lw=1),
                     text_kws=dict(color="black", size=10, weight='bold'))

# ===================== 07 保存图像 =====================
circos.savefig(output_file, dpi=600)
print(f"图已保存至：{output_file}")
plt.show()  # 如果在 Notebook 中运行