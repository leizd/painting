import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与预处理 (针对您的数据)
# ==========================================
df = pd.read_csv("Problem1_Results_PuLP.csv")

# --- 关键步骤：构造适合分类任务的目标变量 ---
# 由于原数据中 Status 全为 1，无法分类，这里我们以"负荷是否高于平均值"作为分类目标
# 1 代表高负荷，0 代表低负荷
df['target'] = (df['Load'] > df['Load'].mean()).astype(int)

# --- 特征选择 ---
# 选取机组出力 (P_Unit) 作为特征，去除 Load (因为它是目标的来源) 和 Status (全为常量)
cols_to_drop = ['target', 'Load', 'Period'] + [c for c in df.columns if 'Status' in c]
X = df.drop(cols_to_drop, axis=1)
y = df['target']

print(f"数据形状: {X.shape}")
print(f"正样本比例: {y.mean():.2f}")

# 划分训练集和测试集
# 注意：由于总共只有 24 条数据，划分后测试集非常小(约7-8条)，仅做代码演示用
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# 2. 模型训练与评估
# ==========================================
# 初始化模型
models = {
    'LR': LogisticRegression(max_iter=1000),
    'RF': RandomForestClassifier(random_state=42),
    'XGB': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LGBM': LGBMClassifier(random_state=42, verbose=-1)
}


# 训练并计算指标
def get_metrics(model, X_set, y_set):
    y_pred = model.predict(X_set)
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_set)[:, 1]
            auc = roc_auc_score(y_set, y_prob)
        except:
            auc = 0.5  # 如果测试集只有一类样本，AUC可能无法计算
    else:
        auc = 0.5

    return [
        round(accuracy_score(y_set, y_pred), 4),
        round(precision_score(y_set, y_pred, zero_division=0), 4),
        round(recall_score(y_set, y_pred, zero_division=0), 4),
        round(f1_score(y_set, y_pred, zero_division=0), 4),
        round(auc, 4)
    ]


# 准备绘图数据
quadrants_test = []
quadrants_train = []
angles_map = {'LR': 90, 'RF': 0, 'XGB': 270, 'LGBM': 180}
bg_colors = {'LR': "#eafaf1", 'RF': "#fff0f5", 'XGB': "#eef8fc", 'LGBM': "#eef8fc"}

print("开始训练模型...")
for name, model in models.items():
    model.fit(X_train, y_train)

    # 记录测试集指标
    quadrants_test.append({
        "name": name,
        "data": get_metrics(model, X_test, y_test),
        "scale_max": 1.0,
        "center_angle": angles_map[name],
        "bg_color": bg_colors[name]
    })

    # 记录训练集指标
    quadrants_train.append({
        "name": name,
        "data": get_metrics(model, X_train, y_train),
        "scale_max": 1.0,
        "center_angle": angles_map[name],
        "bg_color": bg_colors[name]
    })


# ==========================================
# 3. 极坐标绘图函数
# ==========================================
def plot_polar_chart(quadrants, save_path):
    bar_colors = ['#0077b6', '#90e0ef', '#daeaf6', '#ffe5d9', '#f4a261']
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    num_metrics = len(categories)

    for quad in quadrants:
        center_angle_rad = np.deg2rad(quad["center_angle"])
        spread_rad = np.deg2rad(70)  # 扇区展开角度

        # 1. 绘制背景扇区
        ax.bar(center_angle_rad, quad["scale_max"], width=np.deg2rad(90), bottom=0,
               color=quad["bg_color"], alpha=0.3, zorder=0)

        # 2. 绘制指标柱
        angles = np.linspace(center_angle_rad - spread_rad / 2, center_angle_rad + spread_rad / 2, num_metrics)
        width = spread_rad / (num_metrics + 1)

        for i, (val, ang) in enumerate(zip(quad["data"], angles)):
            label = categories[i] if quad["name"] == quadrants[0]["name"] else None
            ax.bar(ang, val, width=width, color=bar_colors[i], zorder=5, label=label)

        # 3. 标注模型名称
        ax.text(center_angle_rad, quad["scale_max"] + 0.12, quad["name"],
                ha='center', va='center', fontweight='bold', fontsize=14)

    # 设置样式
    ax.set_ylim(0, 1.15)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)

    # 自定义网格圈
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.text(np.deg2rad(45), r, str(r), color='gray', fontsize=9)

    # 图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # 排序
    ord_h = [by_label[c] for c in categories if c in by_label]
    ord_l = [c for c in categories if c in by_label]

    ax.legend(ord_h, ord_l, loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存: {save_path}")
    plt.show()


# 生成图表
plot_polar_chart(quadrants_test, "MyData_Test_Polar.pdf")
plot_polar_chart(quadrants_train, "MyData_Train_Polar.pdf")