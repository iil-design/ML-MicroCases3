import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# 新增
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ================== 数据读取 ==================
df = pd.read_excel(
    r'C:\Users\鲁迅先生\Desktop\作业\pyth\@Python大数据分析与机器学习商业案例实战\@Python大数据分析与机器学习商业案例实战\第4章 逻辑回归模型\源代码汇总_PyCharm格式\股票客户流失.xlsx'
)

# ================== 特征与目标 ==================
X = df.drop(columns='是否流失')
y = df['是否流失']

# ================== 数据标准化（SVM需要） ==================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================== 划分训练集和测试集 ==================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1
)

# ================== 逻辑回归 ==================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# ================== 随机森林 ==================
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ================== XGBoost ==================
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# ================== 支持向量机（SVM） ==================
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

# ================== 朴素贝叶斯 ==================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

# ================== 定义绘图函数 ==================
def plot_cm(y_true, y_pred, title="混淆矩阵"):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}:\n", cm)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["未流失(0)", "流失(1)"])
    plt.yticks([0, 1], ["未流失(0)", "流失(1)"])
    plt.xlabel("预测值")
    plt.ylabel("真实值")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.show()

def plot_roc(y_true, y_prob, title="ROC 曲线"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"{title} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("假阳性率 FPR")
    plt.ylabel("真正率 TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    return roc_auc

def ks_curve(y_true, y_prob, title="KS 曲线"):
    data = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    data = data.sort_values(by="y_prob", ascending=False).reset_index(drop=True)
    data["cum_event"] = np.cumsum(data["y_true"]) / data["y_true"].sum()
    data["cum_nonevent"] = np.cumsum(1 - data["y_true"]) / (len(y_true) - data["y_true"].sum())
    ks_values = data["cum_event"] - data["cum_nonevent"]
    x = np.arange(len(y_true)) / len(y_true)
    plt.figure(figsize=(6, 5))
    plt.plot(x, data["cum_event"], label="累计流失率(正类)", color="red")
    plt.plot(x, data["cum_nonevent"], label="累计未流失率(负类)", color="blue")
    plt.plot(x, ks_values, label="KS 曲线", color="green")
    ks_stat = ks_values.max()
    ks_idx = ks_values.idxmax()
    plt.axvline(ks_idx / len(y_true), color="gray", linestyle="--")
    plt.text(ks_idx / len(y_true), ks_stat, f"KS={ks_stat:.3f}", color="black")
    plt.title(title)
    plt.xlabel("累计样本比例")
    plt.ylabel("累计比例")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"{title} KS值: {ks_stat:.3f}")
    return ks_stat

# ================== 打印所有模型结果 ==================
models = [
    ("逻辑回归", y_pred_log, y_prob_log),
    ("随机森林", y_pred_rf, y_prob_rf),
    ("XGBoost", y_pred_xgb, y_prob_xgb),
    ("支持向量机", y_pred_svm, y_prob_svm),
    ("朴素贝叶斯", y_pred_nb, y_prob_nb)
]

for name, y_pred, y_prob in models:
    print(f"\n===== {name} =====")
    acc = accuracy_score(y_test, y_pred)
    print(f"预测准确率: {acc:.3f}")
    plot_cm(y_test, y_pred, title=f"{name} 混淆矩阵")
    auc_score = plot_roc(y_test, y_prob, title=f"{name} ROC 曲线")
    ks_stat = ks_curve(y_test, y_prob, title=f"{name} KS 曲线")
