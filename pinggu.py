import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, \
    roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import json
from scipy.io import mmread

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 加载数据集和标签
x = mmread('datafile/X.mtx')
with open('datafile/result.json', 'r') as f:
    y = json.load(f)

# 分割数据集
takeup = 0.1
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=takeup, random_state=20)

# 加载模型
model_paths = ['D:\\code\\school\\SPAM\\model\\RF_sklearn.pkl',
               'D:\\code\\school\\SPAM\\model\\SVM_sklearn.pkl',
               'D:\\code\\school\\SPAM\\model\\decision_tree_model.pkl']
models = []
for path in model_paths:
    models.append(joblib.load(path))

# 评估模型并绘制评估图
for i, model in enumerate(models):
    # 预测概率
    y_prob = model.predict_proba(test_x)[:, 1]

    # 转换测试标签为整数类型
    y_true = np.array(test_y).astype(int)

    # 混淆矩阵
    cm = confusion_matrix(test_y, model.predict(test_x))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'模型{i + 1} 混淆矩阵')
    plt.colorbar()
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig(f'model_{i + 1}_confusion_matrix.png')

    # 精度-召回曲线
    precision, recall, _ = precision_recall_curve(test_y, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精度')
    plt.title(f'模型{i + 1} 精度-召回曲线')
    plt.savefig(f'model_{i + 1}_precision_recall_curve.png')

    # F1分数-召回曲线
    f1 = np.array([f1_score(test_y, np.where(y_prob > threshold, 1, 0)) for threshold in np.arange(0, 1, 0.01)])
    recall_thresholds = np.arange(0, 1, 0.01)
    plt.figure()
    plt.plot(recall_thresholds, f1, color='blue', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('F1分数')
    plt.title(f'模型{i + 1} F1分数-召回曲线')
    plt.savefig(f'model_{i + 1}_f1_recall_curve.png')
