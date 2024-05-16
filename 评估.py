import numpy as np
import json
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.io
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Load data
x = scipy.io.mmread('datafile/X.mtx')
with open('datafile/result.json', 'r') as f:
    y = json.load(f)

# Split data
takeup = 0.1
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=takeup, random_state=20)

# Load trained SVM model
model_path = 'D:\code\school\SPAM\model\decision_tree_model.pkl'
clf = joblib.load(model_path)

# Predict
predicted = clf.predict(test_x)

# Performance metrics
conf_matrix = confusion_matrix(test_y, predicted)
accuracy = accuracy_score(test_y, predicted)
test_y_int = np.array([int(label) for label in test_y])
predicted_int = np.array([int(label) for label in predicted])
recall = recall_score(test_y_int, predicted_int)
f1 = f1_score(test_y_int, predicted_int)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('决策树混淆矩阵', fontsize=18)
plt.colorbar()
tick_marks = np.arange(len(np.unique(test_y)))
plt.xticks(tick_marks, ['类别 1', '类别 2'], fontsize=12)
plt.yticks(tick_marks, ['类别 1', '类别 2'], fontsize=12)
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")
plt.ylabel('真实类别', fontsize=14)
plt.xlabel('预测类别', fontsize=14)
plt.tight_layout()
plt.savefig('决策树混淆矩阵.png')

print('模型评估完成，图表已保存。')
