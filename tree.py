import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from time import time
import json
import joblib
from sklearn.model_selection import train_test_split
import scipy.io

# 决策树训练类
class DecisionTreeTrainer:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.model = DecisionTreeClassifier(random_state=42)

    def train(self):
        self.model.fit(self.features, self.labels)
        joblib.dump(self.model, 'model/decision_tree_model.pkl')
        predictions = self.model.predict(self.features)
        print(metrics.classification_report(self.labels, predictions))
        self.report_performance(self.labels, predictions)

    def report_performance(self, true_labels, predictions):
        confusion = metrics.confusion_matrix(true_labels, predictions)
        print('Confusion Matrix:\n', confusion)

        # 计算其他性能指标（如准确率、精确度、召回率和F1值）
        # ...


def main():
    start_time = time()
    test_size = 0.1

    # 加载数据
    features = scipy.io.mmread('datafile/X.mtx')
    with open('datafile/result.json', 'r') as file:
        labels = json.load(file)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=20)

    # 训练模型
    trainer = DecisionTreeTrainer(X_train, y_train)
    trainer.train()

    # 加载模型并评估
    trained_model = joblib.load('model/decision_tree_model.pkl')
    test_predictions = trained_model.predict(X_test)
    trainer.report_performance(y_test, test_predictions)

    print(f"Training and evaluation completed in {time() - start_time:.3f} seconds.")


if __name__ == "__main__":
    main()
