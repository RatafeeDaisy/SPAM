import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from time import time
import json
import joblib
from sklearn.model_selection import train_test_split
import scipy.io
from train import performance_report


# 日志类保持不变

class TrainerRandomForest:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_classifier(self):
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, 'model/RF_sklearn.pkl')
        training_result = self.clf.predict(self.training_data)
        print(metrics.classification_report(self.training_target, training_result))
        performance_report(self.training_target, training_result)

# 性能报告函数保持不变

def SVM_train(train_data, train_target):  # 这里可以重命名，避免与SVM混淆
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_data, train_target)
    expected = train_target
    predicted = clf.predict(train_data)

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

# 特征选择函数和select_data函数可以保持不变

if '__main__' == __name__:
    print("********************** trainning start **********************")
    t0 = time()
    takeup = 0.1
    x = scipy.io.mmread('datafile/X.mtx')
    with open('datafile/result.json', 'r') as f:
        y = json.load(f)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=takeup, random_state=20)
    print('takeup finished')

    print('###### train Random Forest model #######')
    start_time = time()
    TrainerRandomForest(train_x, train_y).train_classifier()
    print('training took %fs!' % (time() - start_time))

    modela = joblib.load('model/RF_sklearn.pkl')
    predict = modela.predict(test_x)
    performance_report(test_y, predict)
    print("*************** Trainning done in %0.3fs ***************\n\n" % ((time() - t0)))
