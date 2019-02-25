#svm手写数字识别
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#读取sklearn手写数字数据集
x = load_digits()['data']
y = load_digits()['target']

#分割训练集和测试集合
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size=0.3)

#svm线性核函数，训练
clf = SVC(kernel='linear')
clf.fit(train_x,train_y)

#预测
predict = clf.predict(test_x)

#查看结果
print(classification_report(test_y,predict))
print(confusion_matrix(test_y,predict))
