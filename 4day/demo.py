#svm手写数字识别
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

#读取sklearn手写数字数据集
x = load_digits()['data']
y = load_digits()['target']

#分割训练集和测试集合
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size=0.3)

#svm线性核函数，训练
# clf = SVC(kernel='linear')
# clf.fit(train_x,train_y)

#调参训练
#C表示模型对误差的惩罚系数；gamma反映了数据映射到高维特征空间后的分布，gamma越大，支持向量越多，gamma值越小，支持向量越少。
#C越大，模型越容易过拟合；C越小，模型越容易欠拟合。gamma越小，模型的泛化性变好，但过小，模型实际上会退化为线性模型；
# gamma越大，理论上SVM可以拟合任何非线性数据。
grid = GridSearchCV(SVC(kernel='linear'),param_grid={"C":[0.1,1,10],"gamma":[0.1,0.01,1]})
grid.fit(train_x,train_y)
print(grid.best_params_)
clf = SVC(kernel='linear', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
clf.fit(train_x, train_y)

#预测
predict = clf.predict(test_x)

#查看结果
print(classification_report(test_y,predict))
print(confusion_matrix(test_y,predict))
