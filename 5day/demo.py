#神经网络
import pandas as pd
from sklearn import neural_network as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# 导入sklearn手写数字的数据
x = load_digits()['data']
y = load_digits()['target']

# 拆分训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)

# 用神经网络训练，solver:lbfgs牛顿法，sgd随机梯度下降法，adam基于随机梯度下降的优化器
clf = nn.MLPClassifier(solver='lbfgs')
clf.fit(x,y)

# 预测
predict = clf.predict(test_x)

# 打印结果
print(classification_report(test_y,predict))
print(confusion_matrix(test_y,predict))