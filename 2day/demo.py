#第二天 逻辑回归与线性回归
#导入iris数据
from sklearn.datasets import load_iris
#导入回归方法
from sklearn.linear_model import LinearRegression,LogisticRegression
#导入拆分数据集的方法
from sklearn.model_selection import train_test_split
#用于分析验证测试结果
from sklearn.metrics import confusion_matrix,classification_report

# 载入sklearn数据
iris = load_iris()
# 获取特征数据
iris_data = iris['data']
# 特征列名
columns = iris['feature_names']
# 获取签值
iris_label = iris['target']

# 拆分训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(iris_data,iris_label,test_size=0.3)
clf = LogisticRegression()
#训练
clf.fit(train_x,train_y)
#预测
predict_y = clf.predict(test_x)

print(confusion_matrix(test_y,predict_y))
print(classification_report(test_y,predict_y))

import numpy as np
from matplotlib import pyplot as plt

#生成二维的线性散点，加入一些偏移量
x = np.linspace(0, 50, 100) + 2*np.random.randn(100)
y = 2*x + 5*np.random.randn(100)
#预测需要的x值二维数组，用此方法将数组转换成二维的
x = x.reshape(-1,1)
# 上面例子有，同样是拆分训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)
# 我们先画一下训练集和测试集的数据吧
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.scatter(train_x,train_y,label='train data')
plt.legend()
plt.subplot(1,2,2)
plt.scatter(test_x,test_y,label='test data')
plt.legend()
plt.show()

# 线性回归方法
clf = LinearRegression()
# 用训练集训练
clf.fit(train_x,train_y)
# 预测测试集
predict_y = clf.predict(test_x)
# 打印出来
plt.figure(figsize=(8,6))
plt.xlim([-5,55])
plt.ylim([-10,110])
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(test_x,test_y,c='r',label='really')
plt.plot(test_x,predict_y,color='b',label='predict')
plt.legend()
plt.show()
