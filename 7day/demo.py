# 工业蒸汽量预测
# 具体赛题地址
# https://tianchi.aliyun.com/competition/entrance/231693/introduction
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

#读取txt
def getFormat(file):
    str = ''
    with open(file + '.txt', 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            str = str + lines.replace('\t', ',') + '\n'
    return str

#重写txt为csv
def writeFile(file):
    data = getFormat(file)
    with open(file + '.csv', 'w') as f:
        f.write(data)

#将txt文件转化为csv文件格式
writeFile('7day/dataSource/zhengqi_train')
writeFile('7day/dataSource/zhengqi_test')

#读取csv
train = pd.read_csv('7day/dataSource/zhengqi_train.csv', header=0)
columns = train.shape[1]

#查看数据信息
print(train.head())
train_x = train.ix[:,0:columns-1]
train_y = train['target']

#在拆分训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(train_x,train_y,test_size=0.3)
print(type(train_x))
print(type(train_y))

#训练预测
clf = SGDRegressor()
clf.fit(train_x,train_y)
predict = clf.predict(test_x)

#打印测试结果
plt.plot(test_y,test_y,linewidth=2,color='r')
plt.scatter(test_y,predict)
plt.xlabel('really')
plt.ylabel('predict')
plt.xlim((-4,4))
plt.ylim((-4,4))
plt.show()

#用正式的测试数据进行预测
#并且生成相关结果到txt
test_x = pd.read_csv('7day/dataSource/zhengqi_test.csv', header=0)
predict = clf.predict(test_x)
predict = pd.DataFrame(predict)
predict.to_csv('7day/dataSource/result.txt', index=False)