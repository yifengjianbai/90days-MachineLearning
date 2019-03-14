# 工业蒸汽量预测
# 具体赛题地址
# https://tianchi.aliyun.com/competition/entrance/231693/introduction
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neural_network as nn
from sklearn.decomposition import pca
import matplotlib.pyplot as plt


def my_fit(train_x,train_y,test_x,test_y=None):
    #训练预测
    #clf = nn.MLPRegressor(hidden_layer_sizes=(40,),activation='identity',alpha=0.1)
    clf = SGDRegressor()
    clf.fit(train_x,train_y)
    predict = clf.predict(test_x)
    if test_y is not None:
        #打印测试结果
        # plt.plot(test_y,test_y,linewidth=2,color='r')
        # plt.scatter(test_y,predict)
        # plt.xlabel('really')
        # plt.ylabel('predict')
        # plt.xlim((-4,4))
        # plt.ylim((-4,4))
        # plt.show()
        print(np.sum(np.square(test_y-predict))/test_y.shape[0])
    else:
        predict = pd.DataFrame(predict)
        predict.to_csv('7day/dataSource/result.txt', index=False, header=False)

def kfold_method():
    '''
    用交叉验证法进行预测和模型评估
    '''
    #读取csv
    train = pd.read_csv('7day/dataSource/zhengqi_train.txt', header=0, sep='\t')
    columns = train.shape[1]
    #查看数据信息
    print(train.head())
    data_x = np.array(train.ix[:,0:columns-1])
    data_y = train['target']
    #降维处理
    # pca = pca.PCA(n_components=0.95)
    # train_x = pca.fit_transform(train_x)
    kf = KFold(n_splits=5,shuffle=True,random_state=2019)
    for train_index,test_index in kf.split(data_x):
        train_x,train_y = data_x[train_index],data_y[train_index]
        test_x,test_y = data_x[test_index],data_y[test_index]
        my_fit(train_x,train_y,test_x,test_y)
    test_xx = pd.read_csv('7day/dataSource/zhengqi_test.txt', header=0, sep='\t')
    my_fit(data_x,data_y,test_xx)

kfold_method()