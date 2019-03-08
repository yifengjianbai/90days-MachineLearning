# 工业蒸汽量预测 tensorflow
# 具体赛题地址
# https://tianchi.aliyun.com/competition/entrance/231693/introduction

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split,KFold
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib


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

#读取csv文件
def read_file(type='train'):
    #将txt文件转化为csv文件格式
    writeFile('7day/dataSource/zhengqi_train')
    writeFile('7day/dataSource/zhengqi_test')

    #读取csv
    path = pathlib.Path("7day/dataSource/zhengqi_train.csv")
    if not path.is_file():
        writeFile('7day/dataSource/zhengqi_train')
        writeFile('7day/dataSource/zhengqi_test')
    if type=='train':
        data = pd.read_csv('7day/dataSource/zhengqi_train.csv', header=0)
        columns = data.shape[1]
        #查看数据信息
        print(data.head())
        data_x = np.array(data.ix[:,0:columns-1])
        data_y = data['target']
        return data_x,data_y
    else:
        data = pd.read_csv('7day/dataSource/zhengqi_test.csv', header=0)
        columns = data.shape[1]
        #查看数据信息
        print(data.head())
        data_x = np.array(data)
        return data_x
    
#自定义神经网络层
def add_layer(X, input_size, output_size, active_func=None):
    W = tf.Variable(tf.random_normal([input_size,output_size]))
    b = tf.Variable(tf.zeros([output_size])+0.1)
    y = tf.matmul(X,W) + b
    if active_func is None:
        return y
    elif active_func is not None:
        return active_func(y)

def run_func(train_x,train_y,test_x,test_y=None):
    '''
    如果test_y为None，则只进行结果预测，并返回预测结果
    如果test_y不为None，则返回验证集上的loss
    '''
    x = tf.placeholder(tf.float32,[None,38])
    y = tf.placeholder(tf.float32,[None,1])
    # layer1 = add_layer(x,38,10,tf.nn.relu)
    # predict = add_layer(layer1,10,1)
    l1 = tf.layers.dense(x, 100, tf.nn.relu)
    predict = tf.layers.dense(l1, 1)
    loss = tf.losses.mean_squared_error(y,predict)
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-predict), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(0,train_x.shape[0],50):
            x_batch,y_batch = train_x[i:i+50],train_y[i:i+50]
            # 增加维度
            # 这里需要传入的一个二维数组[[0],[1],[2]]
            # 而我们实际的数据是[0,1,2]
            # 利用np.newaxis可以增加维度
            # [0,1,2][:,np.newaxis]=>[[0],[1],[2]]
            # [0,1,2][np.newaxis,:]=>[[0,1,2]]
            y_batch = y_batch[:,np.newaxis]
            sess.run(train_step,feed_dict={x:x_batch,y:y_batch})
            print(sess.run(loss,feed_dict={x:x_batch,y:y_batch}))

        if test_y is None:
            t_predict = sess.run(predict,feed_dict={x:test_x})
            return t_predict
        else:
            #用测试集来验证loss
            print('loss in test data is :')
            test_y = test_y[:,np.newaxis]
            test_loss = sess.run(loss,feed_dict={x:test_x,y:test_y})
            print(test_loss)
            # y_ = sess.run(predict,feed_dict={x:test_x})
            # plt.scatter(test_y,y_)
            # plt.plot(test_y,test_y,color='r')
            # plt.xlabel('really')
            # plt.ylabel('predict')
            # plt.xlim((-4,4))
            # plt.ylim((-4,4))
            # plt.show()
            return test_loss

#在拆分训练集和测试集，出留法
def method1():
    '''
    用出留法进行预测和模型评估
    '''
    data_x,data_y = read_file()
    train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.3)
    run_func(train_x,train_y,test_x,test_y)

#交叉验证
def method2():
    '''
    用交叉验证法进行预测和模型评估
    '''
    data_x,data_y = read_file()
    #n_splits表示交叉的折数
    kf = KFold(n_splits=5,shuffle=True)
    for train_index,test_index in kf.split(data_x):
        train_x,train_y = data_x[train_index],data_y[train_index]
        test_x,test_y = data_x[test_index],data_y[test_index]
        run_func(train_x,train_y,test_x,test_y)
    test_x = read_file(type='test')
    t_predict = run_func(data_x,data_y,test_x)
    t_predict = pd.DataFrame(t_predict)
    t_predict.to_csv('7day/dataSource/result.txt', index=False, header=False)

method2()