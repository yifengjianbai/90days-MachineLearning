import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


def testInit(file):
    chunk = pd.read_csv('./dataSource/' + file, header=0)
    print('-'*50)
    print(chunk.head())
    apis = chunk[['api']].copy().drop_duplicates()
    tmp = get_feature(chunk, apis, False)
    tmp.to_csv('./dataSource/feature_' + file, index=False)


def init(file):
    chunk = pd.read_csv('./dataSource/' + file, header=0)
    print('-'*50)
    print(chunk.head())

    #所有api名称
    apis = chunk[['api']].copy().drop_duplicates()
    prob0 = chunk[chunk['label'] == 0]
    prob1 = chunk[chunk['label'] == 1]
    prob2 = chunk[chunk['label'] == 2]
    prob3 = chunk[chunk['label'] == 3]
    prob4 = chunk[chunk['label'] == 4]
    prob5 = chunk[chunk['label'] == 5]
    prob6 = chunk[chunk['label'] == 6]
    prob7 = chunk[chunk['label'] == 7]

    print('start get prob0 feature'+'-'*50)
    tmp0 = get_feature(prob0, apis, True)
    print('start get prob1 feature'+'-'*5)
    tmp1 = get_feature(prob1, apis, True)
    print('start get prob2 feature'+'-'*5)
    tmp2 = get_feature(prob2, apis, True)
    print('start get prob3 feature'+'-'*5)
    tmp3 = get_feature(prob3, apis, True)
    print('start get prob4 feature'+'-'*5)
    tmp4 = get_feature(prob4, apis, True)
    print('start get prob5 feature'+'-'*5)
    tmp5 = get_feature(prob5, apis, True)
    print('start get prob6 feature'+'-'*5)
    tmp6 = get_feature(prob6, apis, True)
    print('start get prob7 feature'+'-'*5)
    tmp7 = get_feature(prob7, apis, True)
    print('start concat DataFrame'+'-'*5)

    tmp = [tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7]
    result = pd.concat(tmp)
    result.to_csv('./dataSource/feature_' + file, index=False)

# 添加api名称到列


def add_api_column(df, apis):
    for index, row in apis.iterrows():
        df[row['api']] = 0
        df[row['api']+'_std'] = 0  # 文件每个线程对应api平均最小调用的标准差
        df[row['api']+'_avg'] = 0
    return df


def get_feature(prob, apis, bo=True):
    print('begin get label_prob')
    #文件对应的类型
    if bo:
        label_prob = prob[['file_id', 'label']].copy().drop_duplicates()

    print('begin get tid_prob')
    #文件对应启动线程个数
    tid_prob = prob[['file_id', 'tid']].copy().drop_duplicates()
    tid_prob['tid_count'] = 0
    tid_prob = tid_prob.groupby(by=['file_id'])[
        'tid_count'].count().reset_index()

    print('begin get api_pro')
    #文件对应调用api总次数
    api_pro = prob[['file_id', 'api']].copy()
    api_pro['api_sum_count'] = 0
    api_pro = api_pro.groupby(by=['file_id'])[
        'api_sum_count'].count().reset_index()

    #文件调用了api个数
    distinc_api_pro = prob[['file_id', 'api']].copy().drop_duplicates()
    distinc_api_pro['distinc_api_count'] = 0
    distinc_api_pro = distinc_api_pro.groupby(
        by=['file_id'])['distinc_api_count'].count().reset_index()

    #文件调用每个api的平均index和方差（分布情况）
    print('begin get avgMinMax index')
    avgMinIndex = prob[['file_id', 'tid', 'api', 'index']].copy()
    std = avgMinIndex.groupby(by=['file_id', 'tid', 'api'])['index'].std().reset_index()
    std = std.groupby(by=['file_id', 'api'])['index'].mean().reset_index()
    std.rename(columns={'index': 'std'}, inplace=True)

    avg = avgMinIndex.groupby(by=['file_id', 'tid', 'api'])['index'].mean().reset_index()
    avg = avg.groupby(by=['file_id', 'api'])['index'].mean().reset_index()
    avg.rename(columns={'index': 'avg'}, inplace=True)

    std = pd.merge(std, avg, on=['file_id', 'api'], how='inner')
    print(std)

    #暂时作废，同一线程可能被多个文件调用
    #文件对应每个线程调用api平均次数
    #文件对应每个线程调用api最少次数
    #文件对应每个线程调用api最多次数
    # avg_api_pro = prob[['file_id', 'tid', 'api']].copy()
    # avg_api_pro['tmp_count']=0
    # avg_api_pro = avg_api_pro.groupby(by=['file_id', 'tid'])['tmp_count'].count().reset_index()
    # avg_api_pro['avg_api_count']=avg_api_pro['tmp_count'].median()
    # avg_api_pro['min_api_count']=avg_api_pro['tmp_count'].min()
    # avg_api_pro['max_api_count']=avg_api_pro['tmp_count'].max()
    # print(avg_api_pro)

    print('begin get each_api_prob')
    #文件对应调用每个api的次数
    each_api_prob = prob[['file_id', 'api']].copy()
    each_api_prob['eachapi_count'] = 0
    each_api_prob = each_api_prob.groupby(by=['file_id', 'api'])['eachapi_count'].count().reset_index()
    #print(each_api_prob)

    api_prob = prob[['file_id']].copy().drop_duplicates()
    api_prob = add_api_column(api_prob, apis)
    print('add api feature')
    cnt = 1
    for index, row in std.iterrows():
        if cnt % 1000 == 0:
            print(cnt)
        api_prob.loc[api_prob['file_id'] == row['file_id'], row['api']+'_std'] = row['std']
        api_prob.loc[api_prob['file_id'] == row['file_id'], row['api']+'_avg'] = row['avg']
        cnt = cnt + 1
    cnt = 1
    for index, row in each_api_prob.iterrows():
        if cnt % 1000 == 0:
            print(cnt)
        cnt = cnt + 1
        api_prob.loc[api_prob['file_id'] == row['file_id'], row['api']] = row['eachapi_count']
    print('merge data')
    df = pd.merge(api_prob, tid_prob, on=['file_id'], how='left')
    df = pd.merge(df, api_pro, on=['file_id'], how='left')
    df = pd.merge(df, distinc_api_pro, on=['file_id'], how='left')
    #df = pd.merge(df, avg_api_pro, on=['file_id'], how='left')
    if bo:
        df = pd.merge(df, label_prob, on=['file_id'], how='left')
    return df


def train_feature_data():
    print('-'*50+'get train data')
    if os.path.exists('temp.csv'):
        temp = pd.read_csv('temp.csv')
        return temp
    else:
        feature_list = []
        for i in range(1, 15):
            df = pd.read_csv('./dataSource/feature_security_train_%s.csv' % i, header=0)
            feature_list.append(df)
        temp = pd.concat(feature_list, sort=True)
        temp.fillna(0, inplace=True)
        feature_list.clear()
        # label_id = temp[['file_id','label']].copy().drop_duplicates()
        # for index, row in label_id.iterrows():
        #     temp.loc[temp['file_id']==row['file_id'], 'label'] = row['label']
        temp['avg_api_count'] = temp['api_sum_count'].astype(
            'float')/temp['tid_count'].astype('float')
        temp['avg_distinct_api_count'] = temp['api_sum_count'].astype(
            'float')/temp['distinc_api_count'].astype('float')
        temp.to_csv('temp.csv', index=False)
        return temp


def test_feature_data():
    print('-'*50 + 'get test data')
    feature_list = []
    for i in range(1, 14):
        feature_list.append(pd.read_csv(
            './dataSource/feature_security_test_%s.csv' % i, header=0))
    temp = pd.concat(feature_list, sort=True)
    temp.fillna(0, inplace=True)
    feature_list.clear()
    temp['avg_api_count'] = temp['api_sum_count'].astype(
        'float')/temp['tid_count'].astype('float')
    temp['avg_distinct_api_count'] = temp['api_sum_count'].astype(
        'float')/temp['distinc_api_count'].astype('float')
    return temp


init('security_train_1.csv')
init('security_train_2.csv')
init('security_train_3.csv')
init('security_train_4.csv')
init('security_train_5.csv')
init('security_train_6.csv')
init('security_train_7.csv')
init('security_train_8.csv')
init('security_train_9.csv')
init('security_train_10.csv')
init('security_train_11.csv')
init('security_train_12.csv')
init('security_train_13.csv')
init('security_train_14.csv')

testInit('security_test_1.csv')
testInit('security_test_2.csv')
testInit('security_test_3.csv')
testInit('security_test_4.csv')
testInit('security_test_5.csv')
testInit('security_test_6.csv')
testInit('security_test_7.csv')
testInit('security_test_8.csv')
testInit('security_test_9.csv')
testInit('security_test_10.csv')
testInit('security_test_11.csv')
testInit('security_test_12.csv')
testInit('security_test_13.csv')

train_data = train_feature_data()
test_data_x = test_feature_data()
#只拿测试集合训练集都有的api名称来训练和预测
tmp = pd.concat([train_data[0:1], test_data_x[0:1]], join='inner')
columns_test = tmp.columns.values.tolist()
columns_train = tmp.columns.values.tolist()
columns_train = columns_train.append('label')
train_data = pd.DataFrame(train_data, columns=columns_train)
train_data_y = train_data['label']
train_data_x = train_data.drop(['label'], axis=1)
test_data_x = pd.DataFrame(test_data_x, columns=columns_test)
#train_data_x = preprocessing.scale(train_data_x)

clf = LogisticRegression()
clf.fit(train_data_x, train_data_y)

#test_data_x2 = preprocessing.scale(test_data_x)
predict = clf.predict_proba(test_data_x)
preidct_df = pd.DataFrame(predict, columns=[
                          'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])
result = pd.DataFrame(test_data_x, columns=['file_id'])
result['prob0'] = preidct_df['prob0']
result['prob1'] = preidct_df['prob1']
result['prob2'] = preidct_df['prob2']
result['prob3'] = preidct_df['prob3']
result['prob4'] = preidct_df['prob4']
result['prob5'] = preidct_df['prob5']
result['prob6'] = preidct_df['prob6']
result['prob7'] = preidct_df['prob7']
result.to_csv('finall_result.csv', index=False, float_format='%.5f')
