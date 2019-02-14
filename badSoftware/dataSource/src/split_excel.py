# 按照file_id对文件进行分块
import numpy as np
import pandas as pd
import os

def get_distinct_file_id(kinds):
    df = pd.read_csv('./dataSource/security_%s.csv' % kinds, header=0, usecols=[0])
    files_id = df[['file_id']].copy().drop_duplicates()
    del df
    return files_id

def split_csv(kinds):
    print('get all files id')
    files_id = get_distinct_file_id(kinds)
    chunks = []
    i = 1
    print('all files id count is : %s' % files_id.shape[0])
    for chunk in pd.read_csv('./dataSource/security_%s.csv' % kinds, header=0, chunksize=10000000):
        cnt = 1
        j = 1
        for index, row in files_id.iterrows():
            if cnt > 1000:
                print('save csv %s' % j)
                pd.concat(chunks).to_csv('./dataSource/security_%s_%s_%s.csv' % (kinds, i, j), index=False)
                j = j + 1
                cnt = 0
                chunks.clear()
            chunks.append(chunk[chunk['file_id'] == row['file_id']])
            cnt = cnt + 1
        print('save csv %s' % j)
        pd.concat(chunks).to_csv('./dataSource/security_%s_%s_%s.csv' % (kinds, i, j), index=False)
        chunks.clear()
        i = i + 1

def concat_csv(kinds):
    for i in range(1,14):
        chuncks = []
        for j in range(1,9):
            chunck = pd.read_csv('./dataSource/security_%s_%s_%s.csv' % (kinds, j, i), header=0)
            chuncks.append(chunck)
        pd.concat(chuncks).to_csv('./dataSource/security_%s_%s.csv' % (kinds, i), index=False)
        chuncks.clear()

# temp = pd.read_csv('temp.csv', header=0)
# columns = temp.columns.values.tolist()
# for column in columns:
#     if column.find('std') > -1:
#         temp.drop([column], axis=1, inplace=True)
# temp.to_csv('temp2.csv', index=False)
a = pd.DataFrame([[1,2,1],[1,2,3],[1,2,5],[1,3,12],[1,3,14],[1,3,16]], columns=list('ABC'))
a = a.groupby(by=['A','B'])['C'].std().reset_index()
a.rename(columns={'C':'cc'}, inplace=True)
print(a)