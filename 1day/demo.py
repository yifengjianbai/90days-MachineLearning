# 第一天数据预处理
from sklearn.preprocessing import Imputer,minmax_scale,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# 数据读取
data = pd.read_csv('1day/dataSource/Data.csv', header=0)
print(data)

# 缺失值处理
#方法一
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(data.iloc[:,1:3])
# data.iloc[:,1:3] = imputer.transform(data.iloc[:,1:3])
# print(data)
#方法二
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Salary'].fillna(data['Salary'].mean(), inplace=True)
print(data)

# 解析分类数据
labelEncoder = LabelEncoder()
data['Purchased'] = labelEncoder.fit_transform(data['Purchased'])
data = pd.get_dummies(data)
print(data)

# 归一化处理
data_y = data['Purchased']
data_x = data.drop(['Purchased'], axis=1)
data_x = minmax_scale(data_x)
print(data_x)

# 拆分数据
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3)
print(train_x)
print(train_y)
print(test_x)
print(test_y)