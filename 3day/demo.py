#第三天决策树
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

#我的例子主要是以代码实现和对其中的说明为主
#如果想了解具体的算法原理网上有很多，我也不想复制粘贴了
#这里还是用iris花作为例子
iris_data = load_iris()['data']
iris_label = load_iris()['target']

#将数据分为训练集和测试集合，test_size表示测试集锁占的比例
#得到的四个值分别是训练集的特征、测试集的特征、训练集的label、测试集的label
train_x,test_x,train_y,test_y = train_test_split(iris_data,iris_label,test_size=0.3)

#ID3和C4.5就是基于信息增益和信息增益率的，cart决策树则是基于基尼指数的
#criterion默认为gini指数，我们这里用entropy表示用信息增益
clf = tree.DecisionTreeClassifier(criterion='entropy')

#训练和预测
clf.fit(train_x,train_y)
predict_y = clf.predict(test_x)

#预测结果打印，上一篇逻辑回归和线性回归中有解释classification_report的结果说明
print(classification_report(test_y,predict_y))