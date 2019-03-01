# 文章分类预测
import numpy as np
from sklearn.datasets import fetch_20newsgroups


news = fetch_20newsgroups()
print(news.target_names)

"""
    2、Extracting features from text files 从文本文件中提取特征
    为了在文本文件中使用机器学习算法，首先需要将文本内容转换为数值特征向量
"""

"""
    Bags of words 词袋
    最直接的方式就是词袋表示法
        1、为训练集的任何文档中的每个单词分配一个固定的整数ID（例如通过从字典到整型索引建立字典）
        2、对于每个文档，计算每个词出现的次数，并存储到X[i,j]中。

    词袋表示：n_features 是语料中不同单词的数量，这个数量通常大于100000.
    如果 n_samples == 10000，存储X的数组就需要10000*10000*4byte=4GB,这么大的存储在今天的计算机上是不可能实现的。
    幸运的是，X中的大多数值都是0，基于这种原因，我们说词袋是典型的高维稀疏数据集，我们可以只存储那些非0的特征向量。
    scipy.sparse 矩阵就是这种数据结构，而scikit-learn内置了这种数据结构。
"""

"""
    Tokenizing text with scikit-learn 使用scikit-learn标记文本
    文本处理、分词、过滤停用词都在这些高级组件中，能够建立特征字典并将文档转换成特征向量。
"""
# sklearn中的文本特征提取组件中，导入特征向量计数函数
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
train_x = vect.fit_transform(news.data)
print(train_x)
print(train_x.shape)
print('-----')

"""
    CountVectorizer支持计算单词或序列的N-grams，一旦合适，这个向量化就可以建立特征词典。
    在整个训练预料中，词汇中的词汇索引值与其频率有关。
"""
print(vect.vocabulary_.get(u'algorithm'))
print('-----')
"""
    From occurrences to frequencies 从事件到频率
    计数是一个好的开始，但是也存在一个问题：较长的文本将会比较短的文本有很高的平均计数值，即使他们所表示的话题是一样的。
    为了避免潜在的差异，它可以将文档中的每个单词出现的次数在文档的总字数的比例：这个新的特征叫做词频：tf
    tf-idf:词频-逆文档频率
"""
from sklearn.feature_extraction.text import TfidfTransformer  # sklearn中的文本特征提取组件中，导入词频统计函数

tf_transformer = TfidfTransformer()
print(tf_transformer)  # 输出函数属性 TfidfTransformer(norm=u'l2', smooth_idf=True, sublinear_tf=False, use_idf=False)
print('-----')
X_train_tf = tf_transformer.fit_transform(train_x)  # 使用函数对文本文档进行tf-idf频率计算
print(X_train_tf)
print('-----')
print(X_train_tf.shape)
print('-----')
"""
    Training a classifier 训练一个分类器
    既然已经有了特征，就可以训练分类器来试图预测一个帖子的类别，先使用贝叶斯分类器，贝叶斯分类器提供了一个良好的基线来完成这个任务。
    scikit-learn中包括这个分类器的许多变量，最适合进行单词计数的是多项式变量。
"""
from sklearn.naive_bayes import MultinomialNB  # 使用sklearn中的贝叶斯分类器，并且加载贝叶斯分类器

# 中的MultinomialNB多项式函数
clf = MultinomialNB()  # 加载多项式函数
x_clf = clf.fit(train_x, news.target)  # 构造基于数据的分类器
print(x_clf)  # 分类器属性：MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print('-----')
"""
    为了预测输入的新的文档，我们需要使用与前面相同的特征提取链进行提取特征。
    不同的是，在转换中，使用transform来代替fit_transform，因为训练集已经构造了分类器
"""
docs_new = ['sci.crypt', 'soc.religion.christian']  # 文档
X_new_counts = vect.transform(docs_new)  # 构建文档计数
X_new_tfidf = tf_transformer.transform(X_new_counts)  # 构建文档tfidf
predicted = clf.predict(X_new_tfidf)  # 预测文档
print(predicted)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, news.target_names[category]))  # 将文档和类别名字对应起来
print('-----')
"""
    Building a pipeline 建立管道
    为了使向量转换更加简单(vectorizer => transformer => classifier)，scikit-learn提供了pipeline类来表示为一个复合分类器
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge'))])
text_clf = text_clf.fit(news.data, news.target)
print(text_clf)  # 构造分类器，分类器的属性
print('-----')

test = fetch_20newsgroups(subset='test')
predict = text_clf.predict(test.data)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(test.target,predict))
print(confusion_matrix(test.target,predict))