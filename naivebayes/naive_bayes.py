# -*- coding:utf8 -*- 
import sys
import numpy
import jieba
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer
import re
import codecs
import itertools, copy
f = open('data.txt')         
# sourceInLines = f.readlines()  
#按行读出文件内容
train_data = []      
line = ""                             
for lines in f:
    
    if lines.strip().decode('utf-8') == "----------":
        if line == "":
            continue
        train_data.append(line)
        line = ""
        
    else:
        line += lines.strip().decode('utf-8')
        
# print ''.join(new)
print len(train_data)
f.close()
##########################################
f = open('target.txt')         
# sourceInLines = f.readlines()  
#按行读出文件内容
train_target = []      
line = ""                             
for line in f:
    train_target.append(line)
        
# print ''.join(train_target)
# print  train_target
f.close()
######################################
data = []
for x in range(len(train_target)):
    data.append([train_data[x], train_target[x]])

random.shuffle(data)

comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)


def vectorize(train_words, test_words):
    v = HashingVectorizer(tokenizer=comma_tokenizer, n_features=30000, non_negative=True)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    return train_data, test_data


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred)
    m_recall = metrics.recall_score(actual, pred)
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)


def train_clf(train_data, train_tags):
    clf = MultinomialNB(alpha=0.001) #adjust it !!!!!!!
    clf.fit(train_data, numpy.asarray(train_tags))
    return clf


# 训练集和测试集的比例为7:3 -> ratio=0.7
ratio = 0.8
filesize = int(ratio * len(train_target))
train_words = [each[0] for each in data[:filesize]]
train_tags = [each[1] for each in data[:filesize]]
test_words = [each[0] for each in data[filesize:]]
test_tags =[each[1] for each in data[filesize:]]



# train_words, train_tags, test_words, test_tags = input_data(train_file, test_file)
train_data, test_data = vectorize(train_words, test_words)
# print train_data, test_data
clf = train_clf(train_data, train_tags)
pred = clf.predict(test_data)
# print "".join(pred)
length = len(test_tags)
test_tags = numpy.array(test_tags)
# print test_tags.tolist()
# print pred.tolist()
right_num = (test_tags == pred).sum()
print(right_num)
print length
# print test_tags
evaluate(numpy.asarray(test_tags), pred)