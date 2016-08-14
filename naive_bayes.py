# -*- coding:utf8 -*-
import sys
import numpy
import jieba
import codecs
import jieba.analyse
import jieba.posseg as pseg
import itertools, copy
import random
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer

reload(sys) 
sys.setdefaultencoding('utf-8')
'''
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
'''
##########################################
'''
f = open('target.txt')         
# sourceInLines = f.readlines()  
#按行读出文件内容
train_target = []      
line = ""                             
for line in f:
    train_target.append(line)

print  len(train_target)
f.close()
'''


f = open('C:\\Users\\8cm\\Desktop\\sentence_data.txt')
# sourceInLines = f.readlines()  
#按行读出文件内容
train_data = []      
line = ""                             
for line in f:
    train_data.append(line)

# print  len(train_target)
f.close()


f = open('C:\\Users\\8cm\\Desktop\\sentence_target.txt')
# sourceInLines = f.readlines()  
#按行读出文件内容
train_target = []      
line = ""                             
for line in f:
    train_target.append(line)

print  len(train_target)
f.close()

################# format data #####################
# def cut_paragraph(cut_result):
#     # cut_result = pseg.cut(test_sent)
#     sentence_list = []
#     for word, flag in cut_result:
#         print word
#         sentence_list.append(word)
#         # print(word, "/", flag, ", ", end=' ')
#     return sentence_list

# data = []
# for x in range(len(train_data)):
#     words = pseg.cut(train_data[x])
#     cut_paragraph(words)

data = []
for x in range(len(train_data)):
    data.append([train_data[x], train_target[x]])

random.shuffle(data)
##############################################
comma_tokenizer = lambda x: jieba.cut(x, cut_all = True)

def vectorize(train_words, test_words):
    v = HashingVectorizer(tokenizer = comma_tokenizer, n_features = 30000, non_negative = True)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    return train_data, test_data


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred)
    m_recall = metrics.recall_score(actual, pred)
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)


def train_clf(train_data, train_tags):
    clf = MultinomialNB(alpha=0.0001) #adjust it !!!!!!!
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


def pick(clf,test_data,test_words,multi_label=False,critical_value=0,difference=1):
    pred = []
    prob = clf.predict_proba(test_data)
    if multi_label == False:
        pred = clf.predict(test_data)
    elif multi_label == True:
        for i in range(len(test_words)):
            temp = []
            for j in range(len(clf.classes_)):
                if (prob[i][j] == prob[i].max() or prob[i][j] >= critical_value) and prob[i].max()-prob[i][j] <= difference:
                    temp.append(clf.classes_[j])
            pred.append(temp)
    print 'max number of tags:'+ str(numpy.array([len(i) for i in pred]).max())
    print 'min number of tags:'+ str(numpy.array([len(i) for i in pred]).min())
    return pred

pred=pick(clf,test_data,test_words,multi_label=True,critical_value=0,difference=0.02)
result_str = ''
for i in range(len(test_words)):
    result_str += test_words[i]+'\n'
    for j in range(len(pred[i])):
        result_str += pred[i][j] + '   '
    result_str += '\n\n'

f = open('C:\\Users\\8cm\\Desktop\\multi_label.txt', "wb")
f.write(result_str)
f.close()



'''
pred = clf.predict(test_data)

# print "".join(test_tags)
test_tags = numpy.array(test_tags)
# print test_tags.tolist()
# print pred.tolist()
right_num = (test_tags == pred).sum()
print(right_num)
print len(test_tags),  len(pred.tolist()), len(test_words)
# print test_tags
evaluate(numpy.asarray(test_tags), pred)

result_str = ""
pred_list = pred.tolist()
for x in range(len(test_tags)):
    result_str += pred_list[x] + "    " + test_tags[x] + "\n" + test_words[x] + "\n"
# print result_str

log_f = open('C:\\Users\\8cm\\Desktop\\test.txt',"wb")
log_f.write(result_str)
log_f.close()
'''


