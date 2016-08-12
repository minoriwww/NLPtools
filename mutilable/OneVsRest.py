# -*- coding:utf8 -*- 
import numpy
import numpy as np
import jieba
import sys
import re
import io
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB

reload(sys) 
sys.setdefaultencoding('utf-8')

lables = [
"农业",

"工业",

"交通邮电",

"商业贸易",

"财政",

"固定资产投资", 

"省际物资购销",

"公共卫生", 
"文化事业",

"国民经济",
"金融",
"历史回顾",
"方针思想",
"人才培育",
"法制建设",
"环境保护",
"产业结构",
"海外投资",
"企业改革",
"土地改革",
"服务业",
"私营经济",
"科技研究"
]

for word in lables:
    jieba.add_word(word, freq=10)

train_data = []
train_target= []

# f = open('newkeywords.txt')  
with io.open('newkeywords.txt', 'r', encoding='utf8') as f:       
    # sourceInLines = f.readlines()  
    #按行读出文件内容

    line = "" 
    for lines in f:
        # if lines.strip() in lables:
        if lines.strip().split(" ")[0] in lables:
            if line == "":
                continue
            lables_in_sentence = lines.strip().split(" ")
            # print lables_in_sentence
            # lables_in_sentence_str += lables_in_sentence[i]
            
            train_data.append(line)
            train_target.append(lables_in_sentence)
            # train_target.append(lines.strip())
            # train_target.append(lines.strip().decode('utf-8'))
            line = ""
            
        else:
            line += lines.strip()
        
# print ''.join(train_target)
# for i in train_target : 
#     for j in i:
#         print j
# for i in train_data:print i
# print len(train_data), len(train_target)  
# print train_target
    
######################################################################################
'''
X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"])
# print X_train

y_train_text = [["new york"],["new york"],["new york"],["new york"],["new york"],
                ["new york"],["london"],["london"],["london"],["london"],
                ["london"],["london"],["new york","london"],["new york","london"]]

X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'london is rainy',
                   'it is raining in britian',
                   'it is raining in britian and the big apple',
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too'])
# target_names = ['New York', 'London']
'''

ratio = int(0.8 * len(train_data) )
X_train = np.array(train_data)[:ratio]
y_train_text = train_target[:ratio]
X_test = np.array(train_data)[ratio:]
y_test_text = train_target[ratio:]
# print X_train
##################################################################################
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)
# Y = MultiLabelBinarizer().fit_transform(y_train_text)
##########################################################
comma_tokenizer = lambda x: jieba.cut(x, HMM=False)

classifier = Pipeline([
    ('vectorizer', HashingVectorizer(tokenizer = comma_tokenizer, n_features = 30000, non_negative = True)),
    # ('vectorizer', preprocessing.MultiLabelBinarizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

#######################################################################################
# def vectorize(train_words, test_words, v = HashingVectorizer(tokenizer = comma_tokenizer, n_features = 30000, non_negative = True)):    
#     train_data = v.fit_transform(train_words)
#     test_data = v.fit_transform(test_words)
#     return train_data, test_data

# # ndarray, ndarray, clf
# def train_clf(train_data, train_tags, clf = MultinomialNB(alpha=0.0001)):
#     #adjust alpha !!!!!!!
#     clf.fit(train_data, train_tags)
#     return clf
    
# # 4 ndarray
# train_data, test_data = vectorize(train_words = X_train.tolist(), test_words = X_test.tolist(), v = preprocessing.MultiLabelBinarizer())
# y_train_text, y_test_text = vectorize(train_words = y_train_text, test_words = y_test_text, v = preprocessing.MultiLabelBinarizer())

# clf = train_clf(train_data = train_data, train_tags = y_train_text, clf = OneVsRestClassifier(LinearSVC()))
# pred = clf.predict(X_test)

#############################################################################
classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)
# all_labels = lb.inverse_transform(pred)

for item, labels in zip(X_test, all_labels):
    print '%s => %s' % (item, ', '.join(labels))
