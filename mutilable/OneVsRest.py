# -*- coding:utf8 -*- 
import numpy
import numpy as np
import jieba
import sys
import re
import io
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer

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
            # lables_in_sentence = unicode(lines).strip().split(" ")
            lables_in_sentence = lines.encode('utf-8').strip().split(" ")
            # for i in lables_in_sentence : print i
            # print "-"*10
            # lables_in_sentence_str += lables_in_sentence[i]
            
            train_data.append(line)
            # train_data.append(unicode(line))
            train_target.append(lables_in_sentence)
            line = ""
            
        else:
            line += lines.encode('utf-8').strip()
        
# print ''.join(train_target)
# for i in train_target : 
#     for j in i:
#         print j
# for i in train_data:print i
# print len(train_data), len(train_target)  
# print train_target
    
######################################################################################
'''

X_train = np.array(["“九五”时期是我省经济社会发展较快、较好的时期之一。",
                    "在继续抓好森林资源保护和造林绿化的同时,重点发展特色经济林、生物化工原料林、笋材两用竹林、速生丰产用材林、珍贵用材林、短周期工业原料林。",
                    "经济结构不合理,劳动者科技文化素质偏低和基础设施滞后等制约经济社会发展的深层次矛盾依然突出。",
                    "加强对农业产业化经营的规划和引导,有重点地扶持建设一批农业产业化经营示范项目。",
                    "全省呈现出经济发展、社会进步、民族团结和边疆稳定的大好局面,为实施“十五”计划奠定了坚实基础。",
                    "促进经济作物向适宜种植区集中,优化农业区域布局。",
                    "经济结构不合理,劳动者科技文化素质偏低和基础设施滞后等制约经济社会发展的深层次矛盾依然突出。",
                    "搞好热区农业综合开发和冬季农业开发,巩固提高烤烟、甘蔗、茶叶、橡胶等传统经济作物,加速发展新兴经济作物。",
                    "过去的五年,面对错综复杂的国际国内环境,省委、省政府坚持以邓小平理论为指导,认真贯彻党中央、国务院的一系列方针、政策,按照省第六次党代会提出的“以经济效益为中心,打基础、兴科教、调结构、建支柱,促进经济社会协调发展”的思路,坚持解放思想、实事求是的思想路线和改革开放的总方针,立足省情,发展特色经济,抓住国家实行积极的财政政策、扩大内需的机遇,大力推进两个根本性转变,坚持用创新的思路、改革发展的办法解决前进中的问题,改革开放和现代化建设都取得了令人瞩目的成就。",
                    "第二节调整农业产业和产品结构坚持稳粮调结构、提质增效益,引导农民根据市场变化,自主调整种植、养殖结构。",
                    "同时,还存在着一些困难和问题,主要是:农业基础薄弱,农民增收缓慢,农村贫困面较大;工业企业技术创新能力不强,产品竞争力较弱;各项改革相对滞后,体制障碍仍较明显;市场体系不健全,流通不畅;非公有制经济发展不充分,比重小;企业职工下岗分流人员增多,再就业压力加大;地方财政收入增幅减缓,财政平衡难度增大;投融资机制不活,民间投资增长缓慢;城镇化水平低,城市功能不完善;高层次技术人才和管理人才匮乏。",
                    "生态恶化的趋势仍未根本遏制。"
                    ])
print X_train[0]

y_train_text = [["历史回顾"],["农业"],["历史回顾"],["农业"],["历史回顾"],
                ["农业"],["历史回顾"],["农业"],["历史回顾"],["农业"],
                ["农业","历史回顾"],["农业","历史回顾"]]
print y_train_text

X_test = np.array(['我国经过二十多年改革和快速发展,形成了比较雄厚的物质技术基础和有利的体制环境;国家正实施西部大开发战略,我国即将加入世界贸易组织。',
                   '促进经济作物向适宜种植区集中,优化农业区域布局。',
                   '主要宏观调控目标如期完成,国民经济持续快速健康发展。',
                   '改善农业生产条件,优化农村生态环境,提高农民生活质量。',
                   ])
# target_names = ['New York', 'London']
'''

data = []
for x in range(len(train_data)):
    data.append([train_data[x], train_target[x]])

random.shuffle(data)

ratio = int(0.8 * len(train_data) )
X_train = np.array([each[0] for each in data[:ratio]])
y_train_text = [each[1] for each in data[:ratio]]
X_test = np.array([each[0] for each in data[ratio:]])
y_test_text = [each[1] for each in data[ratio:]]
print len(X_test), len(y_test_text)
# print X_train[0]
# print y_train_text 

##################################################################################
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)
# Y = MultiLabelBinarizer().fit_transform(y_train_text)
##########################################################
comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)
# seg_list = comma_tokenizer(X_train[0]) 
# print(", ".join(seg_list))

classifier = Pipeline([
    ('vectorizer', HashingVectorizer(tokenizer = comma_tokenizer, n_features = 10000, non_negative = True)),
    # ('vectorizer', preprocessing.MultiLabelBinarizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.0001)))
    # ('clf', MultinomialNB(alpha=0.0001))
    ])

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

class_fit = classifier.fit(X_train, Y)

predicted = class_fit.predict(X_test)
print predicted
all_labels = lb.inverse_transform(predicted)
temp_train = classifier.named_steps['vectorizer'].fit_transform(X_train.tolist())

pred = []
prob = classifier.named_steps['clf'].predict_proba(temp_train)
print prob.shape
for i in range(len(y_test_text)):
    temp = []
    for j in range(len(classifier.named_steps['clf'].classes_)):
        if (prob[i][j] == prob[i].max() or prob[i][j] >= 0) and prob[i].max()-prob[i][j] <= 0.02:
            temp.append(classifier.named_steps['clf'].classes_[j])
    pred.append(temp)
print 'max number of tags:'+ str(numpy.array([len(i) for i in pred]).max())
print 'min number of tags:'+ str(numpy.array([len(i) for i in pred]).min())

# result_str = ''
# for i in range(len(X_test)):
#     result_str += X_test[i]+'\n'
#     for j in range(len(pred[i])):
#         # result_str += classifier.named_steps['tfidf'].transform(pred, copy=True)[i][j]       
#         result_str += pred[i][j].decode('utf-8') + '   '
#     result_str += '\n\n'

print classifier.named_steps['clf'].get_params()
# print result_str.decode('utf-8')

# predicted = class_fit.predict_proba
# print predicted
# all_labels = lb.inverse_transform(predicted)

# for item, labels in zip(X_test, all_labels):
#     print u'%s => %s' % (item, ', '.join(labels))
# print len(X_test), type(all_labels)
