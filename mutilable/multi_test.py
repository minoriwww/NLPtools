# -*- coding:utf8 -*- 
import numpy
import numpy as np
import jieba
import sys
import re
import io
import pprint
import random
import pylab as pl
from time import time
from sklearn.pipeline import Pipeline
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

reload(sys) 
sys.setdefaultencoding('utf-8')

label = [
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

for word in label:
    jieba.add_word(word, freq=10)

train_data = []
train_target= []

with io.open('newkeywords.txt', 'r', encoding='utf8') as f:       
    # sourceInLines = f.readlines()  
    #按行读出文件内容

    line = "" 
    for lines in f:
        if lines.strip().split(" ")[0] in label:
            if line == "":
                continue
            lables_in_sentence = lines.encode('utf-8').strip().split(" ")

            
            train_data.append(line)
            # train_data.append(unicode(line))
            train_target.append(lables_in_sentence)
            line = ""
            
        else:
            line += lines.encode('utf-8').strip()

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

lables_utf8 = [i.encode('utf-8') for i in list(set(label))]

##################################################################################
# 没有encoder直接binarizer会报错：multilable应使用稀疏矩阵或01向量。。。
le = preprocessing.LabelEncoder()
le.fit(y_train_text)
print le.classes_
lable_number = le.transform(y_train_text)

le = preprocessing.LabelEncoder()
le.fit(y_train_text)
print le.classes_
lable_number = le.transform(y_train_text)

############## choose one : LabelBinarizer###################
# lb = preprocessing.LabelBinarizer()
# Y = lb.fit_transform(lable_number)
################ choose one : MultiLabelBinarizer#################
pre_lb_fit_trans = []
for x in lable_number:
    pre_lb_fit_trans.append([x])
print pre_lb_fit_trans
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(pre_lb_fit_trans)
Y = MultiLabelBinarizer().fit_transform(y_train_text)
##########################################################
comma_tokenizer = lambda x: jieba.cut(x, cut_all=True,  HMM=False)

classifier = Pipeline([
    # ('vectorizer', HashingVectorizer(tokenizer = comma_tokenizer, non_negative=True)),
    ('vectorizer', CountVectorizer(tokenizer = comma_tokenizer, lowercase=False)),
    # ('bin', preprocessing.MultiLabelBinarizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01)))
    # ('clf', MultinomialNB(alpha=0.0001))
    ])


classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = le.inverse_transform(predicted)
lable_result = ''
for item, labels in zip(X_test, all_labels):
    for i in range(len(labels)):
        lable_result += "," .join(labels[i])
    lable_result += '\n'+'*'*30
    print '%s => %s' % (item, lable_result )
    lable_result = ''

# for item, labels in zip(X_test, predicted):
#     print '%s => %s' % (item, ', '.join(target_names[x] for x in labels))

