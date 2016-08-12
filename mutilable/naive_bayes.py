# -*- coding:utf8 -*- 
import sys
import numpy
import numpy as np
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
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

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
f = open('sentence_data.txt')         
# sourceInLines = f.readlines()  
#按行读出文件内容
train_data = []      
line = ""                             
for line in f:
    train_data.append(line)

# print  len(train_target)
f.close()


f = open('sentence_target.txt')         
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

# log_f = open("result_str.txt","wb")
# log_f.write(result_str)
# log_f.close()
####################################################################


X_train = np.array(["推进农业产业化经营推广“公司+基地+农户”、“订单农业”、贸工农一体化等多种经营模式,发展绿色农业和创汇农业。",
                    "加大科技引进与合作力度实行国内合作与国外合作并举,“引进来”与“走出去”并重,进一步强化与国内外有实力、有优势的高等学校、科研机构和企业的科技合作。",
                    "营造技术创新环境建立健全地方性科技法规、规章体系,研究制定鼓励技术创新的激励政策,建立健全工业、农业科技创新、技术推广和技术培训体系,形成省、地、县、乡(社区)四级科技服务网络。",
                    
                    "加快建立和完善农业科技创新、技术推广体系,依托有关科研院所和高校,对关系农业和农村经济发展的重大应用技术进行研究和开发。",
                    "鼓励加快与国际知名烟草企业合作,利用国外先进技术和知名品牌合作生产卷烟,提高经营管理水平。",
                    "引导烟草行业提高信息化水平、开发能力和装备水平。",
                    "积极探索各种现代营销方式和手段,努力巩固和开拓国内外卷烟市场。加快研究开发适应国内外消费需求的混合型卷烟等新产品。全力争取建设造纸法薄片厂。烤烟种植向适宜区集中,继续控制面积和产量,提高质量和效益。到2005年,烤烟产量控制在61万吨以上,卷烟产量达到600万箱以上。",
                    "建立人才、科技创新、资金、中介组织服务、政策、法律六大支撑体系,实施基础设施建设、良种、试验示范、绿色通道、信息和市场开拓六大工程,努力实现生物资源开发跨越传统发展模式。",
                    "抓好中国云南野生生物种质资源库、中华生物谷、国家中药现代化科技产业(云南)基地和昆明国际花卉拍卖市场建设。把我省建成亚洲最大的花卉生产出口基地、全国最大的生物资源开发创新基地,为建成绿色经济强省奠定坚实的产业基础。到2005年,力争年产值达到800亿元左右。",
                    "引导烟草行业提高信息化水平、开发能力和装备水平。",
                    "加快发展生物资源开发创新产业依靠科技,突出特点,引进与开发并举,改造与培植并重,加快蔗糖、茶叶、天然橡胶、畜牧和水产养殖、林产、以天然药物为主的现代医药、绿色保健食品、花卉及绿化园艺、生物化工等产业的发展。",
                    "努力发展科技、文化教育事业,促进科技进步,形成全省科技创新、开发和应用技术推广中心。",
                   "加速科技型中小企业和民营科技企业的发展。",
                    "全社会研究与开发经费占国内生产总值的比重达到1%以上。"])
y_train_text = [["农业"],["科技研究"],["科技研究"],["科技研究", " 农业"],["农业"],["科技研究", " 农业"],["农业"],
                ["科技研究", " 农业"],["农业"],["农业"],["科技研究", " 农业"],["科技研究"],
                ["科技研究"],["科技研究"]]

X_test = np.array(['发展高新技术产业按照有所为、有所不为的原则,突出比较优势和特色,确定有限目标,重点培育现代生物医药、电子信息、新材料、机光电一体化等高新技术产业。中心城市要充分发挥技术、信息和人才聚集的优势,加快发展高新技术产业,逐步形成高新技术产业的局部优势和跨越式发展。鼓励企业与科研院校结合,建立高新技术研究的开发主体,高起点建设一批高新技术重点实验室、工程研究中心,加快高新技术产业化进程。抓好优质钾盐、贵金属功能材料基地等重大工程。组织实施电子信息、生物与医药和新材料等产业化,以及机光电一体化、优势资源增值转化、传统产业升级和科技创新能力建设等项目,形成一批高新技术企业集团和名牌产品,加快高新技术产业组织重组,培育有特色的新兴产业群。昆明、玉溪、曲靖、大理等地区要成为高新技术产业的聚集区,带动和辐射周边地区高新技术产业发展。抓好文山三七、楚雄医药和个旧大屯科技等工业园区建设。到2005年,力争高新技术产业产值达到600亿元以上。',
                   '努力建设全国最大的低危害烟草科研和生产基地。',
                   '加强研究开发与技术创新推进新的农业科技革命,加强信息技术、医药、生物、新材料、先进制造、优势资源开发、生态环境治理等领域的应用开发研究。建立经科教、产学研相结合的技术创新体制和机制。鼓励企业建立研究开发机构,加强与国内外高等院校和科研机构联合研究开发。加大全社会科技投入,引导企业成为研究开发的主体。重大科技项目探索由政府资助、科研机构和企业共同投资方式,联合攻关。通过自主创新和引进相结合,加快对结构升级的共性、关键和配套技术的开发。每年重点推广一批新技术、新工艺、新成果,加快高新技术、先进适用技术向支柱产业和优势产业渗透,提高重点、骨干产业技术水平,带动产业结构升级。推广信息技术,提高企业综合管理和生产自动化水平。鼓励支持具有比较优势的基础研究和应用基础研究。重视发展和繁荣哲学社会科学,促进自然科学与社会科学的交叉融合,加快社会科学发展和理论创新。'
                   ])
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)
# Y = MultiLabelBinarizer().fit_transform(y_train_text)
classifier = Pipeline([
    ('vectorizer', HashingVectorizer(tokenizer = comma_tokenizer, n_features = 30000, non_negative = True)),
    ('tfidf', TfidfTransformer()),
    # ('clf', OneVsRestClassifier(LinearSVC())),
    ('clf',  MultinomialNB(alpha=0.0001))
    ])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print '%s => %s' % (item, ', '.join(labels))