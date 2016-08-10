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
    jieba.add_word(word, freq=8)

f = open('keywords.txt')         
# sourceInLines = f.readlines()  
#按行读出文件内容
train_data = []
train_target= []
line = "" 
for lines in f:
    
    if lines.strip() in lables:
        if line == "":
            continue
        train_data.append(line)
        train_target.append(lines.strip())
        line = ""
        
    else:
        line += lines.strip().decode('utf-8')
        
# print ''.join(train_target)
print len(train_target)
f.close()
############### format #####################
def cut_paragraph(cut_result):
    # cut_result = pseg.cut(test_sent)
    sentence_list = []
    sentence = ""
    for word, flag in cut_result:
        # print word
        sentence += word
        if  word == "。".decode("utf-8"):
            # print sentence
            sentence_list.append(sentence)
            sentence = ""
        # print(word, "/", flag, ", ", end=' ')
    return sentence_list


sentence_list = []
sentence_data_str = []
sentence_target_str  = []
for x in range(len(train_data)):
    words = pseg.cut(train_data[x])
    sentence_list = cut_paragraph(words)
    for y in range(len(sentence_list)):
        sentence_data_str.append(sentence_list[y])
        sentence_target_str.append(train_target[x])

print len(sentence_data_str)
print len(sentence_target_str)
sentence_data_str = "\n".join(sentence_data_str)
sentence_target_str = "\n".join(sentence_target_str)
# print sentence_data_str
log_f = open("sentence_data.txt","wb")
log_f.write(sentence_data_str)
log_f.close()

log_f = open("sentence_target.txt","wb")
log_f.write(sentence_target_str)
log_f.close()