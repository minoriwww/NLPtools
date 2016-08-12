# encoding=utf-8
import sys
import re
import io
import jieba
import chardet
import codecs
import jieba.analyse
import jieba.posseg as pseg
import itertools, copy
import json
from pyltp import *

reload(sys) 
sys.setdefaultencoding('utf-8')
segmentor = Segmentor()  # 初始化实例
segmentor.load('..\\..\\tools-master\\ltp_data\\cws.model')  # 加载模型
labeller = SementicRoleLabeller()
labeller.load('..\\..\\tools-master\\ltp_data\\srl')
postagger = Postagger() # 初始化实例
postagger.load('..\\..\\tools-master\\ltp_data\\pos.model')
parser = Parser() # 初始化实例
parser.load('..\\..\\tools-master\\ltp_data\\parser.model')  # 加载模型
recognizer = NamedEntityRecognizer()
recognizer.load('..\\..\\tools-master\\ltp_data\\ner.model')

words = segmentor.segment("")
postags = postagger.postag(words)
arcs = parser.parse(words, postags)  # 句法分析

for i in range(len(arcs)): 
    if arcs[i].relation in ["VOB","SBV", "IOB", "POB", "COO", "HED"] and postags[i] == "n":
        print words[i].decode('utf-8'), arcs[i].relation, postags[i]

# s = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
s = "不得不说锤子手机在很多功能操作上的优化真的很用心，尤其是一些 看上去并没有什么卵用但让人感觉确实舒服的小设计。"
# for x, w in jieba.analyse.extract_tags(s, withWeight=True, HMM=True):
#     print('%s %s' % (x, w))

# print('-'*40)
# textrank are poor for new words and frency
# for x, w in jieba.analyse.textrank(s, withWeight=True, HMM=True):
#     print('%s %s' % (x, w))

withWeight = False
tags = jieba.analyse.extract_tags(s, topK=1, withWeight=withWeight)
print(tags[0] + "/")
# if withWeight is True:
#     for tag in tags:
#         print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))
# else:
#     print(",".join(tags))

# print('/'.join(jieba.cut(s, HMM=True)))

# all_the_text = open('small.txt').read()
# print all_the_text.decode('utf-8')
  
f = open('opendata_20w.txt')
# sourceInLines = f.readlines()  
#按行读出文件内容
new = []
tempcount = 0
# with io.open('small.txt', 'r', encoding='unicode') as f:
for line in f:
    # new.append(line.decode('utf-8'))
    # print line.decode('utf-8')
    # words = segmentor.segment(line)   
    # postags = postagger.postag(words)
    # arcs = parser.parse(words, postags)  # 句法分析
    # # print "reading"
    # for i in range(len(arcs)):        
    #     if arcs[i].relation == "SBV" and postags[i] == "n":
    #     # if arcs[i].relation in ["VOB","SBV", "IOB", "POB", "COO", "HED"] and postags[i] == "n":
    #         print i, words[i].decode('utf-8'), arcs[i].relation, postags[i]
    #         tempcount += 1
    #         print tempcount
    #         break
    #     elif arcs[i].relation == "VOB" and postags[i] == "n":
    #         print i, words[i].decode('utf-8'), arcs[i].relation, postags[i]
    #         tempcount += 1
    #         print tempcount
    #         break
    #     else:
    #         # print i
    #         pass

    new.append(line)
            
    # print ("".join(new))
f.close()

print (len(new))


with io.open('my.txt', 'w', encoding='utf8') as json_file:
    count = 0
    for line in new:
        # counter
        count += 1
        tag_str = ""

        tags = jieba.analyse.extract_tags(line, topK=5, withWeight=withWeight)
        # print (line.decode('utf-8'))
        words = segmentor.segment(line)
        postags = postagger.postag(words)
        arcs = parser.parse(words, postags)  # 句法分析
        # print "".join(words).decode('utf-8')
        # print "".join(tags).decode('utf-8')
        for i in range(len(arcs)): 
            if arcs[i].relation in ["VOB","SBV","IOB", "POB", "COO", "HED"] and postags[i] == "n":
                if words[i].decode('utf-8') not in tags :
                    tag_str = words[i].decode('utf-8')
                    # print "in"
                    break
            else: 
                tag_str = tags[0] 
                break
        if count%100 == 0:
            print count

        
        
        # tag_str = ",".join(tags)

        # print (tag_str)
        dic = {
             "content": line.strip(),
             "core_entity": tag_str
           }
        # print chardet.detect(dic)
        # result_json = json.dumps(dic)
        # print (chardet.detect(result_json))
        # print (chardet.detect(result_json.decode("ascii")))
        # file = 'my.json'
        # fp = open(file,'a', encoding='utf-8')
        # print json.dumps(dic,  encoding='utf-8', ensure_ascii=False)
        # fp.write(json.dumps(dic,  encoding='utf-8', ensure_ascii=False))
        # fp.close()
        
        data = json.dumps(dic, encoding='utf-8', ensure_ascii=False)
        # unicode(data) auto-decodes data to unicode if str
        json_file.write(unicode("["))
        json_file.write(unicode(data))
        json_file.write(unicode("]\n"))

