# encoding=utf-8
import jieba
import codecs
import jieba.analyse
import jieba.posseg as pseg
import itertools, copy

jieba.initialize()

test_sent = [
"农业",

"工业",

["交通邮电",
"交通",
"邮电"],

["商业贸易",
"商业",
"贸易"],

"财政",

["固定资产投资", 
"基础建设", 
"更新改造",
"投资"
],

["省际物资购销",
"省际",
"物资",
"购销"],

["公共卫生", "公共","卫生"],
["文化事业", "文化"],

"国民经济",
"金融"
]
for i in range(len(test_sent)):
    filename = []
    item = ""
    if isinstance(test_sent[i], list):
        for x in range(len(test_sent[i])):
            item = test_sent[i][x].decode("utf-8")
            filename.append(item)
            jieba.add_word(item, freq=8)
    else: 
        item = test_sent[i].decode("utf-8")
        filename.append(item)
        jieba.add_word(item, freq=8) 

    txt_utf8 = open(filename[0] +".txt").read().decode("utf-8")

    words = pseg.cut(txt_utf8)
    print filename[0]
    # print type(words)

    adj = 0
    special_wd = []
    special_wd.append(0)
    special_wd.append(0)
    special_wd.append(0)

    word_count = 0.0
    for word, flag in words:
        word_count += 1.0
        # print('%s %s' % (word, flag))
        if flag == "a":
            adj += 1
        if word == filename:
            special_wd[0] += 1
        if word == "大力".decode("utf-8"):
            special_wd[1] += 1
        if word == "加强".decode("utf-8"):
            special_wd[2] += 1
    print word_count
    rate = adj/word_count
    print(u"%d 个形容词" % adj)
    print(u"占比例%f" % rate)
    print(u"%d 个'大力'" % special_wd[1])
    print(u"%d 个'加强'" % special_wd[2])
    print
