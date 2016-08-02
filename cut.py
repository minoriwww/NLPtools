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

"交通",
"邮电",

"商业",
"贸易",

"财政",

"固定资产投资",

"省际",
"物资",
"购销",

"公共卫生",
"文化事业",
"国民经济",
"金融"
]

for word in test_sent:
    print word.decode("utf-8")
    jieba.add_word(word.decode("utf-8"), freq=8)


txt_utf8 = open('yunnan_10_5.txt').read().decode("utf-8")

# seg_list = jieba.cut(txt_utf8, HMM=False)  # 默认是精确模式 开启hmm
# seg_list = " ".join(jieba.cut(txt_utf8, HMM=False))
# print(", ".join(seg_list))

# log_f = open("1.log","wb")
# log_f.write(seg_list.encode('utf-8'))
# log_f.close()

words = pseg.cut(txt_utf8)

# print it.next().word
# tempstr = ["a", "b", "c", "d", "e"]
# it = iter(tempstr)
# it_str = ""
# try:
#     while it_str != None:
#         it_str = it.next()
#         print it_str
# except Exception:
#     pass


# 由于关键字分布均较集中
# 10句号以上 认为离开关键字区域。

# 向后查找10句 如果10句以内有此关键字
# 则打印到关键字出现的那一句 继续向后遍历
# it = iter(words)

# print it.next()
# print it.next()
# temp_it, it = itertools.tee(it, 2)
# # temp_it = it
# print temp_it.next()
# print temp_it.next()
# print it.next()

# def generate_txt(words, test_sent):
#     generate_str = ""
#     c = iter(words)
#     c_str = ""

#     try:
#         while c_str != None:
#             it_str = c.next()
#             # print it_str

#             if it_str.word == test_sent[0].decode("utf-8"):
#                 print it_str.flag    
#                 it, c = itertools.tee(c, 2)
#                 surrend_wd = it.next().word
#                 generate_str += it_str.word
#                 while surrend_wd  != "。".decode("utf-8"):
#                     # print surrend_wd
#                     generate_str += surrend_wd
#                     surrend_wd = it.next().word
                
#                 print "end inner iter"
#                 # else: it.next()
                

#     except StopIteration:
#             pass

#     print generate_str
# 向后查找10句 如果10句以内有此关键字
# 则打印到关键字出现的那一句 继续向后遍历
def generate_txt(words, test_sent):
    generate_str = ""
    c = iter(words)
    c_str = ""

    try:
        while c_str != None:
            it_str = c.next()
            # print it_str

            sentence_num = 0
            while sentence_num < 10:
                # 如果有此关键字
                if it_str.word == test_sent[0].decode("utf-8"):
                    sentence_num = 0
                # 没有关键字继续向后 遇见"。"则+1
                elif it_str.word == "。".decode("utf-8"): 
                    sentence_num += 1
                     
                it, c = itertools.tee(c, 2)
                surrend_wd = it.next().word
                generate_str += it_str.word
                while surrend_wd  != "。".decode("utf-8"):
                    # print surrend_wd
                    generate_str += surrend_wd
                    surrend_wd = it.next().word
                
                print "end inner iter"
                # else: it.next()
    except StopIteration:
            pass

    print generate_str

generate_txt(words, test_sent)

# for w in words:
#     # print w.word, w.flag
#     print type(w)
#     if w.word == test_sent[0].decode("utf-8"):
#         print w.flag
#         sentence_iter = w
#         while sentence_iter.next().word != "。".decode("utf-8"):
#             print sentence_iter.word


withWeight = True




'''
# 关键词提取 50名和权重
# tags = jieba.analyse.extract_tags(txt_utf8, topK=50, withWeight=withWeight)

# if withWeight is True:
#     for tag in tags:
#         print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))
# else:
#     print(",".join(tags))
'''