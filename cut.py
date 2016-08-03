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

# for word in test_sent:
#     print word.decode("utf-8")
#     jieba.add_word(word.decode("utf-8"), freq=8)


txt_utf8 = open('yunnan_10_5.txt').read().decode("utf-8")


words = pseg.cut(txt_utf8)




# 由于关键字分布均较集中
# 10句号以上 认为离开关键字区域。

# 向后查找10句 如果10句以内有此关键字
# 则打印到关键字出现的那一句 继续向后遍历

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
# 向后查找6句 如果6句以内有此关键字
# 则打印到关键字出现的那一句 继续向后遍历
# para: whole words after cut, key word name (have decode yet)
# return: string
def generate_txt(words, key_name):
    print "in generate_txt"
    generate_str = ""
    c = iter(words)
    it = iter(words)
    c_str = ""

    try:
        # 全局向后查找关键字
        while c_str != None:

            it, c = itertools.tee(it, 2)
            it_str = it.next()
            c_str = c.next()

            #每10个一组
            sentence_num = 0
            has_keywd = False
            while sentence_num < 7:

                # 如果有此关键字 这6句都包括进字符串
                if it_str.word == key_name:
                    has_keywd = True
                    # if sentence_num == 0 : generate_str += it_str.word
                    sentence_num = 0
                # 没有关键字继续向后 遇见"。"则+1
                elif it_str.word == "。".decode("utf-8"): 
                    sentence_num += 1

                
                if has_keywd:
                    # 执行字符串拼接
                    generate_str += it_str.word
                it_str = it.next()

                # else: it.next()
    except StopIteration:
        print "end of words"
        pass

    print generate_str
    return generate_str


for i in range(len(test_sent)):
    filename = test_sent[i].decode("utf-8")
    # jieba.add_word(filename, freq=8)
    words, newwords = itertools.tee(words, 2)
    keywd_txt = generate_txt(newwords, filename)

    log_f = open(filename+".txt","wb")
    log_f.write(keywd_txt.encode('utf-8'))
    log_f.close()





'''
# 关键词提取 50名和权重
# tags = jieba.analyse.extract_tags(txt_utf8, topK=50, withWeight=withWeight)

# if withWeight is True:
#     for tag in tags:
#         print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))
# else:
#     print(",".join(tags))
'''