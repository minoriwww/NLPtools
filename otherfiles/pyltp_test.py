# -*- coding: utf-8 -*-
from pyltp import *

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

# words = segmentor.segment('正确处理好改革力度、发展速度和人民群众承受程度之间的关系,进一步加强民主法制建设,坚持依法治省,维护司法公正,强化社会治安综合治理,切实维护社会稳定,为我省改革开放和现代化建设创造良好的社会环境。')  # 分词
words = segmentor.segment('把千方百计增加农民收入作为基本目标,推进农业和农村经济结构调整,以提高农业综合生产能力为基础,以产业化经营为纽带,全面发展农林牧渔各业和农村二、三产业,确保农业稳定发展、农民收入持续增加、农村社会稳定,促进农业和农村经济再上新台阶。')
# words = segmentor.segment('《明天会更好》，是一部励志国产感情电视剧，由导演石美明执导，著名演员郑则仕主演。')
print '\t'.join(words).decode('utf-8')

# words = ['元芳', '你', '怎么', '看']
postags = postagger.postag(words)
print "\t".join(postags)

arcs = parser.parse(words, postags)  # 句法分析
# print "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)

for i in range(len(arcs)): 
	if arcs[i].relation in ["VOB","SBV", "IOB", "POB", "COO", "HED"] and postags[i] == "n":
		print words[i].decode('utf-8'), arcs[i].relation, postags[i]

netags = recognizer.recognize(words, postags)
print "\t".join(netags)

roles = labeller.label(words, postags, netags, arcs)
print len(arcs), len(roles), len(postags)
for role in roles:
    print role.index, "".join(
            ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments])

segmentor.release()
postagger.release()
parser.release()
recognizer.release()
labeller.release()