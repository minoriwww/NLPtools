# coding=utf-8
import matplotlib.pyplot as plt  
from math import *
from numpy import *
import random  
'''
多项式梯度下降法
http://blog.csdn.net/neuldp/article/details/52063613
'''
# x1, x2, ..xn 个变量为矩阵的横行
# 每个变量的不同取值为列

# 1对于一行的相加 2减去y 3对下一行同样操作并加和
# 参数：当前theta(向量) xset(x的矩阵mxn) yset(y的矩阵mx1)
def sum_J(theta, theta_id, xset, yset):
    sum_J_theta_id = 0
    for row in xrange(0, len(yset)):
        J = 0
        for i in xrange(0, len(theta)):
            J += theta[i] * xset[row][i]

        sum_J_theta_id += (J - yset[row]) * xset[row][theta_id]
    print sum_J_theta_id
    return sum_J_theta_id

def scaling(read_file_ndarray):
    return (read_file_ndarray - read_file_ndarray.mean())/(read_file_ndarray.max() - read_file_ndarray.min())

# degree 最大迭代次数
def grediant(xset, yset, alpha, dgree):
    theta = []
    if xset[0] is None:
        return
    # 列的数量
    num_theta = len(xset[0])
    # for i in range(0, num_theta):
    #     theta.append(0) #把所有theta初始化成0
    # 经多次试验 确定初值 使迭代步骤最少
    theta.append(0)
    theta.append(0.5)
    theta.append(0.5)

    # 行数
    length = len(yset)
    # jtheta = 0
    total = 0
    sum_total = 0
    e = 1000 # 误差 初始值可以很大
    iter_num = 0
    # while e >= 1e-2 or iter_num <= dgree:
    # for对每一个theta迭代操作
    while iter_num < dgree:
        print theta
        total = 0

        for j in range(0, num_theta):
            # 全部theta数组， theta下标， x, y, 行号
            total = sum_J(theta, j, xset, yset)
            # 更新所有theta
            theta[j] = theta[j] - (alpha/length)*(total)
        
        if e > (alpha/length)*(total):
            e = (alpha/length)*(total)

        iter_num += 1

    return theta


#X=[1.5,2,1.5,2,3,3,3.5,3.5,4,4,5,5]
#Y=[3,3.2,4,4.5,4,5,4.2,4.5,5,5.5,4.8,6.5]
# x0 == 1
# m x n : m组数据 n列
X = [[1, 2104, 3],
    [1, 1600, 3],
    [1, 2400, 3],
    [1, 1416, 2],
    [1, 3000, 4]
    ]

Y = [400, 330, 369, 232, 540]
# 函数入口
#a = grediant(X,Y,0.0005,10)
#print a

read_file = genfromtxt("ex1data2.txt" , delimiter=',')
print read_file.shape

X = []
for i in xrange(0,len(read_file)):
    
    theta1 = scaling(read_file[:, 0])
    theta0 = ones(theta1.shape)
    theta2 = scaling(read_file[:, 1])

    X = column_stack(( theta0, theta1, theta2 ))

print X
Y = scaling(read_file[:, 2])
print Y
# print read_file[:, 1].mean()
# print read_file[:, 0].mean()
# print read_file[:, 0].max()

a = grediant(X, Y, 0.1, 60)
print a

# a,b,r = linefit(X,Y)

# print "y = %10.5fx + %10.5f" %(b,a)
# x = np.linspace(0, 10, 10000)
#y = b * x + a
# y = a * x + b
# plt.plot(x,y)
# plt.scatter(X, Y)
# plt.show()
