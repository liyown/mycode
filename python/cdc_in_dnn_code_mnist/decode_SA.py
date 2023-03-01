# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:43:52 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:24:36 2019

@author: Administrator
"""
import numpy as np
import copy
#from new_function import Generate_Systemdata, Generate_Systemmatrix
#from new_function_Ax import SAZD
from timeit import default_timer as timer



def count_min(w_list):

    #找出每行的最小值
    count = 0
    for i in range(len(w_list)):
        min_i = np.min(w_list[i])
        for j in range(len(w_list[i])):
            if w_list[i][j] == min_i:
                count = count + 1
        if count == 1:
            return i, np.argmin(w_list[i])  #返回首行最小值
        else:
            count = 0


def add_one(w_list, j):
    w_list[:, j] = w_list[:, j] + 1

    return w_list


def add_mul(w_list, j):

    w_list[:, j] = w_list[:, j] + 1000000
    return w_list


def decode(X, Mz, dim):
    #print("开始解码")
    #解码向量的
    num = len(X)

    #多项式系数

    Decode = list()
    index = list()
    index_result = list()
    for i in range(num):

        Decode.append(np.zeros(dim))
        index.append(0)
        index_result.append(0)

    count = 0
    #stop_SA7 = timer()
    w_list = Mz

    for k in range(1000000):

        #stop_SA0 = timer()


        i, j = count_min(w_list)  #计算暴露位
        #stop_SA1 = timer()
        w_list = add_one(w_list, j)


        #取出暴露位
        value = copy.copy(X[i][index[i]])

        Decode[j][index_result[j]] = value

        index_result[j] = index_result[j] + 1

        #已经完成某个序列的恢复
        if index_result[j] == dim:
            count = count + 1
            w_list = add_mul(w_list, j)


        #找到当前已经到恢复的位置
        in_index = index[i]
        #前向和后向传播标志位
        current = i

        #更新编码矩阵
        #stop_SA2 = timer()
        for b in range(len(X)):
            #取出当前的编码序列
            yi = X[b]
            if(Mz[b][j] < Mz[current][j]):
                #向后传播
                yi[in_index - Mz[current][j] + Mz[b][j]] = yi[in_index - Mz[current][j] + Mz[b][j]] - value

            else:
                #向前传播
                yi[in_index + Mz[b][j] - Mz[current][j]] = yi[in_index + Mz[b][j] - Mz[current][j]] - value

            #Current_X = X[i]
            if yi[index[b]] == 0 & index[b] <= dim:
                index[b] = index[b] + 1
        #恢复完成，退出
        if count == num:
            #print(k+1)
            break


    return Decode













































































