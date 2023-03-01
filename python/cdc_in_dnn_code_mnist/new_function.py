# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:59:39 2019

@author: Administrator
"""

import numpy as np
from itertools import combinations
import time
from datetime import datetime
import copy


def Get_ReceiveList(NumPackets, k):
    receive_list = list()

    for i in range(NumPackets):
        receive_list.append(i)

    result_list = list(combinations(receive_list, k))
    return result_list


#构造系统包，指定分块的数量
def Generate_Systemdata(k, dim, ran, W):
    '''
    k: 系统包的个数 k = 4,9,16
    dim:数据矩阵的维度
    return: 两个已分块的矩阵
    '''
    num = int(k** 0.5)
    list_A = list()
    list_B = list()

    for i in range(num):
        
        
        list_A.append(ran * np.random.randn(dim, W))
        list_B.append(ran * np.random.randn(W, dim))
        #list_A.append((i + 1)*np.ones((dim, dim)))
        #list_B.append((i + 1)*np.ones((dim, dim)))
        #list_A.append(ran * lista)
        #list_B.append(ran * listb)        

    return list_A, list_B



def Generate_Systemdata_secrecy(k, dim, max_shift):

    '''
    k: 系统包的个数 k = 4,9,16
    dim:数据矩阵的维度
    return: 两个已分块的矩阵
    '''
    num = int(k** 0.5)
    list_A = list()
    list_B = list()

    list_A.append(np.random.rand(dim+ max_shift, dim))

    list_B.append(np.random.rand(dim, dim+ max_shift))

    for i in range(num - 1):
        list_A.append((i + 1)*np.ones((dim, dim)))
        list_B.append((i + 1)*np.ones((dim, dim)))

    return list_A, list_B



#构造k = 4时的移位矩阵：[[0 1].[1 0],[0 2], ]
def Generation_Shift(num):
    '''
    num:编码包的个数
    return:移位矩阵
    '''
    shift = [[[0, i+ 1],[i+ 1, 0]] for i in range(num)]
    shift = np.array(shift)
    shift = shift.reshape((2* num, 2))
    return  shift


#循环移位函数
def shift_matrix(lst, a):
    '''
    A:待移位矩阵
    a：移位的位数
    return:已移位的矩阵
    '''

    return lst[-a:] + lst[ :-a]

def Generate_ShiftList(k, n):

    '''
    k:数据包恢复的个数
    n：多少组移位矩阵
    return:[[0 1 3],[3 0 1],[1 3 0]] 或[[0 2 6 12],[12 0 2 6],[6 12 0 2],[2 6 12 0]]
    '''

    k = int(k** 0.5)
    ShiftList = [0]
    for i in range(k):
        ShiftList.append(ShiftList[i]+ i)
    del ShiftList[0]
    ShiftList_n = [n*i for i in ShiftList]

    Shift_List = list()
    for i in range(k):
        Shift_List.append(shift_matrix(ShiftList_n, i))

    return Shift_List


def Generate_ShiftList_new(k, num):

    '''
    k:数据包恢复的个数
    n：多少组移位矩阵
    return:[[0 1 3],[3 0 1],[1 3 0]] 或[[0 2 6 12],[12 0 2 6],[6 12 0 2],[2 6 12 0]]
    '''

    k = int(k** 0.5)

    ShiftList = list()
    for i in range(num):
        ShiftList.append(list([0]))


    for i in range(k-1):
        for j in range(num):
            ShiftList[j].append((j+1 + i*(num) + ShiftList[j][i]))


    ShiftList_n = list()
    for i in range(num):
        for j in range(k):
            ShiftList_n.append(shift_matrix(ShiftList[i], j))

    return ShiftList_n

def Generate_Shift_secrecy_cycle(k, num):

    k = int(k**0.5)

    ShiftList = list()
    for i in range(num):
        ShiftList.append(list([0]))

    for i in range(k-1):
        for j in range(num):
            ShiftList[j].append(int((j+1 + i*(num) + ShiftList[j][i])))

    ShiftList_n = list()
    for i in range(len(ShiftList)):
        for j in range(len(ShiftList[0])-1):
            ShiftList_n.append(shift_matrix(ShiftList[i][1:], j))

    shift = np.array(ShiftList_n)

    zeros = np.zeros((shift.shape[0], 1))
    shift = np.concatenate((zeros, shift), axis = 1).astype(int)

    shift = shift.reshape((num*(k-1), k))

    max_shift = int(np.max(shift))

    return shift, max_shift




#构造移位函数（编码函数）
def Generate_Shift(k, num):

    '''
    num: 移位矩阵的组数，例如num = 2 shift = [[0 1],[1 0],[0 2],[2 0]]
    '''
    shift = list()
    if k==4:
        MaxShift = num
        shift = Generation_Shift(MaxShift)    #编码方式 矩阵D

    else:

        shift = Generate_ShiftList_new(k, num)
        shift = np.array(shift)
        shift = shift.reshape((int(k** 0.5)* num, int(k** 0.5)))


    return shift


#系统包
def mul(A, B):
    return np.dot(A, B)


def Generate_Systemmatrix(list_A, list_B):
    #计算系统包
    matrix_data = list()
    for i in range(len(list_A)):
        #for j in range(len(list_B)):

        matrix_data.append(np.dot(list_A[i], list_B))
    return matrix_data



#两个不同维度的矩阵相加或相减
def matrix_mat(matrix1):
    #先求出两个矩阵的维度
    xshape = list()
    yshape = list()

    for i in range(len(matrix1)):

        xshape.append(matrix1[i].shape[0])
        yshape.append(matrix1[i].shape[1])
    #求出最大维度
    max_x= max(xshape)
    max_y= max(yshape)

    #按最大维度对小的矩阵进行补零
    for i in range(len(matrix1)):

        zero_x = np.zeros((xshape[i], max_y - yshape[i]))
        zero_y = np.zeros((max_x - xshape[i], max_y))

        matrix1[i] = np.concatenate((matrix1[i], zero_x), axis = 1)
        matrix1[i] = np.concatenate((matrix1[i], zero_y), axis = 0)
        
    #判断相加或相减
    result = np.zeros((max_x, max_y))
    for i in range(len(matrix1)):
        result = result + matrix1[i]

    return result


#编码函数   仅限于A = [A1; A2]情况
def Code(shift, A, B):
    A_a = list()
    #B_b = list()
    for i in range(len(shift)):

        zero = np.zeros([shift[i], A[i].shape[1]])
        A_a.append(np.concatenate((zero, A[i]), axis=0))# 纵向堆叠
        #B_b.append(np.concatenate((zero.T, B[i]), axis = 1))

    #print(A_a)
    #print(B_b)

    AddResult_A = matrix_mat(A_a)
    #AddResult_B = matrix_mat(B_b)

    result = np.dot(AddResult_A, B)

    return result


def Generate_Codematrix(list_A, list_B, Shift):

    matrix_data = list()
    for i in range(len(Shift)):
        matrix_data.append(Code(Shift[i], list_A, list_B))

    return matrix_data


#4个系统包，直接打印结果
def decode_4_SystemPackage(receive_data):
    #已经解码完成的标志
    TotalResult = 0
    #print_result(receive_data)

    return TotalResult


def decodeSystemPackage(CodePackageShift,
                        DecodedResults,
                        ValidationPackage,
                        DecodedResults_shape,
                        notReceive_list,
                        CodeData,
                        Is_None):
    '''
    #receive_data：接收到的数据包，包括系统包和编码包
    #receive_data_index：接收到的数据包的索引
    #CodePackageShift：编码包的移位矩阵
    #dim：数据维度
    #CountSystemPackage：系统包的个数
    #DecodedResults：待恢复的数据
    #ValidationPackage：验证作用的系统包
    #notReceive_list：未收到系统包的编号
    '''
    #开始解码
    IterData = IterDifferAll(CodeData,      #全部编码数据包
                             DecodedResults,        #初步解码后的结果
                             CodePackageShift)      #编码包的移位矩阵
    CountTrue = 1
    forwordCountTrue = 0
    q = 0
    while CountTrue != 0:    
        #F函数，功能为寻找暴露位
        q += 1
        #print('q  ', q)
        DecodedResults, Is_None, IterData = F_fuction(Is_None,             #True 和 false矩阵，已恢复的数据为false，未恢复的数据为True
                                                      IterData,            #编码包减去系统包和一维解码的结果
                                                      CodePackageShift,    #
                                                      DecodedResults,      #
                                                      DecodedResults_shape,
                                                      notReceive_list)

        #F函数的前提，重新填写True和False
        CountTrue = Refill_One(Is_None,
                               notReceive_list)
        #print("剩余", CountTrue, "个数据。")
        if forwordCountTrue == CountTrue:
            break
        else:
            forwordCountTrue = CountTrue
        '''
        if q%5 == 0:
            validationNum = Validation(ValidationPackage, DecodedResults)
            print(validationNum == CountTrue)

            
            print('验证数量：', validationNum)
        
        for i in range(len(IterData)):
            for j in range(len(IterData[i])):
                for k in range(len(IterData[i][j])):
                    if IterData[i][j][k] < 0.1:
                        IterData[i][j][k] == 0
        '''
    # validation = Validation(ValidationPackage, DecodedResults)
    #print(validation)

    return np.array(DecodedResults).reshape(-1, 1)
    # return np.array(ValidationPackage).reshape(-1, 1)

#判断验证数据包和解码的数据包是否相等
def Validation(ValidationPackage, DecodedResults):
    TotalResult = 0
    for i in range(len(ValidationPackage)):
        for j in range(len(ValidationPackage[i])):
            for k in range(len(ValidationPackage[i][j])):
                    result  = round(ValidationPackage[i][j][k] - DecodedResults[i][j][k], 3)
                    if result != 0:
                        #求出不相等的个数
                        TotalResult = TotalResult + 1
    return TotalResult



#对初步恢复的数据包填True和False，已恢复填False，未恢复填True
def Refill_One(Is_None, notReceive_list):
    CountTrue = 0
    for i in notReceive_list:
        CountTrue += np.sum(Is_None[i])
    return CountTrue


#此函数实现对None地地方填0操作
def Fill_None(DecodedResults):
    #找出空值的地方，返回bool类型

    Is_None = list()

    for i in DecodedResults:
        Is_None.append(np.isnan(i))

    for i in range(len(DecodedResults)):
        for j in range(len(DecodedResults[i])):
            for k in range(len(DecodedResults[i][j])):
                if (Is_None[i][j][k] == True):
                    DecodedResults[i][j][k] = 0
                    
    return DecodedResults, Is_None

#迭代，多个编码包
def IterDifferAll(CodeData,
                  SystemData,
                  CodePackageShift):

    mid_result = list()

    for i in range(len(CodeData)):

        mid_result.append(differ(CodeData[i],
                                 SystemData,
                                 CodePackageShift[i]))

    return mid_result



def IterDifferAllForXunhun(IterData,      #全部编码数据包
                           DecodedResults,        #初步解码后的结果
                           CodePackageShift,      #编码包的移位矩阵
                           notReceive_list):
    
    mid_result = list()

    for i in range(len(IterData)):
        mid_result.append(differForXunhun(IterData[i],
                                          DecodedResults,
                                          CodePackageShift[i],
                                          notReceive_list))

    return mid_result
    
    
#一个编码包减去多个系统包后的结果
def differForXunhun(CodeData,
           DecodedResults,
           CodePackageShift,
           notReceive_list):
    
    result = list()

    for i in notReceive_list:

        result = ZeroShift(DecodedResults[i],
                           Shift_Choose(CodePackageShift, i))

        CodeData = matrix_dif(CodeData,
                              result,
                              add = 0)

    return CodeData   
    
    

#一个编码包减去多个系统包后的结果
def differ(CodeData,
           SystemData,
           CodePackageShift):
    
    result = list()

    for i in range(len(SystemData)):

        result = ZeroShift(SystemData[i],
                           Shift_Choose(CodePackageShift, i))

        CodeData = matrix_dif(CodeData,
                              result,
                              add = 0)

    return CodeData

#两个不同维度的矩阵相加或相减
def matrix_dif(matrix1, matrix2, add):
    #先求出两个矩阵的维度
    matrix = [matrix1, matrix2]

    x1, y1 = matrix1.shape
    x2, y2 = matrix2.shape

    x = [x1, x2]
    y = [y1, y2]
    #求出最大维度
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    #按最大维度对小的矩阵进行补零
    for i in range(2):

        zero_x = np.zeros((x[i], max_y - y[i]))
        zero_y = np.zeros((max_x - x[i], max_y))

        matrix[i] = np.concatenate((matrix[i], zero_x), axis = 1)
        matrix[i] = np.concatenate((matrix[i], zero_y), axis = 0)
    #判断相加或相减
    if add == 1:
        return matrix[0] + matrix[1]
    else:
        return matrix[0] - matrix[1]



#根据移位矩阵填充系统包
def ZeroShift(data, shift):

    D = shift[0]
    R = shift[1]
    x, y = data.shape

    zero_R = np.zeros((x, R))
    zero_D = np.zeros((D, y + R))

    data = np.concatenate((zero_R, data), axis = 1)
    data = np.concatenate((zero_D, data), axis = 0)

    return data

#对D, R移位进行选择
def Shift_Choose(CodePackageShift, index):
    
    shift_index = list()
    shift_len = len(CodePackageShift)


    for i in range(shift_len):
        for j in range(shift_len):
            shift_index.append([i,j])

    D = CodePackageShift[shift_index[index][0]]
    R = CodePackageShift[shift_index[index][1]]

    return [D, R]


#F函数，功能为寻找暴露位
def F_fuction(Is_None, 
              CodeData,
              CodePackageShift, 
              DecodedResults, 
              DecodedResults_shape, 
              notReceive_list):

    #填0,1
    for i in notReceive_list:
        Is_None[i] = Is_None[i].astype(int)

    newIs_None = copy.deepcopy(Is_None)
    
    for i in range(len(CodeData)):
        #比较哪个地方有相同的“1”
        Mid_DecodedResults = CompareValue(CodeData[-1 -i], #编码包的位置从最大到最小
                                          Is_None,
                                          CodePackageShift[-1 - i], #取出编码矩阵
                                          DecodedResults_shape,
                                          notReceive_list)
        
        j = 0
        #print(Mid_DecodedResults)
        #dt=datetime.now() #创建一个datetime类对象
        #print(dt.minute, '\t', dt.second)
        #print()
        for k in notReceive_list:
            #判断是否应该相加，因为某个位置重复为暴露位
            DecodedResults[k], CodeData, Is_None[k] = Judge_add(DecodedResults[k],
                                                                    Mid_DecodedResults[j],
                                                                    Is_None[k],
                                                                    CodeData,
                                                                    CodePackageShift, k)
            j = j + 1

        # Mid_DecodedResults = list()
    return DecodedResults, Is_None, CodeData

#DecodedResults位置为0的地方可以相加
def Judge_add(DecodedResults, Mid_DecodedResults, newIs_None, CodeData, CodePackageShift, k):
    # for i in range(len(DecodedResults)):
    #     for j in range(len(DecodedResults[i])):
    #         if (DecodedResults[i][j] == 0) & (newIs_None[i][j] == 1) & (Mid_DecodedResults[i][j] != 0):
    #             #标志位，标志哪一位已经恢复
    #             newIs_None[i][j] = 0
    #             DecodedResults[i][j] = Mid_DecodedResults[i][j]
    #
    #             #减去相应的值
    #             for l in range(len(CodeData)):
    #                 shift = Shift_Choose(CodePackageShift[l], k)
    #                 CodeData[l][shift[0] + i][shift[1] + j] = CodeData[l][shift[0] + i][shift[1] + j] - Mid_DecodedResults[i][j]
    #                 #if round((CodeData[l][shift[0] + i][shift[1] + j]), 1) <= 0:
    #                 #    CodeData[l][shift[0] + i][shift[1] + j] = 0

    # 更新newIs_None --> Type2  --> 元素间异或运算
    Mid_boolValue = (Mid_DecodedResults != 0).astype(int)
    newIs_None = (newIs_None ^ (Mid_boolValue)).astype(int)

    # 更新 DecodedResults
    DecodedResults += Mid_DecodedResults

    # CodeData[l]减去相应的值
    for l in range(len(CodeData)):
        shift = Shift_Choose(CodePackageShift[l], k)
        midShiftTemp = ZeroShift(Mid_DecodedResults, shift)
        CodeData[l] = matrix_dif(CodeData[l], midShiftTemp, add=0)

    return DecodedResults, CodeData, newIs_None



def CompareValue(CodeData, Is_None, CodePackageShift, DecodedResults_shape, notReceive_list):

    result = list()
    #对Is_None填充1和0
    for i in notReceive_list:
        result.append(ZeroFill(ZeroShift(Is_None[i],
                                         Shift_Choose(CodePackageShift, i)),    #选择编码矩阵
                                         CodeData))
    #计算所有矩阵的总和
    Sum_result = np.sum(result, axis=0)
    #把总和矩阵中的1找出来，只有1是需要的，2,3,4不需要
    Store_F_Value = list()
    Store_F_Value.append((Sum_result == 1).astype(int))

    #存储经过F函数后的结果，该结果为1的地方为对应系统包的暴露位
    Store_AfterF_Value = list()
    for i in range(len(result)):
        #恢复正确的移位
        # Mid_Result = Recover_Shift(JudgeToOne(Store_F_Value[0], result[i]) * CodeData,
        #                            CodePackageShift,
        #                            notReceive_list[i],
        #                            DecodedResults_shape[notReceive_list[i]])
        Mid_Result = Recover_Shift(Store_F_Value[0]*result[i] * CodeData,
                                   CodePackageShift,
                                   notReceive_list[i],
                                   DecodedResults_shape[notReceive_list[i]])
        #收集数据
        Store_AfterF_Value.append(Mid_Result)

    return Store_AfterF_Value



#填充矩阵，使两个维度一样
def ZeroFill(data, CodePackage):

    matrix = [data, CodePackage]

    x1, y1 = data.shape
    x2, y2 = CodePackage.shape

    x = [x1, x2]
    y = [y1, y2]

    max_x = max(x1, x2)
    max_y = max(y1, y2)

    for i in range(2):

        zero_x = np.zeros((x[i], max_y - y[i]))
        zero_y = np.zeros((max_x - x[i], max_y))

        matrix[i] = np.concatenate((matrix[i], zero_x), axis = 1)
        matrix[i] = np.concatenate((matrix[i], zero_y), axis = 0)
        
    return matrix[0]

#判断矩阵是否为1，此函数可以找出和编码包相同的“1”,这个”1“可以为对应系统包的暴露位
def JudgeToOne(Store_F_Value, result):

    for i in range(len(Store_F_Value)):
        for j in range(len(Store_F_Value[i])):
                if Store_F_Value[i][j] == result[i][j] and result[i][j] == 1:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
    return result


#恢复正确的移位
def Recover_Shift(Mid_Result, CodePackageShift, i, DecodedResults_shape):

    ShiftIndex = Shift_Choose(CodePackageShift, i)

    return recover(Mid_Result, ShiftIndex, DecodedResults_shape)

#从已经解码的含移位的数据包中恢复系统包
def recover(result, shift_index, shape):

    D = shift_index[0]
    R = shift_index[1]

    Row = shape[0]
    column = shape[1]


    result = result[D: D+Row, R: R+column]

    return result



