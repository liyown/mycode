# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:03:26 2019

@author: Administrator
"""
import numpy as np
from datetime import datetime
from decode_SA import decode
import time
from timeit import default_timer as timer


#循环移位函数
def shift_matrix(lst, a):
    '''
    A:待移位矩阵
    a：移位的位数
    return:已移位的矩阵
    '''

    return lst[-a:] + lst[ :-a]

def Generate_ShiftList(k, num):

    '''
    k:数据包恢复的个数
    n：多少组移位矩阵
    return:[[0 1 3],[3 0 1],[1 3 0]] 或[[0 2 6 12],[12 0 2 6],[6 12 0 2],[2 6 12 0]]
    '''
    ShiftList = [0]
    for i in range(k):
        ShiftList.append(ShiftList[i]+ i)

    del ShiftList[0]

    Shift_List = list()
    for i in range(k):
        Shift_List.append(shift_matrix(ShiftList, i))
    Shift_List = np.array(Shift_List)

    Shift_List = Shift_List[:(num-k), :]

    return Shift_List

def Generate_ShiftList_1(k, num):

    '''
    k:数据包恢复的个数
    n：多少组移位矩阵
    return:[[0 1 3],[3 0 1],[1 3 0]] 或[[0 2 6 12],[12 0 2 6],[6 12 0 2],[2 6 12 0]]
    '''
    ShiftList = [0]
    for i in range(k):
        ShiftList.append(ShiftList[i]+ 1)

    del ShiftList[0]

    Shift_List = list()
    for i in range(k):
        Shift_List.append([i*label for label in ShiftList])

    Shift_List = np.array(Shift_List).T

    Shift_List = Shift_List[:(num-k), :]


    return Shift_List




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


#编码函数
def Code(shift, A):
    A_a = list()
    for i in range(len(shift)):
        zero = np.zeros([shift[i], A[i].shape[1]])
        A_a.append(np.concatenate((zero, A[i]), axis = 0))

    AddResult_A = matrix_mat(A_a)

    return AddResult_A


def encoding_SAZD(list_A, Shift):

    matrix_data = list()
    for i in range(len(Shift)):
        matrix_data.append(Code(Shift[i], list_A))

    return matrix_data


#系统包
def mul(A, B):
    return np.dot(A, B)


def Generate_Systemmatrix(list_A, list_B):

    #计算系统包
    matrix_data = list()
    for i in range(len(list_A)):
        # for j in range(len(list_B)):

        result_Ax = mul(list_A[i], list_B)

        matrix_data.append(result_Ax)

    return matrix_data


def computate_SA(AddResult_A, list_A, list_x):

    #合成系统数据，计算Ax
    Systemdata = Generate_Systemmatrix(list_A, list_x)

    Codedata = list()

    for i in range(len(AddResult_A)):

        result = np.dot(AddResult_A[i], list_x)
        # result = result.reshape(AddResult_A[i].shape[0], list_x[0].shape[1])

        Codedata.append(result)


    SAZD_whole_data = Systemdata + Codedata

    return np.array(SAZD_whole_data)




def decode_SAZD(k, SAZD_whole_data, choose_index, Shift, dim):
    #receive = list(receive_index[220])  #接收到的数据包的索引
    #stop_SA1 = timer()
    receive_data = SAZD_whole_data[choose_index]

    CountSystemPackage = 0
    #Systemdata = SAZD_whole_data[ :k]

    #统计系统包的个数
    for i in choose_index:
        if i < k:
            CountSystemPackage += 1

    #取出编码包的索引和移位矩阵
    CodePackageShift = choose_index[CountSystemPackage: ]
    Shift_index = [i-k for i in CodePackageShift]
    CodePackageShift = [Shift[i] for i in Shift_index]

    #解码程序
    if CountSystemPackage < k:
        Decode = decodeSystemPackage(receive_data,
                                     CodePackageShift,
                                     CountSystemPackage,
                                     choose_index,
                                     dim)

        return Decode

    else:
        return receive_data


def change_Shiftmatrix(CodePackageShift, SystemData_index, IterData):

    CodePackageShift = np.delete(CodePackageShift, SystemData_index, axis=1)

    min_shift = np.min(CodePackageShift, axis = 1)


    CodePackageShift = CodePackageShift - min_shift.reshape(min_shift.shape[0], 1)


    for i in range(len(min_shift)):
        IterData[i] = IterData[i][min_shift[i]:]


    return CodePackageShift, IterData



def decodeSystemPackage(receive_data,
                        CodePackageShift,
                        CountSystemPackage,
                        receive_data_index,
                        dim):

    SystemData = receive_data[: CountSystemPackage]#取出系统包
    SystemData_index = receive_data_index[: CountSystemPackage]

    CodeData = receive_data[CountSystemPackage: ]#取出编码包

    #把系统包先放入其中
    IterData = IterDifferAll(CodeData,              #全部编码数据包
                             SystemData,            #初步解码后的结果
                             CodePackageShift,
                             SystemData_index)      #编码包的移位矩阵
    #stop_SA1 = timer()
    Code_Shift, IterData = change_Shiftmatrix(CodePackageShift,
                                              SystemData_index,
                                              IterData)

    Decode_data = decode(IterData, Code_Shift, dim)

    return Decode_data



#迭代，多个编码包
def IterDifferAll(CodeData,
                  SystemData,
                  CodePackageShift,
                  SystemData_index):

    mid_result = list()

    for i in range(len(CodeData)):
        mid_result.append(differ(CodeData[i],
                                 SystemData,
                                 CodePackageShift[i],
                                 SystemData_index))

    return mid_result


#一个编码包减去多个系统包后的结果
def differ(CodeData,
           SystemData,
           CodePackageShift,
           SystemData_index):

    result = list()

    for i in range(len(SystemData)):

        result = ZeroShift(SystemData[i], CodePackageShift[SystemData_index[i]])

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


    zero_D = np.zeros((shift, 1))

    data = np.concatenate((zero_D, data), axis = 0)

    return data













