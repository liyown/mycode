### Activation Functions
import sys

import numpy as np


from new_function import Generate_Systemmatrix, Generate_Codematrix, decodeSystemPackage

from AB_product import  chooseReceiveList,  Generate_Shift_for_AB2, getSystem, getWholeData
from new_function_lc import Generate_Codematrix_LC_ERROR,  decodeLc, decodeLc_ERROR
from new_function_Ax import encoding_SAZD, decode_SAZD, computate_SA
import time
'''
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
'''

def sigmoid(inX):
    from numpy import exp
    #return 1.0/(1+exp(-inX))
    #优化
    if inX>=0:
        return 1.0/(1+exp(-inX))
    else:
        return exp(inX)/(1+exp(inX))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))# -*- coding: utf-8 -*-


def getdata(batch, batch_size):
    x = batch[0][0]
    y = batch[0][1]
    for i in range(1, batch_size):
        x = np.hstack((x, batch[i][0]))
        y = np.hstack((y, batch[i][1]))
        
    return x, y


def codeLC(W, z, n, k):
    # code_start_time = time.time()
    # 横向切分W
    list_A = np.array(np.split(W, k, axis=0))
    list_B = z
    # LC方案编码
    CodedataForLC, codeIndex = Generate_Codematrix_LC_ERROR(list_A, k, n)
    # 20,25,11      20,8                                    8,25,11  8   20
    # LC计算
    CodedataLC = np.array(Generate_Systemmatrix(CodedataForLC, list_B))
    # 20,25,100                                 20,25,11       11,100
    # code_end_time = time.time()
    code_start_time = time.time()
    # 随机选择k个数据包
    receive_data_index, receive_list = chooseReceiveList(n, k)  # 接收到的数据包的索引
    # LC解码
    # tickstartDecodeLC = time.time()
    decodeData = decodeLc(CodedataLC[receive_data_index], codeIndex[receive_data_index])
    # 25,100
    # decodeData = decodeLc_ERROR(CodedataLC[receive_data_index], codeIndex[receive_data_index])
    codesize = get_model_size(CodedataLC[receive_data_index])
    codesize1 = get_model_size(CodedataLC)
    # 拼接数据
    # decodeData = decodeData.reshape(-1, 1)
    code_end_time = time.time()
    decode_time = code_end_time - code_start_time  # decode_time应为子节点上的解码时间，需要除k
    decodeData = np.array(decodeData).reshape(200, 100)
    return decodeData


def codeSAZD(W, z, n, k):
    # code_start_time = time.time()
    Shift = Generate_Shift_for_AB2(k, n)
    # 横向切分W成k份（k台机器）

    list_A = np.split(W, k, axis=0)
    list_B = z

    Systemdata = Generate_Systemmatrix(list_A, list_B)
    # SAZD方案编码
    Codedata = Generate_Codematrix(list_A, list_B, Shift)
    # SAZD计算

    # 使系统包和编码包链接起来
    whole_data = getWholeData(Systemdata, Codedata)
    # code_end_time = time.time()
    receive_data_index, receive_list = chooseReceiveList(n, k)  # 接收到的数据包的索引
    code_start_time = time.time()
    # 取出随机接收到的k个数据包,并统计数据量
    receive_data = whole_data[receive_data_index]
    codesize = get_model_size(receive_data)
    codesize1 = get_model_size(whole_data)
    # 准备阶段，取出不用恢复的系统包，和编码包的移位矩阵
    CodePackageShift, DecodedResults, DecodedResults_shape, notReceive_list, CodeData, Is_None = getSystem(
        receive_data_index,
        k,
        Systemdata,
        Shift,
        receive_list,
        receive_data)

    # SAZD解码
    # 开始解码
    DecodedResults = decodeSystemPackage(CodePackageShift,
                                         DecodedResults,
                                         Systemdata,
                                         DecodedResults_shape,
                                         notReceive_list,
                                         CodeData,
                                         Is_None)

    # 拼接数据
    code_end_time = time.time()
    decode_time = code_end_time - code_start_time  # decode_time应为子节点上的解码时间，需要除k
    return DecodedResults, codesize, decode_time, codesize1


def get_model_size(object):
    # para_num = sum([np.prod(w.shape) for w in model.get_weights()])
    # para_num = object.shape[0]*object.shape[1]
    para_num = object.size
    # para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
    para_size = para_num * 4 / 1024 / 1024
    return para_size

def matrixMultiply(A, B):
    # 获取A的行列数
    A_rows, A_cols = A.shape
    # 获取B的行列数
    B_rows, B_cols = B.shape
    #生成A的行数，B的列数的矩阵
    zeros = np.zeros([A_rows, B_cols])

    for rows in range(A_rows):
        #rows为A矩阵的每一行
        for cols in range(B_cols):
            #cols为B矩阵的每一列
            product = 0
            for row in range(len(A[rows])):
                product += A[rows][row] * B[row][cols]
            zeros[rows][cols] = product
    return zeros

# def codeSAZD(w, z, n, k):
#     Shift = Generate_Shift_for_AB2(k, n)
#     W = np.split(w, k)
#     W_tilde = encoding_SAZD(W, Shift)
#     whole_data = computate_SA(W_tilde, W, z)
#     receive_data_index, receive_list = chooseReceiveList(n, k)  # 接收到的数据包的索引
#     result = np.split(np.dot(w, z), k)
#     decode_data = decode_SAZD(k, whole_data, receive_data_index, Shift, W[0].shape[0])
#     return decode_data







