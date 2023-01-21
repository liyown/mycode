# -*- coding: utf-8 -*-
import numpy as np
import random
import math


def Generate_Codematrix_LC_ERROR(Systemdata, k, n):
    dataLC = list()
    codeIndex = list()

    codeIndex = [[(np.random.rand() * (j / k)) ** i for i in range(k)] for j in range(n)]

    # codeIndex = [[((j/k))**i for i in range(k)] for j in range(n)]

    # codeIndex = []
    # for j in range(n):
    #     subcodeIndex = []
    #     for i in range(k):
    #         subcodeIndex.append((np.random.rand() * j/k) ** i)
    #     codeIndex.append(subcodeIndex)

    CodeDataLC = list()

    for i in range(len(codeIndex)):
        zeros = np.zeros((Systemdata[0].shape[0], Systemdata[0].shape[1]))
        for j in range(len(codeIndex[i])):
            zeros = zeros + codeIndex[i][j] * Systemdata[j]
        CodeDataLC.append(zeros)

    return np.array(CodeDataLC), np.array(codeIndex)


def Generate_Codematrix_LC(list_A, list_B, k, n):
    CodedataLC = list()
    codeIndex = list()
    for i in range(1, n + 1):
        codeIndex.append(i)
        listA = addMatrixA(list_A, i)
        listB = addMatrixB(list_B, i)
        CodedataLC.append(np.dot(listA, listB))

    return np.array(CodedataLC), codeIndex


def addMatrixA(matrix, i):
    lenMatrix = len(matrix)
    listMatrix = list()
    for n in range(lenMatrix):
        num = math.pow(i, n)
        # print('A', num)
        for j in range(len(matrix[n])):
            for k in range(len(matrix[n][j])):
                matrix[n][j][k] = num * matrix[n][j][k]

        listMatrix.append(matrix[n])

    lenL = len(matrix[0])
    lenW = len(matrix[0][0])
    zeroMatrix = np.zeros((lenL, lenW))
    for i in range(len(listMatrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                zeroMatrix[j][k] = zeroMatrix[j][k] + matrix[i][j][k]

    return zeroMatrix


#     x --> (2x)**n
def addMatrixB(matrix, i):
    lenMatrix = len(matrix)
    listMatrix = list()
    for n in range(lenMatrix):
        num = math.pow(2 * i, n)
        # print('B', num)
        for j in range(len(matrix[n])):
            for k in range(len(matrix[n][j])):
                matrix[n][j][k] = num * matrix[n][j][k]

        listMatrix.append(matrix[n])

    lenL = len(matrix[0])
    lenW = len(matrix[0][0])
    zeroMatrix = np.zeros((lenL, lenW))
    for i in range(len(listMatrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                zeroMatrix[j][k] = zeroMatrix[j][k] + matrix[i][j][k]

    return zeroMatrix


def getCodeIndexMatrix(codeIndex, k, num):
    codeMatrix = list()
    for i in range(num):
        code = list()
        for j in range(k):
            code.append(math.pow(codeIndex[i], j))
        codeMatrix.append(code)
    codeMatrix = np.array(codeMatrix)

    return codeMatrix


def decodeLc(CodedataLC, codeMatrix):
    # 8,25,100    8,8
    # 矩阵求逆
    codeMatrix = np.linalg.inv(codeMatrix)

    tempMatrix = np.zeros([25,100])
    temp = list()
    matrix = list()
    # 解码
    result = list()
    for j in range(len(codeMatrix)):
        # temp = list()
        for k in range(len(codeMatrix[j])):
            # matrix = mulMatrix(codeMatrix[j][k], CodedataLC[k])
            matrix = codeMatrix[j][k] * CodedataLC[k]
            tempMatrix = tempMatrix + matrix

        temp.append(tempMatrix)

        # resultMatrix = addMatrix(temp)
        #
        # result.append(resultMatrix)

    # print(result)
    return temp


def decodeLc_ERROR(CodedataLC, codeMatrix):
    # 矩阵求逆
    codeMatrix = np.linalg.inv(codeMatrix)

    # 解码
    result = list()
    for j in range(len(codeMatrix)):
        zeros = np.zeros((CodedataLC[0].shape[0], CodedataLC[0].shape[1]))
        for k in range(len(codeMatrix[j])):
            zeros = zeros + codeMatrix[j][k] * CodedataLC[k]
        zeros = zeros.reshpe(1, -1)
        result.append(zeros)

    return np.array(result)


def mulMatrix(index, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = index * matrix[i][j]

    return matrix


def addMatrix(matrix):
    lenL = len(matrix[0])
    lenW = len(matrix[0][0])

    zeroMatrix = np.zeros((lenL, lenW))

    for i in range(lenL):
        for j in range(lenW):
            temp = 0
            for k in range(len(matrix)):
                temp = temp + matrix[k][i][j]
            zeroMatrix[i][j] = temp

    return zeroMatrix


def countError(Systemdata, decodeData):
    error = list()
    errorSys = list()
    for i in range(len(Systemdata)):
        error.append(np.linalg.norm(decodeData[i] - Systemdata[i]))
        errorSys.append(np.linalg.norm(Systemdata[i]))

    error = np.array(error)
    errorSys = np.array(errorSys)
    errorSum = np.sum(error)
    errorSysSum = np.sum(errorSys)

    return errorSum / errorSysSum
