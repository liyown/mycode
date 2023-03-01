# -*- coding: utf-8 -*-
import numpy as np
import random
from new_function import Fill_None

def Generate_Shift_for_AB(k, num):
    
    k = int(k**0.5)
    ShiftList = list()
    #全1的移位矩阵，只有一个
    one = list()
    for i in range(k):
        one.append(1)
    ShiftList.append(one)
    #剩下的num-1可以分为多少组和几个剩余的
    numGroup = int((num-1-k*k)/k)
    numGroupAfter = int((num-1-k*k)%k)
    
    shiftListTemp = list()
    for i in range(2, numGroup+2):
        listTmp = list();
        for j in range(k):
            listTmp.append(i**j)        
        for l in range(k):
            ShiftList.append(shift_matrix(listTmp, l))
   
    listTmp = list()
    for j in range(k):
        listTmp.append((numGroup+2)**j)
    
    for l in range(numGroupAfter):
        ShiftList.append(shift_matrix(listTmp, l))
    
    
    shift = np.array(ShiftList)
    
    return shift         
    

#循环移位函数
def shift_matrix(lst, a):
    '''
    A:待移位矩阵
    a：移位的位数
    return:已移位的矩阵
    '''

    return lst[-a:] + lst[ :-a]


def getWholeData(Systemdata, Codedata):


    whole_data = Systemdata +  Codedata
    #转型为np.array格式
    whole_data = np.array([i for i in whole_data])
   
    return whole_data
    
   
def chooseReceiveList(num, k):
    
    receive_list = list()

    for i in range(num):
        receive_list.append(i)

    list_lc = random.sample(receive_list, k)

    list_lc.sort()
    
    
    return list_lc, receive_list
    
    

def getSystem(receive_data_index,
              k,
              Systemdata,
              Shift,
              receive_list,
              receive_data):
        
    #统计系统包的个数
    CountSystemPackage = 0

    notReceive_list = receive_list[:k]

    #统计系统包的个数
    for i in receive_data_index:
        if i < k:
            CountSystemPackage = CountSystemPackage + 1
            notReceive_list.remove(i)
            

    #建立待恢复矩阵
    DecodedResults = list()
    DecodedResults_shape = list()
    for i in range(k):
        DecodedResults.append(np.zeros(Systemdata[i].shape))
        DecodedResults_shape.append(Systemdata[i].shape)

    #待恢复矩阵的元素设为空值
    for i in range(len(DecodedResults)):
        for j in range(len(DecodedResults[i])):
            DecodedResults[i][j] = np.array([None])


    #右编码包的情况
    #取出编码包的索引和移位矩阵
    CodePackageShift = receive_data_index[CountSystemPackage: ]
    CodePackageShift = [i-k for i in CodePackageShift]
    CodePackageShift = Shift[CodePackageShift]
    
    
    SystemData = receive_data[: CountSystemPackage]#取出系统包
    CodeData = receive_data[CountSystemPackage: ]#取出编码包

    #把系统包先放入其中
    for i in range(len(SystemData)):
        DecodedResults[receive_data_index[i]] = SystemData[i]


    #此函数实现对None地地方填0操作
    DecodedResults, Is_None = Fill_None(DecodedResults)
    
    #填0,1
    for i in notReceive_list:
        Is_None[i] = Is_None[i].astype(int)
    
    return CodePackageShift, DecodedResults, DecodedResults_shape, notReceive_list, CodeData, Is_None



def Generate_Shift_for_AB2(k, n):
    
    ShiftList = list()
    
    for i in range(1, n-k+1):
        temp = list()
        temp.append(0)
        for j in range(k-1):
            temp.append(i+i*j)
        
        ShiftList.append(temp)
        
    shift = np.array(ShiftList)
    return shift
    
       
    
    

























    
