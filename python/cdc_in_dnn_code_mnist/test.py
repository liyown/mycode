import time  # 引入time模块
import mnist_loader
import numpy as np
from utils import codeLC, codeSAZD, get_model_size
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
from new_function import Generate_Systemmatrix, decodeSystemPackage, Generate_Codematrix, matrix_mat
from AB_product import  chooseReceiveList,  Generate_Shift_for_AB2, getSystem, getWholeData

def code_SA(shift, A):
    A_list = list()
    for i in range(len(shift)):
        zero = np.zeros([shift[i], A[i].shape[1]])
        A_list.append(np.concatenate((zero, A[i]), axis=0))
    result = matrix_mat(A_list)
    return result


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
train_set, val_set, test_set = load_mnist_datasets('./data/mnist.pkl.gz')
train_y, val_y, test_y=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])


def code_SAZD(W_system, W_tilde, input, n, k):
    systemData = Generate_Systemmatrix(W_system, input)  # 系统包
    WholeData = W_tilde
    for i in range(len(W_tilde)):
        WholeData[i] = np.dot(W_tilde[i], input)

    receive_data_index, receive_list = chooseReceiveList(n, k)
    receive_data = WholeData[receive_data_index]
    # 准备阶段，取出不用恢复的系统包，和编码包的移位矩阵
    CodePackageShift, DecodedResults, DecodedResults_shape, notReceive_list, CodeData, Is_None = getSystem(
        receive_data_index,
        k,
        systemData,
        Shift,
        receive_list,
        receive_data)

    # SAZD解码
    # 开始解码
    DecodedResults = decodeSystemPackage(CodePackageShift,
                                         DecodedResults,
                                         systemData,
                                         DecodedResults_shape,
                                         notReceive_list,
                                         CodeData,
                                         Is_None)
    return DecodedResults

# 随机选择训练样本
train_num = train_set[0].shape[0]
def next_batch(batch_size):
    idx = np.random.choice(train_num, batch_size)
    return train_set[0][idx].T, train_y[idx].T
### Parameters

n_epoch = 1
learning_rate = 0.01
batch_size = 1

### Network Architecture
#784*1200,1200*1200,10*1200
n_node_input = 784
n_node_hidden = 1200
n_node_output = 10

steps = train_num//batch_size
n, k = 11, 8
temp_x = None
temp_y = None
### DEU-SAZD-CDC-MP-T

W2_SA = np.random.randn(n_node_hidden, n_node_input)
W3_SA = np.random.randn(n_node_output, n_node_hidden)
b2_SA = np.random.randn(n_node_hidden, 1)
b3_SA = np.random.randn(n_node_output, 1)

W2 = W2_SA
W3 = W3_SA
b2 = b2_SA
b3 = b3_SA

total_time = 0
total_size_SA_DEU = 0
size1 = 0
for j in range(n_epoch):

    # for each batch
    # totalTimeLC = 0
    totalTimeSAZD = 0
    total_acc_SA = 0
    total_loss_SA = 0
    for s in range(1):
        total_code_time = 0
        x, y = next_batch(batch_size)
        temp_x = x
        temp_y = y

        # Feed forward
        a1_SA = x
        ###SAZD方案
        startSAZD = time.time()
        # z2_SA = np.dot(W2_SA, a1) + b2_SA
        result, z2_size, z2_time, z2_size1 = codeSAZD(W2_SA, a1_SA, n, k)
        z2_SA = result + b2_SA
        a2_SA = sigmoid(z2_SA)
        z3_SA = np.dot(W3_SA, a2_SA) + b3_SA
        # z3_SA = codeSAZD(W3_SA, a2_SA, n, k) + b3_SA
        a3_SA = sigmoid(z3_SA)
        total_size_SA_DEU += z2_size
        total_code_time += z2_time
        size1 += z2_size1


        ## Backpropagation

        # Step 1: Error at the output layer [Cross-Entropy Cost]
        delta_3_SA = (a3_SA - y)
        total_loss_SA += np.sum(np.abs(a3_SA - y))
        result_d, delta2_size, delta2_time, delta2_size1 = codeSAZD(W3_SA.transpose(), delta_3_SA, n, k)
        delta_2_SA = a2_SA * result_d
        total_size_SA_DEU += delta2_size
        total_code_time += delta2_time
        size1 += delta2_size1
        # Step 3: Gradient of C in terms of bias
        gradient_b3_SA = delta_3_SA
        gradient_b2_SA = delta_2_SA
        # Step 4: Gradient of C in terms of weight
        gradient_W3_SA = np.dot(delta_3_SA, a2_SA.transpose())
        gradient_W2_SA = np.dot(delta_2_SA, a1_SA.transpose())

        # update gradients
        sum_gradient_b3_SA = np.sum(gradient_b3_SA, axis=1)
        sum_gradient_b2_SA = np.sum(gradient_b2_SA, axis=1)
        sum_gradient_W3_SA = gradient_W3_SA
        sum_gradient_W2_SA = gradient_W2_SA

        total_size_SA_DEU += get_model_size(delta_2_SA) + get_model_size(a1_SA.transpose())


        ## Training Error
        total_acc_SA += np.sum(np.argmax(a3_SA, axis=0) == np.argmax(y, axis=0))

        # update weights & biases
        b3_SA -= (learning_rate * sum_gradient_b3_SA / batch_size).reshape(b3_SA.shape)
        b2_SA -= (learning_rate * sum_gradient_b2_SA / batch_size).reshape(b2_SA.shape)
        W3_SA -= learning_rate * sum_gradient_W3_SA / batch_size
        W2_SA -= learning_rate * sum_gradient_W2_SA / batch_size
        stopSAZD = time.time()
        totalSAZD = (stopSAZD - startSAZD)
        totalTimeSAZD += totalSAZD
        total_time += totalSAZD - total_code_time + total_code_time/k

#         if s % 10000 == 0:
#             print(s)
#             # print('totalTimeLC', totalTimeLC)
#             print('totalTimeSAZD', totalTimeSAZD)
#             # print('total_acc: ', total_acc)
#             print('total_acc_SA: ', total_acc_SA)
#
#     # Report Training Error
#     print('Epoch: ', j)
#     # print("TRAIN_LOSS:\t%5f" % (total_loss / train_num))
#     # print("TRAIN_ACC:\t%5f" % (total_acc / train_num))
#
#     print("TRAIN_LOSS_SA_DEU:\t%5f" % (total_loss_SA / train_num))
#     print("TRAIN_ACC_SA_DEU:\t%5f" % (total_acc_SA / train_num))
#
#
#     a0 = val_set[0].T
#     z2 = np.dot(W2_SA, a0) + b2_SA
#     a2 = sigmoid(z2)
#     z3 = np.dot(W3_SA, a2) + b3_SA
#     a3 = sigmoid(z3)
#
#     ## Test Error
#     # in test data, label info is a number not one-hot vector as in training data
#     sum_of_value_error = np.sum(np.argmax(a3, axis=0) == np.argmax(val_y.T, axis=0))
#
#     # Report Value Error
#     print("VAL_ACC:\t%5f" % (sum_of_value_error / val_set[0].shape[0]))
#
# a0 = test_set[0].T
# z2 = np.dot(W2_SA, a0) + b2_SA
# a2 = sigmoid(z2)
# z3 = np.dot(W3_SA, a2) + b3_SA
# a3 = sigmoid(z3)
#
# ## Test Error
# # in test data, label info is a number not one-hot vector as in training data
# sum_of_test_error = np.sum(np.argmax(a3, axis=0) == np.argmax(test_y.T, axis=0))
#
# # Report Test Error
# print('\n\n',f'n={n}, k={k}')
# print("TEST_ACC:\t%5f" % (sum_of_test_error / test_set[0].shape[0]))
# # tem = np.array(sum_of_test_error)
# # label = str(j)+'.npy'
# # np.save('./SAZD/'+label, tem)
# print("TRAIN_SIZE_SA_DEU:\t%5f" % (total_size_SA_DEU / n_epoch /k + size1 / n_epoch))
# print("TRAIN_TIME_SA_DEU:\t%5f" % (total_time / n_epoch), '\n')

Shift = Generate_Shift_for_AB2(k, n)
W2_list = np.split(W2, k, axis=0)#系统矩阵
W3_list = np.split(W3.transpose(), k, axis=0)
W2_code_list = []#编码矩阵
W3_code_list = []

for i in range(len(Shift)):
    W2_code_list.append(code_SA(shift=Shift[i], A=W2_list))
    W3_code_list.append(code_SA(shift=Shift[i], A=W3_list))

W2_tilde = getWholeData(W2_list, W2_code_list)# 完整的编码矩阵
W3_tilde = getWholeData(W3_list, W3_code_list)
#######################################################################################################################
a1 = temp_x
W2_list = np.split(W2, k, axis=0)#系统矩阵
W3_list = np.split(W3.transpose(), k, axis=0)
z2 = code_SAZD(W2_list, W2_tilde, a1, n, k) + b2
a2 = sigmoid(z2)
z3 = np.dot(W3_SA, a2) + b3_SA
a3 = sigmoid(z3)

###

delta_3 = a3 - temp_y
c3 = code_SAZD(W3_list, W3_tilde, delta_3, n, k)
delta_2 = sigmoid_prime(z2) * c3

gradient_b3 = delta_3
gradient_b2 = delta_2
gradient_W3 = np.dot(delta_3, a2.transpose())
gradient_W2 = np.dot(delta_2, a1.transpose())

sum_gradient_b3 = np.sum(gradient_b3, axis=1)
sum_gradient_b2 = np.sum(gradient_b2, axis=1)
sum_gradient_W3 = gradient_W3
sum_gradient_W2 = gradient_W2


