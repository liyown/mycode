########################################
###   单样本mnist程序，会跑的很慢，要cpu好的电脑跑 ###
###   跑的时候可以先注释掉SAZD方案的程序，跑完再换过来注释掉LC ###
###   N，K的设置要参考论文 ###
#########################################

import time  # 引入time模块
import mnist_loader
import numpy as np
from utils import codeLC, codeSAZD, get_model_size
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical

### Data Loading
### Activation Functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
train_set, val_set, test_set = load_mnist_datasets('./data/mnist.pkl.gz')
train_y, val_y, test_y=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])


# 随机选择训练样本
train_num = train_set[0].shape[0]
def next_batch(batch_size):
    idx = np.random.choice(train_num, batch_size)
    return train_set[0][idx].T, train_y[idx].T

# x1, y1= next_batch(16)
# print("x.shape:{},y.shape:{}".format(x1.shape, y1.shape))

### Parameters

n_epoch = 10
learning_rate = 0.01
batch_size = 1

### Network Architecture
#784*1200,1200*1200,10*1200
n_node_input = 784
n_node_hidden = 1200
n_node_output = 10

### Weight & Bias

W2 = np.random.randn(n_node_hidden, n_node_input)
W3 = np.random.randn(n_node_output, n_node_hidden)
b2 = np.random.randn(n_node_hidden, 1)
b3 = np.random.randn(n_node_output, 1)

W2_SA = np.random.randn(n_node_hidden, n_node_input)
W3_SA = np.random.randn(n_node_output, n_node_hidden)
b2_SA = np.random.randn(n_node_hidden, 1)
b3_SA = np.random.randn(n_node_output, 1)

### Training
test_errors = []
training_errors = []

# file_name_common = 'ce'+'_nHidden'+str(n_node_hidden)+'.txt'
steps = train_num//batch_size

n_k_list = [[5, 2], [6, 3], [9, 6], [11, 8]]

### N-SAZD-CDC-MP-T
def N_SAZD_CDC_MP_T(n, k):
    print('N_SAZD_CDC_MP_T')
    W2_SA = np.random.randn(n_node_hidden, n_node_input)
    W3_SA = np.random.randn(n_node_output, n_node_hidden)
    b2_SA = np.random.randn(n_node_hidden, 1)
    b3_SA = np.random.randn(n_node_output, 1)

    total_time = 0
    total_size_SA = 0
    size1 = 0
    for j in range(n_epoch):
        # for each batch
        # totalTimeLC = 0
        totalTimeSAZD = 0
        total_acc_SA = 0
        total_loss_SA = 0
        for s in range(steps):
            total_code_time = 0
            x, y = next_batch(batch_size)

            # Feed forward
            a1_SA = x
            ###SAZD方案
            startSAZD = time.time()
            # z2_SA = np.dot(W2_SA, a1_SA)+b2_SA
            result, z2_size, z2_time, z2_size1 = codeSAZD(W2_SA, a1_SA, n, k)
            z2_SA = result + b2_SA
            a2_SA = sigmoid(z2_SA)
            z3_SA = np.dot(W3_SA, a2_SA) + b3_SA
            # z3_SA = codeSAZD(W3_SA, a2_SA, n, k) + b3_SA
            a3_SA = sigmoid(z3_SA)
            total_size_SA += z2_size
            total_code_time += z2_time

            ## Backpropagation

            # Step 1: Error at the output layer [Cross-Entropy Cost]
            delta_3_SA = (a3_SA - y)
            total_loss_SA += np.sum(np.abs(a3_SA - y))

            result_d, delta2_size, delta2_time, delta2_size1 = codeSAZD(W3_SA.transpose(), delta_3_SA, n, k)
            delta_2_SA = sigmoid_prime(z2_SA) * result_d
            total_size_SA += delta2_size
            total_code_time += delta2_time

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

            # e = np.identity(784)
            # sum_gradient_W2_SA, sum_gradient_W2_SA_side = codeSAZD(gradient_W2_SA,e,n,k)
            # sum_gradient_W2_SA = sum_gradient_W2_SA.reshape(1200,784)

            total_size_SA += get_model_size(gradient_W2_SA)

            # total_size_SA += sum_gradient_W2_SA_side

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
            if s % 10000 == 0:
                print(s)
                # print('totalTimeLC', totalTimeLC)
                print('totalTimeSAZD', totalTimeSAZD)
                # print('total_acc: ', total_acc)
                print('total_acc_SA: ', total_acc_SA)

        # Report Training Error
        print('Epoch: ', j)
        # print("TRAIN_LOSS:\t%5f" % (total_loss / train_num))
        # print("TRAIN_ACC:\t%5f" % (total_acc / train_num))
        print("TRAIN_LOSS_SA:\t%5f" % (total_loss_SA/train_num))
        print("TRAIN_ACC_SA:\t%5f" % (total_acc_SA/train_num))

        a0 = val_set[0].T
        z2 = np.dot(W2_SA, a0) + b2_SA
        a2 = sigmoid(z2)
        z3 = np.dot(W3_SA, a2) + b3_SA
        a3 = sigmoid(z3)

        ## Test Error
        # in test data, label info is a number not one-hot vector as in training data
        sum_of_value_error = np.sum(np.argmax(a3, axis=0) == np.argmax(val_y.T, axis=0))

        # Report Value Error
        print("VAL_ACC:\t%5f" % (sum_of_value_error / val_set[0].shape[0]))

    a0 = test_set[0].T
    z2 = np.dot(W2_SA, a0) + b2_SA
    a2 = sigmoid(z2)
    z3 = np.dot(W3_SA, a2) + b3_SA
    a3 = sigmoid(z3)

    ## Test Error
    # in test data, label info is a number not one-hot vector as in training data
    sum_of_test_error = np.sum(np.argmax(a3, axis=0) == np.argmax(test_y.T, axis=0))

    # Report Test Error
    # print("TEST_ACC:\t%5f" % (sum_of_test_error/test_set[0].shape[0]))
    # tem = np.array(sum_of_test_error)
    # label = str(j)+'.npy'
    # np.save('./LC/'+label, tem)
    print('\n\n',f'n={n}, k={k}')
    print("TEST_ACC:\t%5f" % (sum_of_test_error / test_set[0].shape[0]))
    # tem = np.array(sum_of_test_error)
    # label = str(j)+'.npy'
    # np.save('./SAZD/'+label, tem)
    print("TRAIN_SIZE_SA:\t%5f" % (total_size_SA / n_epoch /k + size1 / n_epoch))
    print("TRAIN_TIME_SA:\t%5f" % (total_time / n_epoch), '\n')

### DEU-SAZD-CDC-MP-T
def DEU_SAZD_CDC_MP_T(n, k):
    print('DEU_SAZD_CDC_MP_T')
    W2_SA = np.random.randn(n_node_hidden, n_node_input)
    W3_SA = np.random.randn(n_node_output, n_node_hidden)
    b2_SA = np.random.randn(n_node_hidden, 1)
    b3_SA = np.random.randn(n_node_output, 1)
    total_time = 0
    total_size_SA_DEU = 0
    size1 = 0
    for j in range(n_epoch):

        # for each batch
        # totalTimeLC = 0
        totalTimeSAZD = 0
        total_acc_SA = 0
        total_loss_SA = 0
        for s in range(steps):
            total_code_time = 0
            x, y = next_batch(batch_size)

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
            delta_2_SA = sigmoid_prime(z2_SA) * result_d
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

            if s % 10000 == 0:
                print(s)
                # print('totalTimeLC', totalTimeLC)
                print('totalTimeSAZD', totalTimeSAZD)
                # print('total_acc: ', total_acc)
                print('total_acc_SA: ', total_acc_SA)

        # Report Training Error
        print('Epoch: ', j)
        # print("TRAIN_LOSS:\t%5f" % (total_loss / train_num))
        # print("TRAIN_ACC:\t%5f" % (total_acc / train_num))

        print("TRAIN_LOSS_SA_DEU:\t%5f" % (total_loss_SA / train_num))
        print("TRAIN_ACC_SA_DEU:\t%5f" % (total_acc_SA / train_num))


        a0 = val_set[0].T
        z2 = np.dot(W2_SA, a0) + b2_SA
        a2 = sigmoid(z2)
        z3 = np.dot(W3_SA, a2) + b3_SA
        a3 = sigmoid(z3)

        ## Test Error
        # in test data, label info is a number not one-hot vector as in training data
        sum_of_value_error = np.sum(np.argmax(a3, axis=0) == np.argmax(val_y.T, axis=0))

        # Report Value Error
        print("VAL_ACC:\t%5f" % (sum_of_value_error / val_set[0].shape[0]))

    a0 = test_set[0].T
    z2 = np.dot(W2_SA, a0) + b2_SA
    a2 = sigmoid(z2)
    z3 = np.dot(W3_SA, a2) + b3_SA
    a3 = sigmoid(z3)

    ## Test Error
    # in test data, label info is a number not one-hot vector as in training data
    sum_of_test_error = np.sum(np.argmax(a3, axis=0) == np.argmax(test_y.T, axis=0))

    # Report Test Error
    print('\n\n',f'n={n}, k={k}')
    print("TEST_ACC:\t%5f" % (sum_of_test_error / test_set[0].shape[0]))
    # tem = np.array(sum_of_test_error)
    # label = str(j)+'.npy'
    # np.save('./SAZD/'+label, tem)
    print("TRAIN_SIZE_SA_DEU:\t%5f" % (total_size_SA_DEU / n_epoch /k + size1 / n_epoch))
    print("TRAIN_TIME_SA_DEU:\t%5f" % (total_time / n_epoch), '\n')

for n_k in n_k_list:
    # n = n_k[0]
    # k = n_k[1]
    n = 11
    k = 8
    N_SAZD_CDC_MP_T(n, k)
    DEU_SAZD_CDC_MP_T(n, k)

"""
TRAIN_SIZE_SA(DEU):	[11,8]51.879883(Mb), [9,6]66.884359(Mb), [6,3]128.809611(Mb), [5,2]191.497803(Mb)(只传递W2)
TRAIN_SIZE_SA(N):	[11,8]22430.801392(Mb), [9,6]29907.608032(Mb), [6,3]59814.834595(Mb), [5,2]89722.061157(Mb)(只传递W2)
TRAIN_TIME_SA(DEU):    [11,8]1113.741390(s), [9,6]1209.916197(s), [6,3]1215.270137(s), [5,2]1216.609457 (s)
TRAIN_TIME_SA(N):    [11,8]1090.362027(s), [9,6]1199.145646(s), [6,3]1201.936360(s), [5,2]1281.679017(s)
TOTAL_TIME_SA(DEU):    [11,8]1122.3880371666667(s), [9,6]1221.0635901666667(s), [6,3]1236.7384055(s), [5,2]1248.5257575(s)
TOTAL_TIME_SA(N):    [11,8]4828.828925666667(s), [9,6]6183.746984666666(s), [6,3]11171.075459166666(s), [5,2]16235.355876500002(s)

************************************************************************************************************************

TRAIN_SIZE_SA(DEU):	[13,10]43.182373, [11,8]51.879883(Mb), [9,6]66.884359(Mb), [6,3]128.809611(Mb), [5,2]191.497803(Mb)(只传递W2)
TRAIN_SIZE_SA(N):	[13,10]17944.717407, [11,8]22430.801392(Mb), [9,6]29907.608032(Mb), [6,3]59814.834595(Mb), [5,2]89722.061157(Mb)(只传递W2)
TRAIN_TIME_SA(DEU):  [13,10]1332.790819,  [11,8]1309.405455(s), [9,6]1364.573936(s), [6,3]1590.461851(s), [5,2]1913.186219(s)
TRAIN_TIME_SA(N):   [13,10]1287.680740, [11,8]1313.808003(s), [9,6]1365.423235(s), [6,3]1636.743251(s), [5,2]1951.523047(s)
TOTAL_TIME_SA(DEU):  [13,10]43.182373, [11,8]1318.0521021666668(s), [9,6]1375.7213291666667(s), [6,3]1611.9301195(s), [5,2]1945.1025195(s)
TOTAL_TIME_SA(N):  [13,10]43.182373, [11,8]5052.274901666668(s), [9,6]6350.024573666667(s), [6,3]11605.882350166667(s), [5,2]16905.1999065(s)

acc(DEU): [13,10]0.935600
acc(N): [13,10]0.944200
"""