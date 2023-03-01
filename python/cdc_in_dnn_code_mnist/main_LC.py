#########################################
###   单样本mnist程序，会跑的很慢，要cpu好的电脑跑 ###
###   跑的时候可以先注释掉SAZD方案的程序，跑完再换过来注释掉LC ###
###   N，K的设置要参考论文 ###
#########################################
import sys
import time  # 引入time模块
#import numpy as np
import mnist_loader
import numpy as np
from utils import codeLC, codeSAZD, get_model_size
import utils as us
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
import datetime
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

x1, y1= next_batch(16)
print("x.shape:{},y.shape:{}".format(x1.shape, y1.shape))

### Parameters

n_epoch = 10
learning_rate = 0.1
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

file_name_common = 'ce'+'_nHidden'+str(n_node_hidden)+'.txt'
steps = train_num//batch_size


n_k_list = [[5, 2], [6, 3], [9, 6], [11, 8]]

### Poly-CDC-MP-T
def Poly_CDC_MP_T(n, k):
    print('Poly_CDC_MP_T')
    W2 = np.random.randn(n_node_hidden, n_node_input)
    W3 = np.random.randn(n_node_output, n_node_hidden)
    b2 = np.random.randn(n_node_hidden, 1)
    b3 = np.random.randn(n_node_output, 1)
    total_size = 0
    total_time = 0
    size1 = 0
    for j in range(n_epoch):

        # for each batch
        totalTimeLC = 0
        # totalTimeSAZD = 0
        total_acc = 0
        total_loss = 0

        # total_acc_SA = 0
        # total_loss_SA = 0
        # total_size_SA = 0
        for s in range(steps):
            x, y = next_batch(batch_size)
            total_code_time = 0
            # Feed forward
            a1 = x
            ##LC方案

            startLC = time.time()
            #z2 = np.dot(W2, a1) + b2
            result, z2_size, z2_time, z2_size1 = codeLC(W2, a1, n, k)
            z2 = result + b2
            a2 = sigmoid(z2)
            z3 = np.dot(W3, a2) + b3
            #z3 = codeLC(W3, a2, n, k) + b3
            a3 = sigmoid(z3)
            total_size += z2_size
            total_code_time += z2_time
            size1 += z2_size1
            ## Backpropagation

            # Step 1: Error at the output layer [Cross-Entropy Cost]
            delta_3 = (a3 - y)
            total_loss += np.sum(np.abs(a3-y))
            #delta_2 =  sigmoid_prime(z2) * np.dot(W3.transpose(), delta_3)
            result_d, delta2_size, delta2_time, delta2_size1 = codeLC(W3.transpose(), delta_3, n, k)
            delta_2 = sigmoid_prime(z2) * result_d
            total_size += delta2_size
            total_code_time += delta2_time
            size1 += delta2_size1
            # Step 3: Gradient of C in terms of bias
            gradient_b3 = delta_3
            gradient_b2 = delta_2
            # Step 4: Gradient of C in terms of weight
            gradient_W3 = np.dot(delta_3, a2.transpose())
            gradient_W2 = np.dot(delta_2, a1.transpose())

            # update gradients
            sum_gradient_b3 = np.sum(gradient_b3, axis=1)
            sum_gradient_b2 = np.sum(gradient_b2, axis=1)
            sum_gradient_W3 = gradient_W3
            sum_gradient_W2 = gradient_W2

            total_size += get_model_size(gradient_W2)
            ## Training Error
            total_acc += np.sum(np.argmax(a3, axis = 0) == np.argmax(y, axis = 0))

            # update weights & biases
            b3 -= (learning_rate * sum_gradient_b3 / batch_size).reshape(b3.shape)
            b2 -= (learning_rate * sum_gradient_b2 / batch_size).reshape(b2.shape)
            W3 -= learning_rate * sum_gradient_W3 / batch_size
            W2 -= learning_rate * sum_gradient_W2 / batch_size
            stopLC = time.time()
            totalLC = (stopLC - startLC)
            totalTimeLC += totalLC
            total_time += totalLC - total_code_time + total_code_time/k
            #print((stopLC - startLC))
            if s % 10000 == 0:
                print(s)
                print('totalTimeLC', totalTimeLC)
                print('total_acc: ', total_acc)
        # Report Training Error
        print('Epoch: ', j)
        print("TRAIN_LOSS:\t%5f" % (total_loss/train_num))
        print("TRAIN_ACC:\t%5f" % (total_acc/train_num))


        # print("TRAIN_LOSS_SA:\t%5f" % (total_loss_SA/train_num))
        # print("TRAIN_ACC_SA:\t%5f" % (total_acc_SA/train_num))


        a0 = val_set[0].T
        z2 = np.dot(W2, a0) + b2
        #z2 = codeLC(W2, a0, n, k) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(W3, a2) + b3
        #z3 = codeLC(W3, a2, n, k) + b3
        a3 = sigmoid(z3)

        ## Test Error
        # in test data, label info is a number not one-hot vector as in training data
        sum_of_value_error = np.sum(np.argmax(a3, axis=0) == np.argmax(val_y.T, axis=0))

        # Report Value Error
        print("VAL_ACC:\t%5f" % (sum_of_value_error/val_set[0].shape[0]))

    a0 = test_set[0].T
    z2 = np.dot(W2, a0) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)


    ## Test Error
    # in test data, label info is a number not one-hot vector as in training data
    sum_of_test_error = np.sum(np.argmax(a3, axis = 0) == np.argmax(test_y.T, axis = 0))

    # Report Test Error
    print('\n\n',f'n={n}, k={k}')
    print("TEST_ACC:\t%5f" % (sum_of_test_error/test_set[0].shape[0]))
    print("TRAIN_SIZE:\t%5f" % (total_size / n_epoch / k + size1 / n_epoch))
    print("TRAIN_TIME:\t%5f" % (total_time / n_epoch))
    # tem = np.array(sum_of_test_error)
    # label = str(j)+'.npy'
    # np.save('./LC/'+label, tem)

### DEU-Poly-CDC-MP-T
def DEU_Poly_CDC_MP_T(n, k):
    print(DEU_Poly_CDC_MP_T)
    total_size_DEU = 0
    total_time_DEU = 0
    size1 = 0
    W2 = np.random.randn(n_node_hidden, n_node_input)
    W3 = np.random.randn(n_node_output, n_node_hidden)
    b2 = np.random.randn(n_node_hidden, 1)
    b3 = np.random.randn(n_node_output, 1)
    for j in range(n_epoch):

        # for each batch
        totalTimeLC = 0
        # totalTimeSAZD = 0
        total_acc = 0
        total_loss = 0
        # total_acc_SA = 0
        # total_loss_SA = 0
        # total_size_SA = 0
        for s in range(steps):
            x, y = next_batch(batch_size)
            total_code_time = 0
            # Feed forward
            a1 = x
            ##LC方案

            startLC = time.time()
            # z2 = np.dot(W2, a1) + b2
            result, z2_size, z2_time, z2_size1 = codeLC(W2, a1, n, k)
            z2 = result + b2
            a2 = sigmoid(z2)
            z3 = np.dot(W3, a2) + b3
            # z3 = codeLC(W3, a2, n, k) + b3
            a3 = sigmoid(z3)
            total_size_DEU += z2_size
            total_code_time += z2_time
            size1 += z2_size1
            ## Backpropagation

            # Step 1: Error at the output layer [Cross-Entropy Cost]
            delta_3 = (a3 - y)
            total_loss += np.sum(np.abs(a3 - y))
            # delta_2 =  sigmoid_prime(z2) * np.dot(W3.transpose(), delta_3)
            result_d, delta2_size, delta2_time, delta2_size1 = codeLC(W3.transpose(), delta_3, n, k)
            delta_2 = sigmoid_prime(z2) * result_d
            total_size_DEU += delta2_size
            total_code_time += delta2_time
            size1 += delta2_size1
            # Step 3: Gradient of C in terms of bias
            gradient_b3 = delta_3
            gradient_b2 = delta_2
            # Step 4: Gradient of C in terms of weight
            gradient_W3 = np.dot(delta_3, a2.transpose())
            gradient_W2 = np.dot(delta_2, a1.transpose())

            # update gradients
            sum_gradient_b3 = np.sum(gradient_b3, axis=1)
            sum_gradient_b2 = np.sum(gradient_b2, axis=1)
            sum_gradient_W3 = gradient_W3
            sum_gradient_W2 = gradient_W2

            total_size_DEU += get_model_size(delta_2) + get_model_size(a1.transpose())
            ## Training Error
            total_acc += np.sum(np.argmax(a3, axis=0) == np.argmax(y, axis=0))

            # update weights & biases
            b3 -= (learning_rate * sum_gradient_b3 / batch_size).reshape(b3.shape)
            b2 -= (learning_rate * sum_gradient_b2 / batch_size).reshape(b2.shape)
            W3 -= learning_rate * sum_gradient_W3 / batch_size
            W2 -= learning_rate * sum_gradient_W2 / batch_size
            stopLC = time.time()
            totalLC = (stopLC - startLC)
            totalTimeLC += totalLC
            total_time_DEU += totalLC - total_code_time + total_code_time/k
            # print((stopLC - startLC))
            if s % 10000 == 0:
                print(s)
                print('totalTimeLC', totalTimeLC)
                print('total_acc: ', total_acc)
        # Report Training Error
        print('Epoch: ', j)
        print("TRAIN_LOSS:\t%5f" % (total_loss / train_num))
        print("TRAIN_ACC:\t%5f" % (total_acc / train_num))



        # print("TRAIN_LOSS_SA:\t%5f" % (total_loss_SA/train_num))
        # print("TRAIN_ACC_SA:\t%5f" % (total_acc_SA/train_num))

        a0 = val_set[0].T
        z2 = np.dot(W2, a0) + b2
        # z2 = codeLC(W2, a0, n, k) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(W3, a2) + b3
        # z3 = codeLC(W3, a2, n, k) + b3
        a3 = sigmoid(z3)

        ## Test Error
        # in test data, label info is a number not one-hot vector as in training data
        sum_of_value_error = np.sum(np.argmax(a3, axis=0) == np.argmax(val_y.T, axis=0))

        # Report Value Error
        print("VAL_ACC:\t%5f" % (sum_of_value_error / val_set[0].shape[0]))

    a0 = test_set[0].T
    z2 = np.dot(W2, a0) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    ## Test Error
    # in test data, label info is a number not one-hot vector as in training data
    sum_of_test_error = np.sum(np.argmax(a3, axis=0) == np.argmax(test_y.T, axis=0))

    # Report Test Error
    print('\n\n',f'n={n}, k={k}')
    print("TEST_ACC:\t%5f" % (sum_of_test_error / test_set[0].shape[0]))
    print("TRAIN_SIZE_DEU:\t%5f" % (total_size_DEU / n_epoch / k + size1 / n_epoch))
    print("TRAIN_TIME:\t%5f" % (total_time_DEU / n_epoch))
    # tem = np.array(sum_of_test_error)
    # label = str(j)+'.npy'
    # np.save('./LC/'+label, tem)

for n_k in n_k_list:
    # n = n_k[0]
    # k = n_k[1]
    n = 13
    k = 10
    Poly_CDC_MP_T(n, k)
    DEU_Poly_CDC_MP_T(n, k)
    quit()


"""
TRAIN_SIZE_LC(DEU):	[11,8]733.947754(Mb), [9,6]826.009115(Mb), [6,3]1194.254557(Mb), [5,2]1562.500000(Mb)(只传递W2)
TRAIN_SIZE_LC(N):	[11,8]23117.065430(Mb), [9,6]30670.166016(Mb), [6,3]60882.568359(Mb), [5,2]91094.970703(Mb)(只传递W2)
TRAIN_TIME_LC(DEU):    [11,8]1833.873003(s), [9,6]1727.596271(s), [6,3]1761.529932(s), [5,2]1934.451815(s)
TRAIN_TIME_LC(N):    [11,8]1824.287978(s), [9,6]1725.687405(s), [6,3]1764.255969(s), [5,2]1917.295994(s)
TOTAL_TIME_LC(DEU):    [11,8]1956.1976286666666(s), [9,6]1865.2644568333333(s), [6,3]1960.5723581666666(s), [5,2]2194.8684816666664(s)
TOTAL_TIME_LC(N):    [11,8]5677.132216333333(s), [9,6]6837.381740999999(s), [6,3]11911.3506955(s), [5,2]17099.79111116667(s)

************************************************************************************************************************

TRAIN_SIZE_LC(DEU):	[13,10]678.710938, [11,8]733.947754(Mb), [9,6]826.009115(Mb), [6,3]1194.254557(Mb), [5,2]1562.500000(Mb)(只传递W2)
TRAIN_SIZE_LC(N):	[13,10]43.182373, [11,8]23117.065430(Mb), [9,6]30670.166016(Mb), [6,3]60882.568359(Mb), [5,2]91094.970703(Mb)(只传递W2)
TRAIN_TIME_LC(DEU): [13,10]2099.503051, [11,8]2149.873003(s), [9,6]2028.835502(s), [6,3]2002.173469(s), [5,2]2097.038666(s)
TRAIN_TIME_LC(N):   [13,10]2176.907631, [11,8]2138.778666(s), [9,6]2018.466455(s), [6,3]1987.097116(s), [5,2]2084.512417(s)
TOTAL_TIME_LC(DEU): [13,10]18585.205078, [11,8]2272.197628666667(s), [9,6]2166.503687833333(s), [6,3]2201.2158951666665(s), [5,2]2357.4553326666664(s)
TOTAL_TIME_LC(N):   [13,10]43.182373, [11,8]5991.622904333333(s), [9,6]7130.160790999999(s), [6,3]12134.1918425(s), [5,2]17267.007534166667(s)
acc(DEU): [13,10]0.780900
acc(N): [13,10]0.781500
"""
