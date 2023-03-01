#########################################
###   单样本mnist程序，会跑的很慢，要cpu好的电脑跑 ###
###   跑的时候可以先注释掉SAZD方案的程序，跑完再换过来注释掉LC ###
###   N，K的设置要参考论文 ###
#########################################
import sys
import time  # 引入time模块
# import numpy as np
import mnist_loader
import numpy as np
from utils import codeLC, codeSAZD, get_model_size, matrixMultiply
import utils as us
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
import datetime


### Data Loading
### Activation Functions



def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


train_set, val_set, test_set = load_mnist_datasets('./data/mnist.pkl.gz')
train_y, val_y, test_y = to_categorical(train_set[1]), to_categorical(val_set[1]), to_categorical(test_set[1])

# 随机选择训练样本
train_num = train_set[0].shape[0]


def next_batch(batch_size):
    idx = np.random.choice(train_num, batch_size)
    return train_set[0][idx].T, train_y[idx].T


x1, y1 = next_batch(16)
print("x.shape:{},y.shape:{}".format(x1.shape, y1.shape))

### Parameters

n_epoch = 1
learning_rate = 0.1
batch_size = 1

### Network Architecture
# 784*1200,1200*1200,10*1200
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

file_name_common = 'ce' + '_nHidden' + str(n_node_hidden) + '.txt'
steps = train_num // batch_size

k = 3
n = 6
### Poly-CDC-MP-T
total_size = 0
total_time = 0
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

        # Feed forward
        a1 = x
        ##LC方案

        startLC = time.time()

        # z2 = np.dot(W2, a1) + b2
        z2 = matrixMultiply(W2, a1)+b2

        a2 = sigmoid(z2)
        # z3 = np.dot(W3, a2) + b3
        z3 = np.dot(W3, a2) + b3
        a3 = sigmoid(z3)
        # total_size += sys.getsizeof(z2) + sys.getsizeof(z3)
        total_size += get_model_size(z2) + get_model_size(z3)
        ## Backpropagation

        # Step 1: Error at the output layer [Cross-Entropy Cost]
        delta_3 = (a3 - y)
        total_loss += np.sum(np.abs(a3 - y))
        # delta_2 = sigmoid_prime(z2) * np.dot(W3.transpose(), delta_3)
        delta_2 = sigmoid_prime(z2) * matrixMultiply(W3.transpose(), delta_3)
        total_size += get_model_size(delta_2) + get_model_size(delta_3)
        # Step 3: Gradient of C in terms of bias
        gradient_b3 = delta_3
        gradient_b2 = delta_2
        # Step 4: Gradient of C in terms of weight
        gradient_W3 = np.dot(delta_3, a2.transpose())
        gradient_W2 = np.dot(delta_2, a1.transpose())
        # gradient_W3 = matrixMultiply(delta_3, a2.transpose())
        # gradient_W2 = matrixMultiply(delta_2, a1.transpose())
        # update gradients
        sum_gradient_b3 = np.sum(gradient_b3, axis=1)
        sum_gradient_b2 = np.sum(gradient_b2, axis=1)
        sum_gradient_W3 = gradient_W3
        sum_gradient_W2 = gradient_W2
        total_size += get_model_size(gradient_W2) + get_model_size(gradient_W3)

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
        total_time += totalLC
        # print((stopLC - startLC))
        if s % 2000 == 0:
            print(s)
            print('totalTimeLC', totalTimeLC)
            print('total_acc: ', total_acc)
    # Report Training Error
    print('Epoch: ', j)
    print("TRAIN_LOSS:\t%5f" % (total_loss / train_num))
    print("TRAIN_ACC:\t%5f" % (total_acc / train_num))
    print("TRAIN_SIZE:\t%5f" % (total_size/((1+j))))
    print("TRAIN_TIME:\t%5f" % (total_time / (1 + j)))

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
    print("TEST_ACC:\t%5f" % (sum_of_test_error / test_set[0].shape[0]))
    # tem = np.array(sum_of_test_error)
    # label = str(j)+'.npy'
    # np.save('./LC/'+label, tem)

# TOTAL_TIME: 24450.995636(s)