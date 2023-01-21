#########################################
###   BASE LINE + Cross Entropy Loss  ###
#########################################
import time  # 引入time模块
# import numpy as np
import mnist_loader
import numpy as np
from utils import codeLC, codeSAZD
import utils as us
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


### Data Loading
### Activation Functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


dfr = pd.read_csv('winequality-white.csv', sep=';')  # dfr short for dataframe_red
dfw = pd.read_csv('winequality-white.csv', sep=';')  # dfw short for dataframe_white
y = dfr.quality.values  # set 'quality' as target
X = dfr.drop('quality', axis=1).values  # rest are features

seed = 8  # set seed for reproducibility
train_set, test_set, train_y, test_y = train_test_split(X, y, test_size=0.2,
                                                        random_state=seed)

# train_set, val_set, test_set = load_mnist_datasets('./data/mnist.pkl.gz')
# train_y, val_y, test_y = to_categorical(train_set[1]), to_categorical(val_set[1]), to_categorical(test_set[1])


# 随机选择训练样本
train_num = train_set.shape[0]


def next_batch(batch_size):
    idx = np.random.choice(train_num, batch_size)
    return train_set[idx].T, train_y[idx].T


x1, y1 = next_batch(16)
print("x.shape:{},y.shape:{}".format(x1.shape, y1.shape))

### Parameters

n_epoch = 500
learning_rate = 0.01
batch_size = 100

### Network Architecture

n_node_input = 11
n_node_hidden = 200
n_node_output = 1

### Weight & Bias

W2 = np.random.randn(n_node_hidden, n_node_input)
W3 = np.random.randn(n_node_hidden, n_node_hidden)
W4 = np.random.randn(n_node_output, n_node_hidden)
b2 = np.random.randn(n_node_hidden, 1)
b3 = np.random.randn(n_node_hidden, 1)
b4 = np.random.randn(n_node_output, 1)

W2_SA = np.random.randn(n_node_hidden, n_node_input)
W3_SA = np.random.randn(n_node_output, n_node_hidden)
b2_SA = np.random.randn(n_node_hidden, 1)
b3_SA = np.random.randn(n_node_output, 1)

### Training
test_errors = []
training_errors = []

file_name_common = 'ce' + '_nHidden' + str(n_node_hidden) + '.txt'
steps = train_num // batch_size

k = 8
n = 20

for j in range(n_epoch):

    # for each batch
    totalTimeLC = 0
    totalTimeSAZD = 0
    total_acc = 0
    total_loss = 0
    total_acc_SA = 0
    total_loss_SA = 0
    for s in range(steps):
        x, y = next_batch(batch_size)
        ## Feed forward
        a1 = x
        ## LC

        startLC = time.time()
        z2 = np.dot(W2, a1) + b2
                    # 200,11  11,100
        # z2 = codeLC(W2, a1, n, k) + b2
        a2 = sigmoid(z2)
        # 200,100
        z33 = np.dot(W3, a2) + b3
        z3 = codeLC(W3, a2, n, k) + b3
        #           200,200  200,100

        a3 = sigmoid(z3)
        # 200,100
        z4 = np.dot(W4, a3) + b4
        a4 = 10 * sigmoid(z4)
        ## Backpropagation

        # Step 1: Error at the output layer [Cross-Entropy Cost]
        delta_4 = (a4 - y)
        total_loss += np.sum(np.power(np.abs(a4 - y), 2) / 2)
        delta_3 = sigmoid_prime(z3) * np.dot(W4.transpose(), delta_4)
        delta_2 = sigmoid_prime(z2) * np.dot(W3.transpose(), delta_3)
        # delta_2 =  sigmoid_prime(z2) * codeLC(W3.transpose(), delta_3, n , k)
        # Step 3: Gradient of C in terms of bias
        gradient_b4 = delta_4
        gradient_b3 = delta_3
        gradient_b2 = delta_2
        # Step 4: Gradient of C in terms of weight
        gradient_W4 = np.dot(delta_4, a3.transpose())
        gradient_W3 = np.dot(delta_3, a2.transpose())
        gradient_W2 = np.dot(delta_2, a1.transpose())

        # update gradients
        sum_gradient_b4 = np.sum(gradient_b4, axis=1)
        sum_gradient_b3 = np.sum(gradient_b3, axis=1)
        sum_gradient_b2 = np.sum(gradient_b2, axis=1)
        sum_gradient_W4 = gradient_W4
        sum_gradient_W3 = gradient_W3
        sum_gradient_W2 = gradient_W2

        ## Training Error
        # total_acc += np.sum(np.argmax(a3, axis = 0) == np.argmax(y, axis = 0))

        # update weights & biases
        b4 -= (learning_rate * sum_gradient_b4 / batch_size).reshape(b4.shape)
        b3 -= (learning_rate * sum_gradient_b3 / batch_size).reshape(b3.shape)
        b2 -= (learning_rate * sum_gradient_b2 / batch_size).reshape(b2.shape)
        W4 -= learning_rate * sum_gradient_W4 / batch_size
        W3 -= learning_rate * sum_gradient_W3 / batch_size
        W2 -= learning_rate * sum_gradient_W2 / batch_size
        '''
        a1_SA = x
        ###SAZD
        startSAZD = time.time()
        #z2_SA = np.dot(W2_SA, a1) + b2_SA
        z2_SA = codeSAZD(W2_SA, a1_SA, n, k) + b2_SA
        a2_SA = sigmoid(z2_SA)
        z3_SA = np.dot(W3_SA, a2_SA) + b3_SA
        #z3_SA = codeSAZD(W3_SA, a2_SA, n, k) + b3_SA
        a3_SA = sigmoid(z3_SA)
        ## Backpropagation
        
        # Step 1: Error at the output layer [Cross-Entropy Cost]
        delta_3_SA = (a3_SA - y)
        total_loss_SA += np.sum(np.abs(a3_SA - y))
        delta_2_SA =  sigmoid_prime(z2_SA) * codeSAZD(W3_SA.transpose(), delta_3_SA, n, k)
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

        ## Training Error
        total_acc_SA += np.sum(np.argmax(a3_SA, axis = 0) == np.argmax(y, axis = 0))

        # update weights & biases
        b3_SA -= (learning_rate * sum_gradient_b3_SA / batch_size).reshape(b3_SA.shape)
        b2_SA -= (learning_rate * sum_gradient_b2_SA / batch_size).reshape(b2_SA.shape)
        W3_SA -= learning_rate * sum_gradient_W3_SA / batch_size
        W2_SA -= learning_rate * sum_gradient_W2_SA / batch_size
        stopSAZD = time.time()
        totalSAZD = (stopSAZD - startSAZD)
        totalTimeSAZD += totalSAZD
        
        if s%10 == 0:
            print(j, s)
            #print('totalTimeLC', totalTimeLC)
            print('totalTimeSAZD', totalTimeSAZD)
            #print('total_acc: ', total_acc)
            print('total_acc_SA: ', total_acc_SA)
        '''

    # Report Training Error
    print('Epoch: ', j)
    print("TRAIN_LOSS:\t%5f" % (total_loss / train_num))
    # print("TRAIN_ACC:\t%5f" % (total_acc/train_num))

    # print("TRAIN_LOSS_SA:\t%5f" % (total_loss_SA/train_num))
    # print("TRAIN_ACC_SA:\t%5f" % (total_acc_SA/train_num))

    a0 = test_set.T
    z2 = np.dot(W2, a0) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(W3, a2) + b3
    # z3 = codeLC(W3, a2, n, k) + b3
    a3 = sigmoid(z3)
    z4 = np.dot(W4, a3) + b4
    a4 = 10 * sigmoid(z4)

    ## Test Error
    # in test data, label info is a number not one-hot vector as in training data
    sum_of_test_loss = np.sum(np.power(np.abs(a4 - test_y), 2) / 2)

    # Report Test Error
    print("TEST_ACC:\t%5f" % (sum_of_test_loss / test_set.shape[0]))

    '''
    a0_SA = test_set[0].T
    z2_SA = np.dot(W2_SA, a0_SA) + b2_SA
    a2_SA = sigmoid(z2_SA)
    z3_SA = np.dot(W3_SA, a2_SA) + b3_SA
    a3_SA = sigmoid(z3_SA)

    ## Test Error
    # in test data, label info is a number not one-hot vector as in training data
    sum_of_test_error_SA = np.sum(np.argmax(a3_SA, axis = 0) == np.argmax(test_y.T, axis = 0))

    # Report Test Error
    print("VAL_ACC_SA:\t%5f" % (sum_of_test_error_SA/test_set[0].shape[0]))
    '''
