#########################################
###   base程序，用来测试的  ###
#########################################


import numpy as np
import mnist_loader
import utils as us
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
### Data Loading

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

n_epoch = 30
learning_rate = 1
batch_size = 1

### Network Architecture

n_node_input = 784
n_node_hidden = 500
n_node_output = 10

### Weight & Bias

W2=np.random.randn(n_node_hidden, n_node_input)
b2=np.random.randn(n_node_hidden, 1)

W3=np.random.randn(n_node_output, n_node_hidden)
b3=np.random.randn(n_node_output, 1)

### Activation Functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

### Training
test_errors = []
training_errors = []

file_name_common = 'ce'+'_nHidden'+str(n_node_hidden)+'.txt'
steps = train_num//batch_size

for j in range(n_epoch):

    # for each batch
    total_acc = 0
    total_loss = 0
    for s in range(steps):
        x, y = next_batch(batch_size)
        ## Feed forward

        a1 = x
        z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(W3, a2) + b3
        a3 = sigmoid(z3)

        ## Backpropagation

        # Step 1: Error at the output layer [Cross-Entropy Cost]
        delta_3 = (a3-y)
        total_loss += np.sum(np.abs(a3-y))
        # Step 2: Error relationship between two adjacent layers
        delta_2 =  sigmoid_prime(z2)*np.dot(W3.transpose(), delta_3)
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

        ## Training Error
        total_acc += np.sum(np.argmax(a3, axis = 0) == np.argmax(y, axis = 0))

        # update weights & biases
        b3 -= (learning_rate * sum_gradient_b3 / batch_size).reshape(b3.shape)
        b2 -= (learning_rate * sum_gradient_b2 / batch_size).reshape(b2.shape)
        W3 -= learning_rate * sum_gradient_W3 / batch_size
        W2 -= learning_rate * sum_gradient_W2 / batch_size

    # Report Training Error
    print('Epoch: ', j)
    print("TRAIN_LOSS:\t%5f" % (total_loss/train_num))
    print("TRAIN_ACC:\t%5f" % (total_acc/train_num))



    a0 = val_set[0].T
    z2 = np.dot(W2, a0) + b2
    a2 = us.sigmoid(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = us.sigmoid(z3)

    ## Test Error
    # in test data, label info is a number not one-hot vector as in training data
    sum_of_test_error = np.sum(np.argmax(a3, axis = 0) == np.argmax(val_y.T, axis = 0))

    # Report Test Error
    print("VAL_ACC:\t%5f" % (sum_of_test_error/val_set[0].shape[0]))

a0 = test_set[0].T
z2 = np.dot(W2, a0) + b2
a2 = us.sigmoid(z2)
z3 = np.dot(W3, a2) + b3
a3 = us.sigmoid(z3)

## Test Error
# in test data, label info is a number not one-hot vector as in training data
sum_of_test_error = np.sum(np.argmax(a3, axis = 0) == np.argmax(test_y.T, axis = 0))

# Report Test Error
print("VAL_ACC:\t%5f" % (sum_of_test_error/test_set[0].shape[0]))





















