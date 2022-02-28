import numpy as np
from sklearn.utils import shuffle

# This module implements neural network with one hidden layer, to classify cats vs. random images.


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def preprocessing():

    # This function deals with preprocessing raw images which are stored in form of numpy arrays in files 'train_x.npy'
    # and 'train_y.npy'. Function outputs train and test sets of inputs and outputs for neural network:
    # - train_x is array of shape (n_x, m), where n_x is number of pixel an image has (in this case its 64*64*3) and
    #   m is number of training examples, which is number around 1600
    # - train_y is array of shape (1, m), it contains labels for images in train_x (1 if it is a cat image, 0 otherwise)
    # - test_x and test_y are same as train_x/y, they just contain around 400 examples

    X = np.load('train_x.npy')
    Y = np.load('train_y.npy')
    Y = np.transpose(Y)

    # here we set number of images we want to train on (maximum is given by total number of random images, since there
    # are only 2000 of those compared to 4000 cat images)

    n_images = 1000
    n_cats = 4000

    tmp1 = X[0:n_images]
    tmp2 = X[n_cats:n_cats + n_images]
    X = np.concatenate((tmp1, tmp2), axis=0)
    tmp1 = Y[0:n_images]
    tmp2 = Y[n_cats:n_cats + n_images]
    Y = np.concatenate((tmp1, tmp2), axis=0)

    X, Y = shuffle(X, Y, random_state=0)

    # split the data into train and test sets
    mask = np.random.rand(len(Y)) <= 0.8
    train_x = X[mask]
    test_x = X[~mask]
    train_y = Y[mask]
    test_y = Y[~mask]

    # convert the data into array of shape (64*64*3, m)
    train_x_flat = train_x.reshape(train_x.shape[0], -1)
    test_x_flat = test_x.reshape(test_x.shape[0], -1)

    train_x = train_x_flat / 255.
    test_x = test_x_flat / 255.
    train_x = train_x.T
    test_x = test_x.T
    train_y = train_y.T
    test_y = test_y.T

    return train_x, train_y, test_x, test_y


def initialize(n_x, n_h):

    # This function sets start values for weights and biases. We use random initialization

    w1 = np.random.randn(n_h, n_x) * 0.01
    w2 = np.random.randn(1, n_h) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    b2 = np.zeros(shape=(1, 1))

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (1, n_h))
    assert (b2.shape == (1, 1))

    w = {'w1': w1,
         'w2': w2}
    b = {'b1': b1,
         'b2': b2}

    return w, b


def forward_propagation(X, w, b):
    w1 = w['w1']
    w2 = w['w2']
    b1 = b['b1']
    b2 = b['b2']
    m = X.shape[1]

    z1 = np.dot(w1, X) + b1
    A1 = np.tanh(z1)
    z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(z2)

    assert (A2.shape == (1, X.shape[1]))

    computed = {'z1': z1,
                'A1': A1,
                'z2': z2,
                'A2': A2}

    return computed


def backward_propagation(X, Y, w, b, computed):
    m = X.shape[1]
    A2 = computed['A2']
    A1 = computed['A1']
    w2 = w['w2']

    dz2 = A2 - Y
    dw2 = (1 / m) * np.dot(dz2, A1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(A1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    dw = {'dw1': dw1,
          'dw2': dw2}
    db = {'db1': db1,
          'db2': db2}

    return dw, db


def update_parameters(w, b, dw, db, learning_rate):
    w1, w2, b1, b2 = w['w1'], w['w2'], b['b1'], b['b2']
    dw1, dw2, db1, db2 = dw['dw1'], dw['dw2'], db['db1'], db['db2']

    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    w = {'w1': w1,
         'w2': w2}
    b = {'b1': b1,
         'b2': b2}

    return w, b


def predict(X, w, b):
    computed = forward_propagation(X, w, b)
    A2 = computed['A2']
    pred = np.round(A2)

    return pred


def model(X, Y, test_X, test_Y, n_x, n_h, learning_rate=0.03, num_iterations=2000, print_cost=False):
    w, b = initialize(n_x, n_h)
    m = train_x.shape[1]

    for i in range(num_iterations):
        computed = forward_propagation(X, w, b)
        dw, db = backward_propagation(X, Y, w, b, computed)
        w, b = update_parameters(w, b, dw, db, learning_rate)

        if print_cost and i % 100 == 0:
            A2 = computed['A2']
            cost = (- 1 / m) * np.sum(np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2)))
            print("The cost after {} iterations is: {}".format(i, cost))

    train_predict = predict(X, w, b)
    test_predict = predict(test_X, w, b)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predict - Y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predict - test_Y)) * 100))

    return w, b


train_x, train_y, test_x, test_y = preprocessing()
n_x = train_x.shape[0]
n_h = 10

w, b = model(train_x, train_y, test_x, test_y, n_x, n_h, learning_rate=0.03, num_iterations=2500, print_cost=True)

w1 = w['w1']
b1 = b['b1']
w2 = w['w2']
b2 = b['b2']

np.save('w1_0.02', w1)
np.save('w2_0.02', w2)
np.save('b1_0.02', b1)
np.save('b2_0.02', b2)
