import numpy as np
from sklearn.utils import shuffle

# this module preprocesses data and trains the model using gradient descent on simple neural network with one layer


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    # calculates gradients od weights and biases (dw and db) and cost after one forward propagation

    m = Y.shape[1]
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    dz = A - Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return dw, db, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    # implements gradient descent and prints cost after each 100 iterations

    costs = []

    for i in range(num_iterations):
        dw, db, cost = propagate(w, b, X, Y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return w, b, costs


def predict(w, b, X):
    # for given weights and biases computes prediction

    m = X.shape[1]
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    pred = np.zeros((1, A.shape[1]))

    for i in range(A.shape[1]):
        pred[0, i] = np.round(A[0, i])

    assert (pred.shape == (1, m))

    return pred


def model(X_train, X_test, Y_train, Y_test, num_iterations, learning_rate, print_cost=False):
    # combines whole training process in one function, also computes train and test accuracy

    w, b = initialize_with_zeros(train_x.shape[0])

    w, b, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    train_predict = predict(w, b, X_train)
    test_predict = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predict - Y_test)) * 100))
    print(Y_train[0:10], train_predict[0:10])
    return w, b


# PREPROCESSING

X = np.load('train_x.npy')
Y = np.load('train_y.npy')
Y = np.transpose(Y)

# here we set number of images we want to train on (maximum is given by total number of random images, since there are
# only 2000 of those compared to 4000 cat images)
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

train_x_flat = train_x.reshape(train_x.shape[0], -1)
test_x_flat = test_x.reshape(test_x.shape[0], -1)

train_x = train_x_flat / 255.
test_x = test_x_flat / 255.
train_x = train_x.T
test_x = test_x.T
train_y = train_y.T
test_y = test_y.T

# TRAINING AND SAVING THE MODEL
w, b = model(train_x, test_x, train_y, test_y, num_iterations=10000, learning_rate=0.001, print_cost=True)
np.save('weights', w)
np.save("biases", b)
