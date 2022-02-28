import numpy as np
from PIL import Image
import os

# this module takes cat images and random images and puts them in one numpy array "train_x" along with labels for those
# images stored in "train_y" (1 if image is that of cat, 0 otherwise)


directory_cats = 'training_set\\training_set\\cats'
directory_rand = 'images'

n_cats = len(os.listdir(directory_cats)) - 1  # minus one because there is junk file _DS_Store in cats dir
n_rand = len(os.listdir(directory_rand))
n_train = n_rand + n_cats
train_y = np.zeros((1, n_train))
train_y[0][0:n_cats] = 1  # label cat images as 1
train_y[0][n_cats:n_train] = 0  # label random images as 0

train_x = np.zeros((n_train, 64, 64, 3))  # train_x contains all images used for training the model
i = 0
for filename in os.listdir(directory_cats):
    if filename == '_DS_Store':
        continue
    image = Image.open('training_set\\training_set\\cats\\' + filename)
    image = image.resize((64, 64))
    data = np.asarray(image)
    train_x[i] = data
    i = i + 1
j = 0
for filename in os.listdir(directory_rand):
    image = Image.open('images\\' + filename)
    image = image.resize((64, 64))
    if image.mode == 'L':  # some images aren't in RGB, we skip those
        continue
    data = np.asarray(image)
    train_x[i+j] = data
    j = j + 1

np.save('train_x', train_x)
np.save('train_y', train_y)
