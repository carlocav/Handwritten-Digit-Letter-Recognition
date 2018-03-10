#### to open tensorboard
## python -m tensorboard.main --logdir tf_logs/
# poi -> http://0.0.0.0:6006

import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.metrics import log_loss
import tensorflow as tf


# ----- EMNIST Balanced -----

from scipy import io as spio
emnist = spio.loadmat("matlab/emnist-balanced.mat")

X_train = emnist["dataset"][0][0][0][0][0][0][:94000]
X_val = emnist["dataset"][0][0][0][0][0][0][94000:]
X_test = emnist["dataset"][0][0][1][0][0][0]

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
# normalize
X_train /= 255
X_val /= 255
X_test /= 255

y_train = emnist["dataset"][0][0][0][0][0][1][:94000].astype("int").reshape(-1,)
y_val = emnist["dataset"][0][0][0][0][0][1][94000:].astype("int").reshape(-1,)
y_test = emnist["dataset"][0][0][1][0][0][1].astype("int").reshape(-1,)

labels = list(range(10))
labels.extend(list(map(chr, range(65, 91))))
labels.extend(['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'])


# ----- EMNIST Digits -----

from scipy import io as spio
emnist = spio.loadmat("matlab/emnist-digits.mat")

X_train = emnist["dataset"][0][0][0][0][0][0][:200000]
X_val = emnist["dataset"][0][0][0][0][0][0][200000:]
X_test = emnist["dataset"][0][0][1][0][0][0]

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
# normalize
X_train /= 255
X_val /= 255
X_test /= 255

y_train = emnist["dataset"][0][0][0][0][0][1][:200000].astype("int").reshape(-1,)
y_val = emnist["dataset"][0][0][0][0][0][1][200000:].astype("int").reshape(-1,)
y_test = emnist["dataset"][0][0][1][0][0][1].astype("int").reshape(-1,)


# ----- MNIST -----

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images
X_val = mnist.validation.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_val = mnist.validation.labels.astype("int")
y_test = mnist.test.labels.astype("int")