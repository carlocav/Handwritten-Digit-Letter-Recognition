library(R.matlab)

emnist = readMat("matlab/emnist-balanced.mat")
X_train = emnist["dataset"]$dataset[[1]][[1]][1:94000,]
X_val = emnist["dataset"]$dataset[[1]][[1]][94001:112800,]
X_test = emnist["dataset"]$dataset[[2]][[1]]
# normalize
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

y_train = emnist["dataset"]$dataset[[1]][[2]][1:94000,]
y_val = emnist["dataset"]$dataset[[1]][[2]][94001:112800,]
y_test = emnist["dataset"]$dataset[[2]][[2]]

m1 = matrix(X_train[45336,],28,28,byrow = T)
m1 = m1[1:nrow(m1),ncol(m1):1]
image(m1, col= grey.colors(100, 1, 0, gamma = 0.5), add=TRUE)