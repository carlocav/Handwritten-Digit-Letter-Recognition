### 3rd neural network (cnn accuracy >88%) 

library(tensorflow)
labels <- c(0:9, LETTERS, 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't')

log_dir <- function(prefix="") {
  now = format(Sys.time(), "%Y%m%d%H%M%S")
  root_logdir = "model"
  name = paste0(root_logdir, "/", prefix, "-run-", now, "/")
  return (name)
}
log_dir("mnist_dnn")

n_inputs = 28*28
n_hidden1 = 400
n_hidden2 = 200
n_outputs = 47
learning_rate = 0.01
dropout_rate = 0.3

tf$reset_default_graph()
logdir = log_dir("emnist_dnn")

X = tf$placeholder(tf$float32, shape=list(NULL, 28*28), name="X")
X_reshaped = tf$reshape(X, shape = as.integer(c(-1, 28, 28, 1)))
y = tf$placeholder(tf$int32, shape=list(NULL), name="y")
training = tf$placeholder_with_default(FALSE, shape=list(), name='training')

X_drop = tf$layers$dropout(X_reshaped, dropout_rate, training=training)

with (tf$name_scope("cnn"), {
  conv1 = tf$layers$conv2d(inputs=X_drop, filters=25, kernel_size = list(5, 5), padding="same", activation=tf$nn$relu)
  pool1 = tf$layers$max_pooling2d(inputs=conv1, pool_size = as.integer(c(2, 2)), strides=as.integer(2))
  conv2 = tf$layers$conv2d(inputs=pool1, filters=50, kernel_size = list(5, 5), padding="same", activation=tf$nn$relu)
  pool2 = tf$layers$max_pooling2d(inputs=conv2, pool_size = as.integer(c(2, 2)), strides=as.integer(2))
  conv3 = tf$layers$conv2d(inputs=pool2, filters=100, kernel_size = list(5, 5), padding="same", activation=tf$nn$relu)
})

with (tf$name_scope("dnn"), {
  to_flat = tf$reshape(conv3, as.integer(c(-1, 7 * 7 * 100)))
  dense1 = tf$layers$dense(to_flat, 1000, name="dense1", activation=tf$nn$relu)
  dense1_drop = tf$layers$dropout(dense1, dropout_rate, training=training)
  dense2 = tf$layers$dense(dense1_drop, 500, name="dense2", activation=tf$nn$relu)
  dense2_drop = tf$layers$dropout(dense2, dropout_rate, training=training)
  logits = tf$layers$dense(dense2_drop, 47, name="outputs")
})

with (tf$name_scope("loss"), {
  xentropy = tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
  loss = tf$reduce_mean(xentropy, name="loss")
  # loss_summary = tf$summary$scalar('log_loss', loss)
})

with (tf$name_scope("eval"), {
  y_proba = tf$nn$softmax(logits)
  # correct = tf$nn$in_top_k(logits, y, 1)
  # accuracy = tf$reduce_mean(tf$cast(correct, tf$float32))
})

init = tf$global_variables_initializer()
saver = tf$train$Saver()


predict <- function(m) {
  sess <- tf$Session() 
  sess$run(init)
  saver$restore(sess, "model/my_deep_emnist_model")
  pred <- sess$run(logits, feed_dict = dict(X = m))
  y_proba <- sess$run(y_proba, feed_dict= dict(X = m))
  sess$close()
  #return (list(labels[apply(pred,1,which.max)], y_proba)[[1]])
  return (labels[apply(pred,1,which.max)])
}

# sess <- tf$Session()
# accuracy_val <-  sess$run(accuracy, feed_dict = dict(X = X_test, y = c(y_test)))
# saver$restore(sess, "model/my_deep_emnist_model")
# accuracy_val