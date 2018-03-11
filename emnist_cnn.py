from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


def random_batch(X_train, y_train, batch_size):
    #SRSWR
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train.reshape(-1,1)[rnd_indices]
    return X_batch, y_batch.reshape(-1,)

# test
X_batch, y_batch = random_batch(X_train, y_train, 5)
X_batch
y_batch


# CONSTRUCTION PHASE

tf.reset_default_graph()
logdir = log_dir("emnist_dnn")

X = tf.placeholder(tf.float32, shape=(None, 28 * 28), name="X")
X_reshaped = tf.reshape(X, shape=[-1, 28, 28, 1])
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.3
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.1
global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

X_drop = tf.layers.dropout(X_reshaped, dropout_rate, training=training)

with tf.name_scope("cnn"):
    conv1 = tf.layers.conv2d(inputs=X_drop, filters=25, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # output 28x28x25
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # output 14x14x25
    conv2 = tf.layers.conv2d(inputs=pool1, filters=50, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # output 14x14x50
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # output 7x7x50
    conv3 = tf.layers.conv2d(inputs=pool2, filters=100, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # output 7x7x100

with tf.name_scope("dnn"):
    to_flat = tf.reshape(conv3, [-1, 7 * 7 * 100])
    dense1 = tf.layers.dense(to_flat, 1000, name="dense1", activation=tf.nn.relu)
    dense1_drop = tf.layers.dropout(dense1, dropout_rate, training=training)
    dense2 = tf.layers.dense(dense1_drop, 500, name="dense2", activation=tf.nn.relu)
    dense2_drop = tf.layers.dropout(dense2, dropout_rate, training=training)
    logits = tf.layers.dense(dense2_drop, 47, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)

with tf.name_scope("eval"):
    y_proba = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

train_summary_writer = tf.summary.FileWriter(os.path.join(logdir, "train"), tf.get_default_graph())
validation_summary_writer = tf.summary.FileWriter(os.path.join(logdir, "validation"), tf.get_default_graph())


# EXECUTION PHASE

n_epochs = 100
batch_size = 100

checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "model/my_deep_emnist_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 30

# Inside the with block, the session is set as the default session.
with tf.Session() as sess:
    init.run()  # equivalent to calling tf.get_default_session().run(init) i.e. sess.run(init)

    for epoch in range(n_epochs):
        for iteration in range(len(y_train) // batch_size):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        # X_val_batch, y_val_batch = random_batch(X_val, y_val, 10000)
        acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
        loss_val = loss.eval(feed_dict={X: X_val, y: y_val})
        # summary_loss = loss_summary.eval(feed_dict={X: X_train, y: y_train})
        # summary_acc_train = accuracy_summary.eval(feed_dict={X: X_train, y: y_train})
        summary_acc_test = accuracy_summary.eval(feed_dict={X: X_val, y: y_val})
        # train_summary_writer.add_summary(summary_loss, epoch)
        # train_summary_writer.add_summary(summary_acc_train, epoch)
        validation_summary_writer.add_summary(summary_acc_test, epoch)

        print("Epoch:", epoch,
            "\tValidation accuracy: {:.3f}%".format(acc_val * 100),
            "\tLoss: {:.5f}".format(loss_val))
        saver.save(sess, checkpoint_path)
        with open(checkpoint_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))
        if loss_val < best_loss:
            saver.save(sess, final_model_path)
            best_loss = loss_val
        else:
            epochs_without_progress += 1
            if epochs_without_progress > max_epochs_without_progress:
                print("Early stopping")
                break

os.remove(checkpoint_epoch_path)

train_summary_writer.close()
validation_summary_writer.close()

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
    correct = correct.eval(feed_dict={X: X_test, y: y_test})
    y_proba = y_proba.eval(feed_dict={X: X_test, y: y_test})

accuracy_val
my_correct = [labels[np.argmax(y_proba[i,])]==labels[y_test[i,]] for i in range(len(y_test))]
y_pred = np.argmax(y_proba, axis = 1)

accuracy_score(y_test, y_pred)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix_pd = pd.DataFrame(cnf_matrix, index=labels, columns=labels)

import seaborn as sn
import matplotlib.pyplot as plt

plt.figure(figsize = (20,14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(cnf_matrix_pd, annot=True,annot_kws={"size": 5}, fmt='g')    # font size