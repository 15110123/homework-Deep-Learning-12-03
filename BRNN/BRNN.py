import data
import utils
import testutils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Tắt log
tf.logging.set_verbosity(tf.logging.ERROR)

x_train, y_train, x_valid, y_valid = data.load_data(mode='train')
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))

num_input = 28          # Kích thước input 
timesteps = 28          # Số step 
n_classes = 10          # Số lớp 

learning_rate = 0.001 # Learning rate (thường ký hiệu là a) 
epochs = 2           # Số epochs 
batch_size = 200      # Kích thước batch 
display_freq = 100    # Mật độ xuất hiện trạng thái huấn luyện 

num_hidden_units = 128  # Số lượng hidden unit (kích thước output của một layer). 

x = tf.placeholder(tf.float32, shape=[None, timesteps, num_input], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

# Tạo weight matrix 
W = utils.weight_variable(shape=[2*num_hidden_units, n_classes])

# Tạo bias vector 
b = utils.bias_variable(shape=[n_classes])

output_logits = utils.BiRNN(x, W, b, timesteps, num_hidden_units)
y_pred = tf.nn.softmax(output_logits)

# Model predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Định nghĩa hàm loss, optimizer, và accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Tạo global variable cho graph 
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
# Iterations cho mỗi epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    x_train, y_train = data.randomize(x_train, y_train)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = data.get_next_batch(x_train, y_train, start, end)
        x_batch = x_batch.reshape((batch_size, timesteps, num_input))
        # back propagation 
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Tính và hiện loss và accuracy 
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))

    # Chạy validation sau mỗi epoch

    feed_dict_valid = {x: x_valid[:1000].reshape((-1, timesteps, num_input)), y: y_valid[:1000]}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')

# ------------------------------------
# Giai đoạn test

# Test với 1000 sample
x_test, y_test = data.load_data(mode='test')
feed_dict_test = {x: x_test[:1000].reshape((-1, timesteps, num_input)), y: y_test[:1000]}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')

# Plot some of the correct and misclassified examples
cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
cls_true = np.argmax(y_test, axis=1)
testutils.plot_images(x_test, cls_true, cls_pred, title='Correct Examples')
testutils.plot_example_errors(x_test[:1000], cls_true[:1000], cls_pred, title='Misclassified Examples')
plt.show()