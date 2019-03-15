import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from helpers import ConvolutionHelpers

DATA_DIR = '/tmp/data'
_ = ConvolutionHelpers()

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Model Definition
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = _.conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = _.max_pool_2x2(conv1)
conv2 = _.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = _.max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])

full_1 = tf.nn.relu(_.full_layer(conv2_flat, 1024))
keep_prob = tf.placeholder(tf.float32)
full_1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = _.full_layer(full_1_drop, 10)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_conv, logits=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(
                accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step {}, training accuracy {}'.format(i, train_accuracy))
        sess.run(train_step, feed_dict={
                 x: batch[0], y_: batch[1], keep_prob: 0.5})
    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean([sess.run(accuracy, feed_dict={
                            x: X[i], y_:Y[i], keep_prob:1.0}) for i in range(10)])
    print('test accuracy: {:.2}%'.format(test_accuracy))
