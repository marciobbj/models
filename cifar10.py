import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from helpers import ConvolutionHelpers

_ = ConvolutionHelpers()

DATA_PATH = './cifar10'


def data_loader(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        files_dict = pickle.load(fo, encoding='bytes')
        return files_dict


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def display_cifar(images, size):
    n = len(images)
    plt.figure
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack(
        [images[np.random.choice(n)] for i in range(size)]
    ) for i in range(size)])
    plt.imshow(im)
    plt.show()


class CifarLoader:

    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [data_loader(f) for f in self._source]
        images = np.vstack([d[b'data'] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(
            0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d[b'labels'] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i: self._i +
                           batch_size], self.labels[self._i: self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y


class CifarDataManager:

    def __init__(self):
        self.train = CifarLoader(
            [f'data_batch_{i}' for i in range(1, 6)]).load()
        self.test = CifarLoader(['test_batch']).load()


cifar = CifarDataManager()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)
conv1 = _.conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = _.max_pool_2x2(conv1)

conv2 = _.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = _.max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

full_1 = tf.nn.relu(_.full_layer(conv2_flat, 1024))
full_1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = _.full_layer(full_1_drop, 10)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={
                  x: X[i], y_:Y[i], keep_prob:1.0}) for i in range(10)])
    print('accuracy: {}'.format(acc*100))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = cifar.train.next_batch(50)
        train_accuracy = sess.run(train_step, feed_dict={
                                  x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('step {}, training accuracy {}'.format(i, train_accuracy))
    test(sess)
