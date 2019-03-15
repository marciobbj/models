import tensorflow as tf

class ConvolutionHelpers:

    def full_layer(self, input, size):
        in_size = int(input.get_shape()[1])
        W = self._variable_weight([in_size, size])
        b = self._variable_bias([size])
        return tf.matmul(input, W) + b
    
    def conv_layer(self, input, shape):
        W = self._variable_weight(shape)
        b = self._variable_bias([shape[3]])
        return tf.nn.relu(self._conv2d(input, W) + b)
    
    def _variable_weight(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def _variable_bias(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
