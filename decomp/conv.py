from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

# Interface class
class Conv(object):
    def __init__(self, shape, name=None, stride=[1, 1, 1, 1], padding='SAME'):
        self.shape = shape
        self.name = name if name else 'conv'
        self.stride = stride
        self.padding = padding

    def conv(self):
        return self.graph

    def set_shape(self, shape):
        self.shape = shape

    def set_input(self, input):
        self.input = input

# Conv Depth, horizontal, vertical
class ConvSepDHV(Conv):
    def __init__(self, shape, name=None, init=[None, None, None], 
    	stride=[1, 1, 1, 1], padding='SAME'):
        super(ConvSepDHV, self).__init__(shape, name)
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        self.init = [i if i else tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype) for i in init]

    def set_input(self, input):
        self.input = input
        with tf.variable_scope(self.name) as scope:
            rows, cols, channels, filters = self.shape
            conv1_shape = [1, 1, channels, filters]
            conv2_shape = [1, cols, filters, 1]
            conv3_shape = [rows, 1, filters, 1]

            # kernel shape: 1, 1, channels, filters
            kernel1 = _variable_on_cpu('weights_d', conv1_shape, self.init[0])
            # kernel shape: rows, 1, filters, 1
            kernel2 = _variable_on_cpu('weights2_h', conv2_shape, self.init[1])
            # kernel shape: 1, cols, filters, 1
            kernel3 = _variable_on_cpu('weights3_v', conv3_shape, self.init[2])

            print(self.input.shape)
            conv1 = tf.nn.conv2d(self.input, kernel1, self.stride, padding=self.padding)
            print(conv1.shape)
            conv2 = tf.nn.depthwise_conv2d(conv1, kernel2, self.stride, padding=self.padding)
            print(conv2.shape)
            conv3 = tf.nn.depthwise_conv2d(conv2, kernel3, self.stride, padding=self.padding)
            print(conv3.shape)

            biases = _variable_on_cpu('biases', 
                [self.shape[-1]], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv3, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)
        self.graph = conv

# Conv horizontal, vertical
class ConvSepHV(Conv):
    def __init__(self, shape, name=None, init=[None, None], 
            stride=[1, 1, 1, 1], padding='SAME'):
        super(ConvSepHV, self).__init__(shape, name)
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        self.init = [i if i else tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype) for i in init]

    def set_input(self, input):
        self.input = input
        with tf.variable_scope(self.name) as scope:
            rows, cols, channels, filters = self.shape
            conv1_shape = [1, cols, channels, filters]
            conv2_shape = [rows, 1, channels]

            # kernel shape: rows, cols, channels, filters
            kernel1 = _variable_on_cpu('weights_h', conv1_shape, self.init[0])
            kernel2 = _variable_on_cpu('weights_v', 
                conv2_shape + [filters, 1], self.init[1])

            conv1 = tf.nn.depthwise_conv2d(self.input, kernel1, 
                self.stride, padding=self.padding)
            conv1_reshape_shape = list(conv1.shape[:3]) + [channels, filters]
            print(conv1_reshape_shape)
            conv1_reshape = tf.reshape(conv1, conv1_reshape_shape)

            conv2 = []
            for f in xrange(0, filters):
                conv2.append(tf.nn.conv2d(conv1_reshape[:,:,:,:, f], 
                    kernel2[:,:,:, f], self.stride, padding=self.padding))

            conv2_stack = tf.stack(conv2, axis=3)
            conv2_stack = tf.reshape(conv2_stack, conv2_stack.shape[:-1])
            print(conv2_stack.shape)

            biases = _variable_on_cpu('biases', 
                [self.shape[-1]], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv2_stack, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)
        self.graph = conv

class ConvSep2D(Conv):
    def __init__(self, shape, name=None, init=[None, None], 
            stride=[1, 1, 1, 1], padding='SAME'):
        super(ConvSep2D, self).__init__(shape, name)
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        self.init = [i if i else tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype) for i in init]

    def set_input(self, input):
        self.input = input
        with tf.variable_scope(self.name) as scope:
            # kernel shape: rows, cols, channels, filters
            rows, cols, channels, filters = self.shape
            kernel_depth_shape = [rows, cols, channels, 1]
            kernel_point_shape = [1, 1, channels, filters]
            kernel_depth = _variable_on_cpu('weights_d', kernel_depth_shape, self.init[0])
            kernel_point = _variable_on_cpu('weights_p', kernel_point_shape, self.init[1])
            conv = tf.nn.separable_conv2d(self.input, kernel_depth, kernel_point, 
                self.stride, padding=self.padding)
            biases = _variable_on_cpu('biases', 
                [self.shape[-1]], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)

        self.graph = conv

# Regular 3D convolution
class Conv3D(Conv):
    def __init__(self, shape, name=None, init=[None], 
            stride=[1, 1, 1, 1], padding='SAME'):
        super(Conv3D, self).__init__(shape, name)
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        self.init = [i if i else tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype) for i in init]

    def set_input(self, input):
        self.input = input
        with tf.variable_scope(self.name) as scope:
            # kernel shape: rows, cols, channels, filters
            kernel = _variable_on_cpu('weights', self.shape, self.init[0])
            conv = tf.nn.conv2d(self.input, kernel, self.stride, padding=self.padding)
            biases = _variable_on_cpu('biases', 
                [self.shape[-1]], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)

        self.graph = conv