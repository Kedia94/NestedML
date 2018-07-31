# Question 2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

# Code from NestedNet START
l2r = 1.0/2.0    # actual density: (l2r^2) *100
l1r = 1.0/4.0    # actual density: (l1r^2) *100

#temp variables
BN_DECAY = 0.999
BN_EPSILON = 0.00001

def create_variables(name, shape):

    n1 = np.sqrt(6. / (shape[0] * shape[1] * l1r * (shape[-2] + shape[-1])))
    n2 = np.sqrt(6. / (shape[0] * shape[1] * l2r * (shape[-2] + shape[-1])))
    n3 = np.sqrt(6. / (shape[0] * shape[1] * (shape[-2] + shape[-1])))

    shape1 = [shape[0], shape[1], int(l1r * shape[2]), int(l1r * shape[3])]
    shape2_1 = [shape[0], shape[1], int((l2r - l1r) * shape[2]), int(l1r * shape[3])]
    shape2_2 = [shape[0], shape[1], int(l2r * shape[2]), int((l2r - l1r) * shape[3])]
    shape3_1 = [shape[0], shape[1], int((1. - l2r) * shape[2]), int(l2r * shape[3])]
    shape3_2 = [shape[0], shape[1], shape[2], int((1. - l2r) * shape[3])]

    lv1_variables = tf.get_variable(name + '_l1', initializer=tf.random_uniform(shape1, -n3, n3, tf.float32, seed=None))
    lv2_1_variables = tf.get_variable(name + '_l2_1', initializer=tf.random_uniform(shape2_1, -n3, n3, tf.float32, seed=None))
    lv2_2_variables = tf.get_variable(name + '_l2_2', initializer=tf.random_uniform(shape2_2, -n3, n3, tf.float32, seed=None))
    lv3_1_variables = tf.get_variable(name + '_l3_1', initializer=tf.random_uniform(shape3_1, -n3, n3, tf.float32, seed=None))
    lv3_2_variables = tf.get_variable(name + '_l3_2', initializer=tf.random_uniform(shape3_2, -n3, n3, tf.float32, seed=None))

    return lv1_variables, lv2_1_variables, lv2_2_variables, lv3_1_variables, lv3_2_variables

def output_layer(input1, input2, input3, num_labels):

    input_dim1 = input1.get_shape().as_list()[-1]
    input_dim2 = input2.get_shape().as_list()[-1]
    input_dim3 = input3.get_shape().as_list()[-1]

    fc_w1 = tf.get_variable('fc_weights_l1', shape=[input_dim1, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w2 = tf.get_variable('fc_weights_l2', shape=[input_dim2, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w3 = tf.get_variable('fc_weights_l3', shape=[input_dim3, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))

    fc_b1 = tf.get_variable(name='fc_bias_l1', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b2 = tf.get_variable(name='fc_bias_l2', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b3 = tf.get_variable(name='fc_bias_l3', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h1 = tf.matmul(input1, fc_w1) + fc_b1
    fc_h2 = tf.matmul(input2, fc_w2) + fc_b2
    fc_h3 = tf.matmul(input3, fc_w3) + fc_b3
    return fc_h1, fc_h2, fc_h3

def batch_normalization_layer(name, input_layer, dimension, is_training=True):

    beta = tf.get_variable(name + 'beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name + 'gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    mu = tf.get_variable(name + 'mu', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
    sigma = tf.get_variable(name + 'sigma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

    if is_training is True:
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        train_mean = tf.assign(mu, mu * BN_DECAY + mean * (1 - BN_DECAY))
        train_var = tf.assign(sigma, sigma * BN_DECAY + variance * (1 - BN_DECAY))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    else:
        bn_layer = tf.nn.batch_normalization(input_layer, mu, sigma, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_layer(input_layer, filter_shape, stride, is_training):

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer('l1_l2_l3', input_layer, in_channel, is_training)
    bn_layer = tf.nn.relu(bn_layer)

    n1 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l1r * filter_shape[-1])))
    n2 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l2r * filter_shape[-1])))
    n3 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + filter_shape[-1])))
    filter1 = tf.get_variable('conv_l1',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l1r*filter_shape[3])],
                                                            -n3, n3,   tf.float32, seed=None))
    filter2 = tf.get_variable('conv_l2',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int((l2r-l1r)*filter_shape[3])],
                                                            -n3, n3,   tf.float32, seed=None))
    filter3 = tf.get_variable('conv_l3',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int((1.-l2r)*filter_shape[3])],
                                                            -n3, n3,   tf.float32, seed=None))

    conv1 = tf.nn.conv2d(bn_layer, filter1, strides=[1, stride, stride, 1], padding='SAME')
    conv2 = tf.concat((conv1, tf.nn.conv2d(bn_layer, filter2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((conv2, tf.nn.conv2d(bn_layer, filter3, strides=[1, stride, stride, 1], padding='SAME')), 3)

    return conv1, conv2, conv3


def bn_relu_conv_layer(input1, input2, input3, filter_shape, stride, is_training):

    in_channel1 = input1.get_shape().as_list()[-1]
    in_channel2 = input2.get_shape().as_list()[-1]
    in_channel3 = input3.get_shape().as_list()[-1]

    bn_layer1 = batch_normalization_layer('l1', input1, in_channel1, is_training)
    bn_layer1 = tf.nn.relu(bn_layer1)
    bn_layer2 = batch_normalization_layer('l2', input2, in_channel2, is_training)
    bn_layer2 = tf.nn.relu(bn_layer2)
    bn_layer3 = batch_normalization_layer('l3', input3, in_channel3, is_training)
    bn_layer3 = tf.nn.relu(bn_layer3)

    filter1, filter2_1, filter2_2, filter3_1, filter3_2 = create_variables(name='conv', shape=filter_shape)

    conv1 = tf.nn.conv2d(bn_layer1, filter1, strides=[1, stride, stride, 1], padding='SAME')
    conv2 = tf.concat((tf.add(tf.nn.conv2d(bn_layer2[:, :, :, :int(l1r * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer2[:, :, :, int(l1r * filter_shape[2]):int(l2r * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1],
                                           padding='SAME')),
                       tf.nn.conv2d(bn_layer2, filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((tf.add(tf.nn.conv2d(bn_layer3[:, :, :, :int(l1r * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer3[:, :, :, int(l1r * filter_shape[2]):int(l2r * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1],
                                           padding='SAME')),
                       tf.nn.conv2d(bn_layer3[:, :, :, :int(l2r * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((tf.add(conv3, tf.nn.conv2d(bn_layer3[:, :, :, int(l2r * filter_shape[2]):], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer3, filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    return conv1, conv2, conv3


def residual_block(input1, input2, input3, output_channel, wide_scale, is_training, first_block=False):

    input_channel = input3.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    output_channel = int(output_channel * wide_scale)

    if input_channel * wide_scale == output_channel:
        increase_dim = True
        stride = 1
    else:
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            conv1, conv2, conv3 = conv_layer(input1, [3, 3, input_channel, output_channel], stride, is_training)
        else:
            conv1, conv2, conv3 = bn_relu_conv_layer(input1, input2, input3, [3, 3, input_channel, output_channel], stride, is_training)

    with tf.variable_scope('conv2_in_block'):
        conv1, conv2, conv3 = bn_relu_conv_layer(conv1, conv2, conv3, [3, 3, output_channel, output_channel], 1, is_training)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        if input_channel * wide_scale == output_channel:
            if first_block:
                np0 = int((output_channel * l1r - input_channel) / 2)
                np1 = int((output_channel * l2r - input_channel) / 2)
                np2 = int((output_channel * 1 - input_channel) / 2)
                padded_input1 = tf.pad(input1, [[0, 0], [0, 0], [0, 0], [np0, np0]])
                padded_input2 = tf.pad(input2, [[0, 0], [0, 0], [0, 0], [np1, np1]])
                padded_input3 = tf.pad(input3, [[0, 0], [0, 0], [0, 0], [np2, np2]])
            else:
                np1 = int((output_channel - input_channel) / 2 * l1r)
                np2 = int((output_channel - input_channel) / 2 * l2r)
                np3 = int((output_channel - input_channel) / 2)
                padded_input1 = tf.pad(input1, [[0, 0], [0, 0], [0, 0], [np1, np1]])
                padded_input2 = tf.pad(input2, [[0, 0], [0, 0], [0, 0], [np2, np2]])
                padded_input3 = tf.pad(input3, [[0, 0], [0, 0], [0, 0], [np3, np3]])
        else:
            pooled_input1 = tf.nn.avg_pool(input1, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input1 = tf.pad(pooled_input1, [[0, 0], [0, 0], [0, 0], [int(input_channel*l1r) // 2,
                                                                            int(input_channel*l1r) // 2]])
            pooled_input2 = tf.nn.avg_pool(input2, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input2 = tf.pad(pooled_input2, [[0, 0], [0, 0], [0, 0], [int(input_channel*l2r) // 2,
                                                                            int(input_channel*l2r) // 2]])
            pooled_input3 = tf.nn.avg_pool(input3, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input3 = tf.pad(pooled_input3, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                            input_channel // 2]])
    else:
        padded_input1 = input1
        padded_input2 = input2
        padded_input3 = input3

    output1 = conv1 + padded_input1
    output2 = conv2 + padded_input2
    output3 = conv3 + padded_input3

    return output1, output2, output3

# Code from NestedNet END

def bias_variable(shape, start_val=0.1):
    initialization = tf.constant(start_val, shape=shape)
    return tf.Variable(initialization)

def nested_relu(input1, input2, input3, size):
	size1 = int(size*l1r)
	size2 = int(size*l2r) - int(size*l1r)
	size3 = size - int(size*l2r)

	bias1 = bias_variable(shape=(size1,))
	bias2 = tf.concat((bias1, bias_variable(shape=(size2,))),0)
	bias3 = tf.concat((bias2, bias_variable(shape=(size3,))),0)

	relu1 = tf.nn.relu(input1 + bias1)
	relu2 = tf.nn.relu(input2 + bias2)
	relu3 = tf.nn.relu(input3 + bias3)
	return relu1, relu2, relu3

def nested_pool(input1, input2, input3):
	pool1 = tf.nn.max_pool(input1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool2 = tf.nn.max_pool(input2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool3 = tf.nn.max_pool(input3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	return pool1, pool2, pool3

def nested_drop(input1, input2, input3, keep_prob):
	drop1 = tf.nn.dropout(input1, keep_prob=keep_prob)
	drop2 = tf.nn.dropout(input2, keep_prob=keep_prob)
	drop3 = tf.nn.dropout(input3, keep_prob=keep_prob)
	return drop1, drop2, drop3

def inference(x, n_classes, keep_prob, is_training):
    
	layers1 = []
	layers2 = []
	layers3 = []

	conv1_out = 64
	with tf.variable_scope('conv1_in_block'):
		conv1_1, conv1_2, conv1_3 = conv_layer(x, [3,3,1,conv1_out], 1, is_training)

	relu1_1, relu1_2, relu1_3 = nested_relu(conv1_1, conv1_2, conv1_3, conv1_out)

	pool1_1, pool1_2, pool1_3 = nested_pool(relu1_1, relu1_2, relu1_3)
	drop1_1, drop1_2, drop1_3 = nested_drop(pool1_1, pool1_2, pool1_3, keep_prob)

	conv2_out = 128
	with tf.variable_scope('conv2_in_block'):
		conv2_1, conv2_2, conv2_3 = bn_relu_conv_layer(drop1_1, drop1_2, drop1_3, [3, 3, conv1_out, conv2_out], 1, is_training)

	relu2_1, relu2_2, relu2_3 = nested_relu(conv2_1, conv2_2, conv2_3, conv2_out)

	pool2_1, pool2_2, pool2_3 = nested_pool(relu2_1, relu2_2, relu2_3)
	drop2_1, drop2_2, drop2_3 = nested_drop(pool2_1, pool2_2, pool2_3, keep_prob)

	fc0_1 = tf.concat([flatten(drop1_1), flatten(drop2_1)], 1)
	fc0_2 = tf.concat([flatten(drop1_2), flatten(drop2_2)], 1)
	fc0_3 = tf.concat([flatten(drop1_3), flatten(drop2_3)], 1)

	fc1_out = 64
	with tf.variable_scope('fc1_in_block'):
		fc1_1, fc1_2, fc1_3 = output_layer(fc0_1, fc0_2, fc0_3, fc1_out)

	drop_fc1_1, drop_fc1_2, drop_fc1_3 = nested_drop(fc1_1, fc1_2, fc1_3, keep_prob)

	fc2_out = n_classes
	with tf.variable_scope('fc2_in_block'):
		logits1, logits2, logits3 = output_layer(drop_fc1_1, drop_fc1_2, drop_fc1_3, fc2_out)

	return logits1, logits2, logits3
    



def weight_variable(shape, mu=0, sigma=0.1):
    initialization = tf.truncated_normal(shape=shape, mean=mu, stddev=sigma)
    return tf.Variable(initialization)



def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# network architecture definition
def my_net(x, n_classes, keep_prob):

    c1_out = 64
    conv1_W = weight_variable(shape=(3, 3, 1, c1_out))
    conv1_b = bias_variable(shape=(c1_out,))
    conv_temp = conv2d(x, conv1_W)
    conv1 = tf.nn.relu(conv2d(x, conv1_W) + conv1_b)

    pool1 = max_pool_2x2(conv1)

    drop1 = tf.nn.dropout(pool1, keep_prob=keep_prob)

    c2_out = 128
    conv2_W = weight_variable(shape=(3, 3, c1_out, c2_out))
    conv2_b = bias_variable(shape=(c2_out,))
    conv2 = tf.nn.relu(conv2d(drop1, conv2_W) + conv2_b)

    pool2 = max_pool_2x2(conv2)

    drop2 = tf.nn.dropout(pool2, keep_prob=keep_prob)

    fc0 = tf.concat([flatten(drop1), flatten(drop2)], 1)

    fc1_out = 64
    fc1_W = weight_variable(shape=(fc0._shape[1].value, fc1_out))
    fc1_b = bias_variable(shape=(fc1_out,))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    drop_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    fc2_out = n_classes
    fc2_W = weight_variable(shape=(drop_fc1._shape[1].value, fc2_out))
    fc2_b = bias_variable(shape=(fc2_out,))
    logits = tf.matmul(drop_fc1, fc2_W) + fc2_b

    return logits
