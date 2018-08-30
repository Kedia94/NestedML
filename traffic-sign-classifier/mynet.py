# Question 2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

# Code from NestedNet START
l2r = 1.0/2.0    # actual density: (l2r^2) *100
l1r = 1.0/4.0    # actual density: (l1r^2) *100

l10r1 = 1.0 / 64.0
l10r2 = 1.0 / 32.0
l10r3 = 1.0 / 16.0
l10r4 = 1.0 / 8.0
l10r5 = 1.0 / 4.0
l10r6 = 1.0 / 3.0
l10r7 = 1.0 / 2.0
l10r8 = 2.0 / 3.0
l10r9 = 3.0 / 4.0

n = 10  																								# n-level
lnr = [1.0/64.0, 1.0/32.0, 1.0/20.0, 1.0/16.0, 1.0/12.0, 1.0/8.0, 1.0/6.0, 1.0/4.0, 1.0/2.0, 1.0/1.0] # incremental order, last one is 1, n-array

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

def create_variables10(name, shape):

    n1 = np.sqrt(6. / (shape[0] * shape[1] * l10r1 * (shape[-2] + shape[-1])))
    n2 = np.sqrt(6. / (shape[0] * shape[1] * l10r2 * (shape[-2] + shape[-1])))
    n3 = np.sqrt(6. / (shape[0] * shape[1] * l10r3 * (shape[-2] + shape[-1])))
    n4 = np.sqrt(6. / (shape[0] * shape[1] * l10r4 * (shape[-2] + shape[-1])))
    n5 = np.sqrt(6. / (shape[0] * shape[1] * l10r5 * (shape[-2] + shape[-1])))
    n6 = np.sqrt(6. / (shape[0] * shape[1] * l10r6 * (shape[-2] + shape[-1])))
    n7 = np.sqrt(6. / (shape[0] * shape[1] * l10r7 * (shape[-2] + shape[-1])))
    n8 = np.sqrt(6. / (shape[0] * shape[1] * l10r8 * (shape[-2] + shape[-1])))
    n9 = np.sqrt(6. / (shape[0] * shape[1] * l10r9 * (shape[-2] + shape[-1])))
    n10 = np.sqrt(6. / (shape[0] * shape[1] * (shape[-2] + shape[-1])))
    shape1 = [shape[0], shape[1], int(l10r1 * shape[2]), int(l10r1 * shape[3])]
    shape2_1 = [shape[0], shape[1], int(l10r2 * shape[2]) - int(l10r1 * shape[2]), int(l10r1 * shape[3])]
    shape2_2 = [shape[0], shape[1], int(l10r2 * shape[2]), int(l10r2 * shape[3]) - int(l10r1 * shape[3])]
    shape3_1 = [shape[0], shape[1], int(l10r3 * shape[2]) - int(l10r2 * shape[2]), int(l10r2 * shape[3])]
    shape3_2 = [shape[0], shape[1], int(l10r3 * shape[2]), int(l10r3 * shape[3]) - int(l10r2 * shape[3])]
    shape4_1 = [shape[0], shape[1], int(l10r4 * shape[2]) - int(l10r3 * shape[2]), int(l10r3 * shape[3])]
    shape4_2 = [shape[0], shape[1], int(l10r4 * shape[2]), int(l10r4 * shape[3]) - int(l10r3 * shape[3])]
    shape5_1 = [shape[0], shape[1], int(l10r5 * shape[2]) - int(l10r4 * shape[2]), int(l10r4 * shape[3])]
    shape5_2 = [shape[0], shape[1], int(l10r5 * shape[2]), int(l10r5 * shape[3]) - int(l10r4 * shape[3])]
    shape6_1 = [shape[0], shape[1], int(l10r6 * shape[2]) - int(l10r5 * shape[2]), int(l10r5 * shape[3])]
    shape6_2 = [shape[0], shape[1], int(l10r6 * shape[2]), int(l10r6 * shape[3]) - int(l10r5 * shape[3])]
    shape7_1 = [shape[0], shape[1], int(l10r7 * shape[2]) - int(l10r6 * shape[2]), int(l10r6 * shape[3])]
    shape7_2 = [shape[0], shape[1], int(l10r7 * shape[2]), int(l10r7 * shape[3]) - int(l10r6 * shape[3])]
    shape8_1 = [shape[0], shape[1], int(l10r8 * shape[2]) - int(l10r7 * shape[2]), int(l10r7 * shape[3])]
    shape8_2 = [shape[0], shape[1], int(l10r8 * shape[2]), int(l10r8 * shape[3]) - int(l10r7 * shape[3])]
    shape9_1 = [shape[0], shape[1], int(l10r9 * shape[2]) - int(l10r8 * shape[2]), int(l10r8 * shape[3])]
    shape9_2 = [shape[0], shape[1], int(l10r9 * shape[2]), int(l10r9 * shape[3]) - int(l10r8 * shape[3])]
    shape10_1 = [shape[0], shape[1], shape[2] - int(l10r9 * shape[2]), int(l10r9 * shape[3])]
    shape10_2 = [shape[0], shape[1], shape[2], shape[3] - int(l10r9 * shape[3])]

    lv1_variables = tf.get_variable(name + '_l1', initializer=tf.random_uniform(shape1, -n10, n10, tf.float32, seed=None))
    lv2_1_variables = tf.get_variable(name + '_l2_1', initializer=tf.random_uniform(shape2_1, -n10, n10, tf.float32, seed=None))
    lv2_2_variables = tf.get_variable(name + '_l2_2', initializer=tf.random_uniform(shape2_2, -n10, n10, tf.float32, seed=None))
    lv3_1_variables = tf.get_variable(name + '_l3_1', initializer=tf.random_uniform(shape3_1, -n10, n10, tf.float32, seed=None))
    lv3_2_variables = tf.get_variable(name + '_l3_2', initializer=tf.random_uniform(shape3_2, -n10, n10, tf.float32, seed=None))
    lv4_1_variables = tf.get_variable(name + '_l4_1', initializer=tf.random_uniform(shape4_1, -n10, n10, tf.float32, seed=None))
    lv4_2_variables = tf.get_variable(name + '_l4_2', initializer=tf.random_uniform(shape4_2, -n10, n10, tf.float32, seed=None))
    lv5_1_variables = tf.get_variable(name + '_l5_1', initializer=tf.random_uniform(shape5_1, -n10, n10, tf.float32, seed=None))
    lv5_2_variables = tf.get_variable(name + '_l5_2', initializer=tf.random_uniform(shape5_2, -n10, n10, tf.float32, seed=None))
    lv6_1_variables = tf.get_variable(name + '_l6_1', initializer=tf.random_uniform(shape6_1, -n10, n10, tf.float32, seed=None))
    lv6_2_variables = tf.get_variable(name + '_l6_2', initializer=tf.random_uniform(shape6_2, -n10, n10, tf.float32, seed=None))
    lv7_1_variables = tf.get_variable(name + '_l7_1', initializer=tf.random_uniform(shape7_1, -n10, n10, tf.float32, seed=None))
    lv7_2_variables = tf.get_variable(name + '_l7_2', initializer=tf.random_uniform(shape7_2, -n10, n10, tf.float32, seed=None))
    lv8_1_variables = tf.get_variable(name + '_l8_1', initializer=tf.random_uniform(shape8_1, -n10, n10, tf.float32, seed=None))
    lv8_2_variables = tf.get_variable(name + '_l8_2', initializer=tf.random_uniform(shape8_2, -n10, n10, tf.float32, seed=None))
    lv9_1_variables = tf.get_variable(name + '_l9_1', initializer=tf.random_uniform(shape9_1, -n10, n10, tf.float32, seed=None))
    lv9_2_variables = tf.get_variable(name + '_l9_2', initializer=tf.random_uniform(shape9_2, -n10, n10, tf.float32, seed=None))
    lv10_1_variables = tf.get_variable(name + '_l10_1', initializer=tf.random_uniform(shape10_1, -n10, n10, tf.float32, seed=None))
    lv10_2_variables = tf.get_variable(name + '_l10_2', initializer=tf.random_uniform(shape10_2, -n10, n10, tf.float32, seed=None))

    return lv1_variables, lv2_1_variables, lv2_2_variables, \
        lv3_1_variables, lv3_2_variables, lv4_1_variables, lv4_2_variables, \
        lv5_1_variables, lv5_2_variables, lv6_1_variables, lv6_2_variables, \
        lv7_1_variables, lv7_2_variables, lv8_1_variables, lv8_2_variables, \
        lv9_1_variables, lv9_2_variables, lv10_1_variables, lv10_2_variables

def create_variablesn(name, shape):
    ni = np.sqrt(6. / (shape[0] * shape[1] * (shape[-2] + shape[-1])))

    shape1 = []
    shape2 = []
    shape1.append([shape[0], shape[1], int(lnr[0] * shape[2]), int(lnr[0] * shape[3])])
	
    for i in range(1, n):
        shape1.append([shape[0], shape[1], int(lnr[i] * shape[2]) - int(lnr[i-1] * shape[2]), int(lnr[i-1] * shape[3])] )
        shape2.append([shape[0], shape[1], int(lnr[i] * shape[2]), int(lnr[i] * shape[3]) - int(lnr[i-1] * shape[3])] )

    lv1_variables = []
    lv2_variables = []
    lv1_variables.append(tf.get_variable(name + '_l_0', initializer=tf.random_uniform(shape1[0], -ni, ni, tf.float32, seed=None)))
    for i in range(1, n):
        lv1_variables.append(tf.get_variable(name + '_l1_' + str(i), initializer=tf.random_uniform(shape1[i], -ni, ni, tf.float32, seed=None)))
        lv2_variables.append(tf.get_variable(name + '_l2_' + str(i), initializer=tf.random_uniform(shape2[i-1], -ni, ni, tf.float32, seed=None)))

    return lv1_variables, lv2_variables

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

def output_layer10(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, num_labels):

    input_dim1 = input1.get_shape().as_list()[-1]
    input_dim2 = input2.get_shape().as_list()[-1]
    input_dim3 = input3.get_shape().as_list()[-1]
    input_dim4 = input4.get_shape().as_list()[-1]
    input_dim5 = input5.get_shape().as_list()[-1]
    input_dim6 = input6.get_shape().as_list()[-1]
    input_dim7 = input7.get_shape().as_list()[-1]
    input_dim8 = input8.get_shape().as_list()[-1]
    input_dim9 = input9.get_shape().as_list()[-1]
    input_dim10 = input10.get_shape().as_list()[-1]

    fc_w1 = tf.get_variable('fc_weights_l1', shape=[input_dim1, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w2 = tf.get_variable('fc_weights_l2', shape=[input_dim2, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w3 = tf.get_variable('fc_weights_l3', shape=[input_dim3, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w4 = tf.get_variable('fc_weights_l4', shape=[input_dim4, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w5 = tf.get_variable('fc_weights_l5', shape=[input_dim5, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w6 = tf.get_variable('fc_weights_l6', shape=[input_dim6, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w7 = tf.get_variable('fc_weights_l7', shape=[input_dim7, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w8 = tf.get_variable('fc_weights_l8', shape=[input_dim8, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w9 = tf.get_variable('fc_weights_l9', shape=[input_dim9, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w10 = tf.get_variable('fc_weights_l10', shape=[input_dim10, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))

    fc_b1 = tf.get_variable(name='fc_bias_l1', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b2 = tf.get_variable(name='fc_bias_l2', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b3 = tf.get_variable(name='fc_bias_l3', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b4 = tf.get_variable(name='fc_bias_l4', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b5 = tf.get_variable(name='fc_bias_l5', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b6 = tf.get_variable(name='fc_bias_l6', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b7 = tf.get_variable(name='fc_bias_l7', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b8 = tf.get_variable(name='fc_bias_l8', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b9 = tf.get_variable(name='fc_bias_l9', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b10 = tf.get_variable(name='fc_bias_l10', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h1 = tf.matmul(input1, fc_w1) + fc_b1
    fc_h2 = tf.matmul(input2, fc_w2) + fc_b2
    fc_h3 = tf.matmul(input3, fc_w3) + fc_b3
    fc_h4 = tf.matmul(input4, fc_w4) + fc_b4
    fc_h5 = tf.matmul(input5, fc_w5) + fc_b5
    fc_h6 = tf.matmul(input6, fc_w6) + fc_b6
    fc_h7 = tf.matmul(input7, fc_w7) + fc_b7
    fc_h8 = tf.matmul(input8, fc_w8) + fc_b8
    fc_h9 = tf.matmul(input9, fc_w9) + fc_b9
    fc_h10 = tf.matmul(input10, fc_w10) + fc_b10

    return fc_h1, fc_h2, fc_h3, fc_h4, fc_h5, fc_h6, fc_h7, fc_h8, fc_h9, fc_h10

def output_layern(inputs, num_labels):
    input_dim = []
    for i in range(0, n):
        input_dim.append(inputs[i].get_shape().as_list()[-1])

    fc_w = []
    for i in range(0, n):
        fc_w.append(tf.get_variable('fc_weights_l'+ str(i+1), shape=[input_dim[i], num_labels], initializer=tf.initializers.variance_scaling(scale=1.0)))

    fc_b = []
    for i in range(0, n):
        fc_b.append(tf.get_variable(name='fc_bias_l'+ str(i+1), shape=[num_labels], initializer=tf.zeros_initializer()))

    fc_h = []
    for i in range(0, n):
        fc_h.append(tf.matmul(inputs[i], fc_w[i]) + fc_b[i])

    return fc_h

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

def conv_layer10(input_layer, filter_shape, stride, is_training):

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer('l1_l2_l3', input_layer, in_channel, is_training)
    bn_layer = tf.nn.relu(bn_layer)

    n1 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r1 * filter_shape[-1])))
    n2 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r2 * filter_shape[-1])))
    n3 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r3 * filter_shape[-1])))
    n4 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r4 * filter_shape[-1])))
    n5 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r5 * filter_shape[-1])))
    n6 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r6 * filter_shape[-1])))
    n7 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r7 * filter_shape[-1])))
    n8 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r8 * filter_shape[-1])))
    n9 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l10r9 * filter_shape[-1])))
    n10 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + filter_shape[-1])))

    filter1 = tf.get_variable('conv_l1',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r1*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter2 = tf.get_variable('conv_l2',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r2*filter_shape[3])-int(l10r1*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter3 = tf.get_variable('conv_l3',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r3*filter_shape[3])-int(l10r2*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter4 = tf.get_variable('conv_l4',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r4*filter_shape[3])-int(l10r3*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter5 = tf.get_variable('conv_l5',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r5*filter_shape[3])-int(l10r4*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter6 = tf.get_variable('conv_l6',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r6*filter_shape[3])-int(l10r5*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter7 = tf.get_variable('conv_l7',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r7*filter_shape[3])-int(l10r6*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter8 = tf.get_variable('conv_l8',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r8*filter_shape[3])-int(l10r7*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter9 = tf.get_variable('conv_l9',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l10r9*filter_shape[3])-int(l10r8*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))
    filter10 = tf.get_variable('conv_l10',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]-int(l10r9*filter_shape[3])],
                                                            -n10, n10,   tf.float32, seed=None))

    conv1 = tf.nn.conv2d(bn_layer, filter1, strides=[1, stride, stride, 1], padding='SAME')
    conv2 = tf.concat((conv1, tf.nn.conv2d(bn_layer, filter2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((conv2, tf.nn.conv2d(bn_layer, filter3, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv4 = tf.concat((conv3, tf.nn.conv2d(bn_layer, filter4, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv5 = tf.concat((conv4, tf.nn.conv2d(bn_layer, filter5, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv6 = tf.concat((conv5, tf.nn.conv2d(bn_layer, filter6, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv7 = tf.concat((conv6, tf.nn.conv2d(bn_layer, filter7, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv8 = tf.concat((conv7, tf.nn.conv2d(bn_layer, filter8, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((conv8, tf.nn.conv2d(bn_layer, filter9, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((conv9, tf.nn.conv2d(bn_layer, filter10, strides=[1, stride, stride, 1], padding='SAME')), 3)

    return conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10

def conv_layern(input_layer, filter_shape, stride, is_training):

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer('l1_l2_l3', input_layer, in_channel, is_training)
    bn_layer = tf.nn.relu(bn_layer)

    ni = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + filter_shape[-1])))

    filters = []
    filters.append(tf.get_variable('conv_l1',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(lnr[0]*filter_shape[3])],
                                                            -ni, ni,   tf.float32, seed=None)))
    for i in range (1, n):
        filters.append(tf.get_variable('conv_l'+str(i+1),
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(lnr[i]*filter_shape[3])-int(lnr[i-1]*filter_shape[3])],
                                                            -ni, ni,   tf.float32, seed=None)))

    conv = []
    conv.append(tf.nn.conv2d(bn_layer, filters[0], strides=[1, stride, stride, 1], padding='SAME'))
    for i in range (1, n):
        conv.append(tf.concat((conv[i-1], tf.nn.conv2d(bn_layer, filters[i], strides=[1, stride, stride, 1], padding='SAME')), 3))

    return conv

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

def bn_relu_conv_layer10(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, filter_shape, stride, is_training):

    in_channel1 = input1.get_shape().as_list()[-1]
    in_channel2 = input2.get_shape().as_list()[-1]
    in_channel3 = input3.get_shape().as_list()[-1]
    in_channel4 = input4.get_shape().as_list()[-1]
    in_channel5 = input5.get_shape().as_list()[-1]
    in_channel6 = input6.get_shape().as_list()[-1]
    in_channel7 = input7.get_shape().as_list()[-1]
    in_channel8 = input8.get_shape().as_list()[-1]
    in_channel9 = input9.get_shape().as_list()[-1]
    in_channel10 = input10.get_shape().as_list()[-1]

    bn_layer1 = batch_normalization_layer('l1', input1, in_channel1, is_training)
    bn_layer1 = tf.nn.relu(bn_layer1)
    bn_layer2 = batch_normalization_layer('l2', input2, in_channel2, is_training)
    bn_layer2 = tf.nn.relu(bn_layer2)
    bn_layer3 = batch_normalization_layer('l3', input3, in_channel3, is_training)
    bn_layer3 = tf.nn.relu(bn_layer3)
    bn_layer4 = batch_normalization_layer('l4', input4, in_channel4, is_training)
    bn_layer4 = tf.nn.relu(bn_layer4)
    bn_layer5 = batch_normalization_layer('l5', input5, in_channel5, is_training)
    bn_layer5 = tf.nn.relu(bn_layer5)
    bn_layer6 = batch_normalization_layer('l6', input6, in_channel6, is_training)
    bn_layer6 = tf.nn.relu(bn_layer6)
    bn_layer7 = batch_normalization_layer('l7', input7, in_channel7, is_training)
    bn_layer7 = tf.nn.relu(bn_layer7)
    bn_layer8 = batch_normalization_layer('l8', input8, in_channel8, is_training)
    bn_layer8 = tf.nn.relu(bn_layer8)
    bn_layer9 = batch_normalization_layer('l9', input9, in_channel9, is_training)
    bn_layer9 = tf.nn.relu(bn_layer9)
    bn_layer10 = batch_normalization_layer('l10', input10, in_channel10, is_training)
    bn_layer10 = tf.nn.relu(bn_layer10)

    filter1, filter2_1, filter2_2, filter3_1, filter3_2 , filter4_1, filter4_2, filter5_1, filter5_2, \
        filter6_1, filter6_2, filter7_1, filter7_2, filter8_1, filter8_2, filter9_1, filter9_2, filter10_1, filter10_2 = create_variables10(name='conv', shape=filter_shape)
    conv1 = tf.nn.conv2d(bn_layer1, filter1, strides=[1, stride, stride, 1], padding='SAME')

    conv2 = tf.concat((tf.add(tf.nn.conv2d(bn_layer2[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer2[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer2, filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv3 = tf.concat((tf.add(tf.nn.conv2d(bn_layer3[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer3[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer3[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((tf.add(conv3, 
					          tf.nn.conv2d(bn_layer3[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer3, filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv4 = tf.concat((tf.add(tf.nn.conv2d(bn_layer4[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer4[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer4[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv4 = tf.concat((tf.add(conv4, 
					          tf.nn.conv2d(bn_layer4[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer4[:, :, :, :int(l10r3 * filter_shape[2])], filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv4 = tf.concat((tf.add(conv4, 
					          tf.nn.conv2d(bn_layer4[:, :, :, int(l10r3 * filter_shape[2]):int(l10r4 * filter_shape[2])], filter4_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer4, filter4_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv5 = tf.concat((tf.add(tf.nn.conv2d(bn_layer5[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer5[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer5[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv5 = tf.concat((tf.add(conv5, 
					          tf.nn.conv2d(bn_layer5[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer5[:, :, :, :int(l10r3 * filter_shape[2])], filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv5 = tf.concat((tf.add(conv5, 
					          tf.nn.conv2d(bn_layer5[:, :, :, int(l10r3 * filter_shape[2]):int(l10r4 * filter_shape[2])], filter4_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer5[:, :, :, :int(l10r4 * filter_shape[2])], filter4_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv5 = tf.concat((tf.add(conv5, 
					          tf.nn.conv2d(bn_layer5[:, :, :, int(l10r4 * filter_shape[2]):int(l10r5 * filter_shape[2])], filter5_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer5, filter5_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv6 = tf.concat((tf.add(tf.nn.conv2d(bn_layer6[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer6[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer6[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv6 = tf.concat((tf.add(conv6, 
					          tf.nn.conv2d(bn_layer6[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer6[:, :, :, :int(l10r3 * filter_shape[2])], filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv6 = tf.concat((tf.add(conv6, 
					          tf.nn.conv2d(bn_layer6[:, :, :, int(l10r3 * filter_shape[2]):int(l10r4 * filter_shape[2])], filter4_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer6[:, :, :, :int(l10r4 * filter_shape[2])], filter4_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv6 = tf.concat((tf.add(conv6, 
					          tf.nn.conv2d(bn_layer6[:, :, :, int(l10r4 * filter_shape[2]):int(l10r5 * filter_shape[2])], filter5_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer6[:, :, :, :int(l10r5 * filter_shape[2])], filter5_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv6 = tf.concat((tf.add(conv6, 
					          tf.nn.conv2d(bn_layer6[:, :, :, int(l10r5 * filter_shape[2]):int(l10r6 * filter_shape[2])], filter6_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer6, filter6_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv7 = tf.concat((tf.add(tf.nn.conv2d(bn_layer7[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer7[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer7[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv7 = tf.concat((tf.add(conv7, 
					          tf.nn.conv2d(bn_layer7[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer7[:, :, :, :int(l10r3 * filter_shape[2])], filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv7 = tf.concat((tf.add(conv7, 
					          tf.nn.conv2d(bn_layer7[:, :, :, int(l10r3 * filter_shape[2]):int(l10r4 * filter_shape[2])], filter4_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer7[:, :, :, :int(l10r4 * filter_shape[2])], filter4_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv7 = tf.concat((tf.add(conv7, 
					          tf.nn.conv2d(bn_layer7[:, :, :, int(l10r4 * filter_shape[2]):int(l10r5 * filter_shape[2])], filter5_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer7[:, :, :, :int(l10r5 * filter_shape[2])], filter5_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv7 = tf.concat((tf.add(conv7, 
					          tf.nn.conv2d(bn_layer7[:, :, :, int(l10r5 * filter_shape[2]):int(l10r6 * filter_shape[2])], filter6_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer7[:, :, :, :int(l10r6 * filter_shape[2])], filter6_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv7 = tf.concat((tf.add(conv7, 
					          tf.nn.conv2d(bn_layer7[:, :, :, int(l10r6 * filter_shape[2]):int(l10r7 * filter_shape[2])], filter7_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer7, filter7_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv8 = tf.concat((tf.add(tf.nn.conv2d(bn_layer8[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer8[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer8[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv8 = tf.concat((tf.add(conv8, 
					          tf.nn.conv2d(bn_layer8[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer8[:, :, :, :int(l10r3 * filter_shape[2])], filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv8 = tf.concat((tf.add(conv8, 
					          tf.nn.conv2d(bn_layer8[:, :, :, int(l10r3 * filter_shape[2]):int(l10r4 * filter_shape[2])], filter4_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer8[:, :, :, :int(l10r4 * filter_shape[2])], filter4_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv8 = tf.concat((tf.add(conv8, 
					          tf.nn.conv2d(bn_layer8[:, :, :, int(l10r4 * filter_shape[2]):int(l10r5 * filter_shape[2])], filter5_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer8[:, :, :, :int(l10r5 * filter_shape[2])], filter5_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv8 = tf.concat((tf.add(conv8, 
					          tf.nn.conv2d(bn_layer8[:, :, :, int(l10r5 * filter_shape[2]):int(l10r6 * filter_shape[2])], filter6_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer8[:, :, :, :int(l10r6 * filter_shape[2])], filter6_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv8 = tf.concat((tf.add(conv8, 
					          tf.nn.conv2d(bn_layer8[:, :, :, int(l10r6 * filter_shape[2]):int(l10r7 * filter_shape[2])], filter7_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer8[:, :, :, :int(l10r7 * filter_shape[2])], filter7_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv8 = tf.concat((tf.add(conv8, 
					          tf.nn.conv2d(bn_layer8[:, :, :, int(l10r7 * filter_shape[2]):int(l10r8 * filter_shape[2])], filter8_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer8, filter8_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv9 = tf.concat((tf.add(tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer9[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((tf.add(conv9, 
					          tf.nn.conv2d(bn_layer9[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r3 * filter_shape[2])], filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((tf.add(conv9, 
					          tf.nn.conv2d(bn_layer9[:, :, :, int(l10r3 * filter_shape[2]):int(l10r4 * filter_shape[2])], filter4_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r4 * filter_shape[2])], filter4_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((tf.add(conv9, 
					          tf.nn.conv2d(bn_layer9[:, :, :, int(l10r4 * filter_shape[2]):int(l10r5 * filter_shape[2])], filter5_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r5 * filter_shape[2])], filter5_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((tf.add(conv9, 
					          tf.nn.conv2d(bn_layer9[:, :, :, int(l10r5 * filter_shape[2]):int(l10r6 * filter_shape[2])], filter6_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r6 * filter_shape[2])], filter6_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((tf.add(conv9, 
					          tf.nn.conv2d(bn_layer9[:, :, :, int(l10r6 * filter_shape[2]):int(l10r7 * filter_shape[2])], filter7_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r7 * filter_shape[2])], filter7_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((tf.add(conv9, 
					          tf.nn.conv2d(bn_layer9[:, :, :, int(l10r7 * filter_shape[2]):int(l10r8 * filter_shape[2])], filter8_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9[:, :, :, :int(l10r8 * filter_shape[2])], filter8_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv9 = tf.concat((tf.add(conv9, 
					          tf.nn.conv2d(bn_layer9[:, :, :, int(l10r8 * filter_shape[2]):int(l10r9 * filter_shape[2])], filter9_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer9, filter9_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    conv10 = tf.concat((tf.add(tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r1 * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer10[:, :, :, int(l10r1 * filter_shape[2]):int(l10r2 * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r2 * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r2 * filter_shape[2]):int(l10r3 * filter_shape[2])], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r3 * filter_shape[2])], filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r3 * filter_shape[2]):int(l10r4 * filter_shape[2])], filter4_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r4 * filter_shape[2])], filter4_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r4 * filter_shape[2]):int(l10r5 * filter_shape[2])], filter5_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r5 * filter_shape[2])], filter5_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r5 * filter_shape[2]):int(l10r6 * filter_shape[2])], filter6_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r6 * filter_shape[2])], filter6_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r6 * filter_shape[2]):int(l10r7 * filter_shape[2])], filter7_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r7 * filter_shape[2])], filter7_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r7 * filter_shape[2]):int(l10r8 * filter_shape[2])], filter8_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r8 * filter_shape[2])], filter8_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r8 * filter_shape[2]):int(l10r9 * filter_shape[2])], filter9_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10[:, :, :, :int(l10r9 * filter_shape[2])], filter9_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv10 = tf.concat((tf.add(conv10, 
					          tf.nn.conv2d(bn_layer10[:, :, :, int(l10r9 * filter_shape[2]):], filter10_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer10, filter10_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    return conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10

def bn_relu_conv_layern(inputs, filter_shape, stride, is_training):

    in_channel = []
    for i in range(0, n):
        in_channel.append(inputs[i].get_shape().as_list()[-1])

    bn_layer = []
    for i in range(0, n):
        temp_bn_layer = batch_normalization_layer('l'+str(i+1), inputs[i], in_channel[i], is_training)
        bn_layer.append(tf.nn.relu(temp_bn_layer))

    filter1, filter2 = create_variablesn(name='conv', shape=filter_shape)

    conv = []
    conv.append(tf.nn.conv2d(bn_layer[0], filter1[0], strides=[1, stride, stride, 1], padding='SAME'))

    for i in range(1, n):
        temp_conv = tf.concat((tf.add(tf.nn.conv2d(bn_layer[i][:, :, :, :int(lnr[0] * filter_shape[2])], filter1[0], strides=[1, stride, stride, 1], padding='SAME'),
                                      tf.nn.conv2d(bn_layer[i][:, :, :, int(lnr[0] * filter_shape[2]):int(lnr[1] * filter_shape[2])], filter1[1], strides=[1, stride, stride, 1], padding='SAME')),
                               tf.nn.conv2d(bn_layer[i][:, :, :, :int(lnr[1] * filter_shape[2])], filter2[0], strides=[1, stride, stride, 1], padding='SAME')), 3)
        for j in range(1, i):
            temp_conv = tf.concat((tf.add(temp_conv, 
					                      tf.nn.conv2d(bn_layer[i][:, :, :, int(lnr[j] * filter_shape[2]):int(lnr[j+1] * filter_shape[2])], filter1[j+1], strides=[1, stride, stride, 1], padding='SAME')),
                                  tf.nn.conv2d(bn_layer[i][:, :, :, :int(lnr[j+1] * filter_shape[2])], filter2[j], strides=[1, stride, stride, 1], padding='SAME')), 3)
        conv.append(temp_conv)

    return conv

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

def residual_block10(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output_channel, wide_scale, is_training, first_block=False):

    input_channel = input10.get_shape().as_list()[-1]

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
            conv1, conv2, conv3, conv4, conv5, \
                conv6, conv7, conv8, conv9, conv10 = conv_layer10(input1, [3, 3, input_channel, output_channel], stride, is_training)
        else:
            conv1, conv2, conv3, conv4, conv5, \
                conv6, conv7, conv8, conv9, conv10 = bn_relu_conv_layer10(input1, input2, input3, input4, input5, \
						                                                  input6, input7, input8, input9, input10, [3, 3, input_channel, output_channel], stride, is_training)

    with tf.variable_scope('conv2_in_block'):
        conv1, conv2, conv3, conv4, conv5, \
            conv6, conv7, conv8, conv9, conv100 = bn_relu_conv_layer10(conv1, conv2, conv3, conv4, conv5, \
                                                                       conv6, conv7, conv8, conv9, conv10, [3, 3, output_channel, output_channel], 1, is_training)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        if input_channel * wide_scale == output_channel:
            if first_block:
                np0 = int((output_channel * l10r1 - input_channel) / 2)
                np1 = int((output_channel * l10r2 - input_channel) / 2)
                np2 = int((output_channel * l10r3 - input_channel) / 2)
                np3 = int((output_channel * l10r4 - input_channel) / 2)
                np4 = int((output_channel * l10r5 - input_channel) / 2)
                np5 = int((output_channel * l10r6 - input_channel) / 2)
                np6 = int((output_channel * l10r7 - input_channel) / 2)
                np7 = int((output_channel * l10r8 - input_channel) / 2)
                np8 = int((output_channel * l10r9 - input_channel) / 2)
                np9 = int((output_channel * 1 - input_channel) / 2)
                padded_input1 = tf.pad(input1, [[0, 0], [0, 0], [0, 0], [np0, np0]])
                padded_input2 = tf.pad(input2, [[0, 0], [0, 0], [0, 0], [np1, np1]])
                padded_input3 = tf.pad(input3, [[0, 0], [0, 0], [0, 0], [np2, np2]])
                padded_input4 = tf.pad(input4, [[0, 0], [0, 0], [0, 0], [np3, np3]])
                padded_input5 = tf.pad(input5, [[0, 0], [0, 0], [0, 0], [np4, np4]])
                padded_input6 = tf.pad(input6, [[0, 0], [0, 0], [0, 0], [np5, np5]])
                padded_input7 = tf.pad(input7, [[0, 0], [0, 0], [0, 0], [np6, np6]])
                padded_input8 = tf.pad(input8, [[0, 0], [0, 0], [0, 0], [np7, np7]])
                padded_input9 = tf.pad(input9, [[0, 0], [0, 0], [0, 0], [np8, np8]])
                padded_input10 = tf.pad(input10, [[0, 0], [0, 0], [0, 0], [np9, np9]])
            else:
                np1 = int((output_channel - input_channel) / 2 * l10r1)
                np2 = int((output_channel - input_channel) / 2 * l10r2)
                np3 = int((output_channel - input_channel) / 2 * l10r3)
                np4 = int((output_channel - input_channel) / 2 * l10r4)
                np5 = int((output_channel - input_channel) / 2 * l10r5)
                np6 = int((output_channel - input_channel) / 2 * l10r6)
                np7 = int((output_channel - input_channel) / 2 * l10r7)
                np8 = int((output_channel - input_channel) / 2 * l10r8)
                np9 = int((output_channel - input_channel) / 2 * l10r9)
                np10 = int((output_channel - input_channel) / 2)
                padded_input1 = tf.pad(input1, [[0, 0], [0, 0], [0, 0], [np1, np1]])
                padded_input2 = tf.pad(input2, [[0, 0], [0, 0], [0, 0], [np2, np2]])
                padded_input3 = tf.pad(input3, [[0, 0], [0, 0], [0, 0], [np3, np3]])
                padded_input4 = tf.pad(input4, [[0, 0], [0, 0], [0, 0], [np4, np4]])
                padded_input5 = tf.pad(input5, [[0, 0], [0, 0], [0, 0], [np5, np5]])
                padded_input6 = tf.pad(input6, [[0, 0], [0, 0], [0, 0], [np6, np6]])
                padded_input7 = tf.pad(input7, [[0, 0], [0, 0], [0, 0], [np7, np7]])
                padded_input8 = tf.pad(input8, [[0, 0], [0, 0], [0, 0], [np8, np8]])
                padded_input9 = tf.pad(input9, [[0, 0], [0, 0], [0, 0], [np9, np9]])
                padded_input10 = tf.pad(input10, [[0, 0], [0, 0], [0, 0], [np10, np10]])
        else:
            pooled_input1 = tf.nn.avg_pool(input1, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input1 = tf.pad(pooled_input1, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r1) // 2,
                                                                            int(input_channel*l10r1) // 2]])
            pooled_input2 = tf.nn.avg_pool(input2, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input2 = tf.pad(pooled_input2, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r2) // 2,
                                                                            int(input_channel*l10r2) // 2]])
            pooled_input3 = tf.nn.avg_pool(input3, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input3 = tf.pad(pooled_input3, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r3) // 2,
                                                                            int(input_channel*l10r3) // 2]])
            pooled_input4 = tf.nn.avg_pool(input4, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input4 = tf.pad(pooled_input4, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r4) // 2,
                                                                            int(input_channel*l10r4) // 2]])
            pooled_input5 = tf.nn.avg_pool(input5, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input5 = tf.pad(pooled_input5, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r5) // 2,
                                                                            int(input_channel*l10r5) // 2]])
            pooled_input6 = tf.nn.avg_pool(input6, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input6 = tf.pad(pooled_input6, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r6) // 2,
                                                                            int(input_channel*l10r6) // 2]])
            pooled_input7 = tf.nn.avg_pool(input7, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input7 = tf.pad(pooled_input7, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r7) // 2,
                                                                            int(input_channel*l10r7) // 2]])
            pooled_input8 = tf.nn.avg_pool(input8, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input8 = tf.pad(pooled_input8, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r8) // 2,
                                                                            int(input_channel*l10r8) // 2]])
            pooled_input9 = tf.nn.avg_pool(input9, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input9 = tf.pad(pooled_input9, [[0, 0], [0, 0], [0, 0], [int(input_channel*l10r9) // 2,
                                                                            int(input_channel*l10r9) // 2]])
            pooled_input10 = tf.nn.avg_pool(input10, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input10 = tf.pad(pooled_input10, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                            input_channel // 2]])
    else:
        padded_input1 = input1
        padded_input2 = input2
        padded_input3 = input3
        padded_input4 = input4
        padded_input5 = input5
        padded_input6 = input6
        padded_input7 = input7
        padded_input8 = input8
        padded_input9 = input9
        padded_input10 = input10

    output1 = conv1 + padded_input1
    output2 = conv2 + padded_input2
    output3 = conv3 + padded_input3
    output4 = conv4 + padded_input4
    output5 = conv5 + padded_input5
    output6 = conv6 + padded_input6
    output7 = conv7 + padded_input7
    output8 = conv8 + padded_input8
    output9 = conv9 + padded_input9
    output10 = conv10 + padded_input10

    return output1, output2, output3, output4, output5, output6, output7, output8, output9, output10

def residual_blockn(inputs, output_channel, wide_scale, is_training, first_block=False):

    input_channel = inputs[n-1].get_shape().as_list()[-1]

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
            conv = conv_layern(inputs[0], [3, 3, input_channel, output_channel], stride, is_training)
        else:
            conv = bn_relu_conv_layern(inputs, [3, 3, input_channel, output_channel], stride, is_training)

    with tf.variable_scope('conv2_in_block'):
        conv = bn_relu_conv_layern(conv, [3, 3, output_channel, output_channel], 1, is_training)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        if input_channel * wide_scale == output_channel:
            if first_block:
                np = []
                for i in range(0, n):
                    np.append(int((output_channel * lnr[i] - input_channel) / 2))

                padded_input = []
                for i in range(0, n):
                    padded_input.append(tf.pad(inputs[i], [[0, 0], [0, 0], [0, 0], [np[i], np[i]]]))
            else:
                np = []
                for i in range(0, n):
                    np.append(int((output_channel - input_channel) / 2) * lnr[i])
                padded_input = []
                for i in range(0, n):
                    padded_input.append(tf.pad(inputs[i], [[0, 0], [0, 0], [0, 0], [np[i], np[i]]]))
        else:
            pooled_input = []
            padded_input = []
            for i in range(0, n):
                pooled_input.append(tf.nn.avg_pool(inputs[i], ksize=[1, 2, 2, 1],
							                       strides=[1, 2, 2, 1], padding='VALID'))
                padded_input.append(tf.pad(pooled_input[i], [[0, 0], [0, 0], [0, 0], [int(input_channel*lnr[i]) // 2,
                                                                                      int(input_channel*lnr[i]) // 2]]))
    else:
        padded_input = []
        for i in range(0, n):
            padded_input.append(inputs[i])

    output = []
    for i in range(0, n):
        output.append(conv[i] + padded_input[i])

    return output

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

def nested_relu10(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, size):
	size1 = int(size*l10r1)
	size2 = int(size*l10r2) - int(size*l10r1)
	size3 = int(size*l10r3) - int(size*l10r2)
	size4 = int(size*l10r4) - int(size*l10r3)
	size5 = int(size*l10r5) - int(size*l10r4)
	size6 = int(size*l10r6) - int(size*l10r5)
	size7 = int(size*l10r7) - int(size*l10r6)
	size8 = int(size*l10r8) - int(size*l10r7)
	size9 = int(size*l10r9) - int(size*l10r8)
	size10 = size - int(size*l10r9)


	bias1 = bias_variable(shape=(size1,))
	bias2 = tf.concat((bias1, bias_variable(shape=(size2,))),0)
	bias3 = tf.concat((bias2, bias_variable(shape=(size3,))),0)
	bias4 = tf.concat((bias3, bias_variable(shape=(size4,))),0)
	bias5 = tf.concat((bias4, bias_variable(shape=(size5,))),0)
	bias6 = tf.concat((bias5, bias_variable(shape=(size6,))),0)
	bias7 = tf.concat((bias6, bias_variable(shape=(size7,))),0)
	bias8 = tf.concat((bias7, bias_variable(shape=(size8,))),0)
	bias9 = tf.concat((bias8, bias_variable(shape=(size9,))),0)
	bias10 = tf.concat((bias9, bias_variable(shape=(size10,))),0)

	relu1 = tf.nn.relu(input1 + bias1)
	relu2 = tf.nn.relu(input2 + bias2)
	relu3 = tf.nn.relu(input3 + bias3)
	relu4 = tf.nn.relu(input4 + bias4)
	relu5 = tf.nn.relu(input5 + bias5)
	relu6 = tf.nn.relu(input6 + bias6)
	relu7 = tf.nn.relu(input7 + bias7)
	relu8 = tf.nn.relu(input8 + bias8)
	relu9 = tf.nn.relu(input9 + bias9)
	relu10 = tf.nn.relu(input10 + bias10)
	return relu1, relu2, relu3, relu4, relu5, relu6, relu7, relu8, relu9, relu10

def nested_relun(inputs, size):
    sizes = []
    sizes.append(int(size*lnr[0]))
    for i in range(1, n):
        sizes.append(int(size*lnr[i]) - int(size*lnr[i-1]))

    bias = []
    bias.append(bias_variable(shape=(sizes[0],)))
    for i in range(1,n):
        bias.append(tf.concat((bias[i-1], bias_variable(shape=(sizes[i],))),0))

    relu = []
    for i in range(0, n):
        relu.append(tf.nn.relu(inputs[i] + bias[i]))
    return relu

def nested_pool(input1, input2, input3):
	pool1 = tf.nn.max_pool(input1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool2 = tf.nn.max_pool(input2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool3 = tf.nn.max_pool(input3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	return pool1, pool2, pool3

def nested_pool10(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10):
	pool1 = tf.nn.max_pool(input1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool2 = tf.nn.max_pool(input2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool3 = tf.nn.max_pool(input3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool4 = tf.nn.max_pool(input4, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool5 = tf.nn.max_pool(input5, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool6 = tf.nn.max_pool(input6, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool7 = tf.nn.max_pool(input7, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool8 = tf.nn.max_pool(input8, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool9 = tf.nn.max_pool(input9, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	pool10 = tf.nn.max_pool(input10, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	return pool1, pool2, pool3, pool4, pool5, pool6, pool7, pool8, pool9, pool10

def nested_pooln(inputs):
    pool = []
    for i in range(0, n):
        pool.append(tf.nn.max_pool(inputs[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME'))
    return pool

def nested_drop(input1, input2, input3, keep_prob):
	drop1 = tf.nn.dropout(input1, keep_prob=keep_prob)
	drop2 = tf.nn.dropout(input2, keep_prob=keep_prob)
	drop3 = tf.nn.dropout(input3, keep_prob=keep_prob)
	return drop1, drop2, drop3

def nested_drop10(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, keep_prob):
	drop1 = tf.nn.dropout(input1, keep_prob=keep_prob)
	drop2 = tf.nn.dropout(input2, keep_prob=keep_prob)
	drop3 = tf.nn.dropout(input3, keep_prob=keep_prob)
	drop4 = tf.nn.dropout(input4, keep_prob=keep_prob)
	drop5 = tf.nn.dropout(input5, keep_prob=keep_prob)
	drop6 = tf.nn.dropout(input6, keep_prob=keep_prob)
	drop7 = tf.nn.dropout(input7, keep_prob=keep_prob)
	drop8 = tf.nn.dropout(input8, keep_prob=keep_prob)
	drop9 = tf.nn.dropout(input9, keep_prob=keep_prob)
	drop10 = tf.nn.dropout(input10, keep_prob=keep_prob)
	return drop1, drop2, drop3, drop4, drop5, drop6, drop7, drop8, drop9, drop10

def nested_dropn(inputs, keep_prob):
    drop = []
    for i in range(0, n):
        drop.append(tf.nn.dropout(inputs[i], keep_prob=keep_prob))
    return drop

def nested_fc_concat(drop1_1, drop1_2, drop1_3, drop2_1, drop2_2, drop2_3):
    fc0_1 = tf.concat([flatten(drop1_1), flatten(drop2_1)], 1)
    fc0_2 = tf.concat([flatten(drop1_2), flatten(drop2_2)], 1)
    fc0_3 = tf.concat([flatten(drop1_3), flatten(drop2_3)], 1)
    return fc0_1, fc0_2, fc0_3

def nested_fc_concat10(drop1_1, drop1_2, drop1_3, drop1_4, drop1_5, drop1_6, drop1_7, drop1_8, drop1_9, drop1_10, drop2_1, drop2_2, drop2_3, drop2_4, drop2_5, drop2_6, drop2_7, drop2_8, drop2_9, drop2_10):
    fc0_1 = tf.concat([flatten(drop1_1), flatten(drop2_1)], 1)
    fc0_2 = tf.concat([flatten(drop1_2), flatten(drop2_2)], 1)
    fc0_3 = tf.concat([flatten(drop1_3), flatten(drop2_3)], 1)
    fc0_4 = tf.concat([flatten(drop1_4), flatten(drop2_4)], 1)
    fc0_5 = tf.concat([flatten(drop1_5), flatten(drop2_5)], 1)
    fc0_6 = tf.concat([flatten(drop1_6), flatten(drop2_6)], 1)
    fc0_7 = tf.concat([flatten(drop1_7), flatten(drop2_7)], 1)
    fc0_8 = tf.concat([flatten(drop1_8), flatten(drop2_8)], 1)
    fc0_9 = tf.concat([flatten(drop1_9), flatten(drop2_9)], 1)
    fc0_10 = tf.concat([flatten(drop1_10), flatten(drop2_10)], 1)
    return fc0_1, fc0_2, fc0_3, fc0_4, fc0_5, fc0_6, fc0_7, fc0_8, fc0_9, fc0_10
    
def nested_fc_concatn(drop1, drop2):
    fc0 = []
    for i in range(0, n):
        fc0.append(tf.concat([flatten(drop1[i]), flatten(drop2[i])], 1))
    return fc0

def inference(x, n_classes, keep_prob, is_training):
    
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

	fc0_1, fc0_2, fc0_3 = nested_fc_concat(drop1_1, drop1_2, drop1_3, drop2_1, drop2_2, drop2_3)

	fc1_out = 64
	with tf.variable_scope('fc1_in_block'):
		fc1_1, fc1_2, fc1_3 = output_layer(fc0_1, fc0_2, fc0_3, fc1_out)

	drop_fc1_1, drop_fc1_2, drop_fc1_3 = nested_drop(fc1_1, fc1_2, fc1_3, keep_prob)

	fc2_out = n_classes
	with tf.variable_scope('fc2_in_block'):
		logits1, logits2, logits3 = output_layer(drop_fc1_1, drop_fc1_2, drop_fc1_3, fc2_out)

	return logits1, logits2, logits3
    
def inference10(x, n_classes, keep_prob, is_training):
    
	conv1_out = 64
	with tf.variable_scope('conv1_in_block'):
		conv1_1, conv1_2, conv1_3, conv1_4, conv1_5, \
            conv1_6, conv1_7, conv1_8, conv1_9, conv1_10 = conv_layer10(x, [3,3,1,conv1_out], 1, is_training)

	relu1_1, relu1_2, relu1_3, relu1_4, relu1_5, \
        relu1_6, relu1_7, relu1_8, relu1_9, relu1_10 = nested_relu10(conv1_1, conv1_2, conv1_3, conv1_4, conv1_5, \
                                                                     conv1_6, conv1_7, conv1_8, conv1_9, conv1_10, conv1_out)

	pool1_1, pool1_2, pool1_3, pool1_4, pool1_5, \
        pool1_6, pool1_7, pool1_8, pool1_9, pool1_10 = nested_pool10(relu1_1, relu1_2, relu1_3, relu1_4, relu1_5, \
                                                                     relu1_6, relu1_7, relu1_8, relu1_9, relu1_10)
	drop1_1, drop1_2, drop1_3, drop1_4, drop1_5, \
        drop1_6, drop1_7, drop1_8, drop1_9, drop1_10 = nested_drop10(pool1_1, pool1_2, pool1_3, pool1_4, pool1_5, \
                                                                     pool1_6, pool1_7, pool1_8, pool1_9, pool1_10, keep_prob)

	conv2_out = 128
	with tf.variable_scope('conv2_in_block'):
		conv2_1, conv2_2, conv2_3, conv2_4, conv2_5, \
            conv2_6, conv2_7, conv2_8, conv2_9, conv2_10 = bn_relu_conv_layer10(drop1_1, drop1_2, drop1_3, drop1_4, drop1_5, \
                                                                               drop1_6, drop1_7, drop1_8, drop1_9, drop1_10, [3, 3, conv1_out, conv2_out], 1, is_training)

	relu2_1, relu2_2, relu2_3, relu2_4, relu2_5, \
        relu2_6, relu2_7, relu2_8, relu2_9, relu2_10 = nested_relu10(conv2_1, conv2_2, conv2_3, conv2_4, conv2_5, \
                                                                   conv2_6, conv2_7, conv2_8, conv2_9, conv2_10, conv2_out)

	pool2_1, pool2_2, pool2_3, pool2_4, pool2_5, \
        pool2_6, pool2_7, pool2_8, pool2_9, pool2_10 = nested_pool10(relu2_1, relu2_2, relu2_3, relu2_4, relu2_5, \
                                                                     relu2_6, relu2_7, relu2_8, relu2_9, relu2_10)
	drop2_1, drop2_2, drop2_3, drop2_4, drop2_5, \
        drop2_6, drop2_7, drop2_8, drop2_9, drop2_10 = nested_drop10(pool2_1, pool2_2, pool2_3, pool2_4, pool2_5, \
                                                                     pool2_6, pool2_7, pool2_8, pool2_9, pool2_10, keep_prob)

	fc0_1, fc0_2, fc0_3, fc0_4, fc0_5, \
        fc0_6, fc0_7, fc0_8, fc0_9, fc0_10 = nested_fc_concat10(drop1_1, drop1_2, drop1_3, drop1_4, drop1_5, \
                                                                drop1_6, drop1_7, drop1_8, drop1_9, drop1_10, \
                                                                drop2_1, drop2_2, drop2_3, drop2_4, drop2_5, \
                                                                drop2_6, drop2_7, drop2_8, drop2_9, drop2_10)

	fc1_out = 64
	with tf.variable_scope('fc1_in_block'):
		fc1_1, fc1_2, fc1_3, fc1_4, fc1_5, \
            fc1_6, fc1_7, fc1_8, fc1_9, fc1_10 = output_layer10(fc0_1, fc0_2, fc0_3, fc0_4, fc0_5, \
                                                                fc0_6, fc0_7, fc0_8, fc0_9, fc0_10, fc1_out)

	drop_fc1_1, drop_fc1_2, drop_fc1_3, drop_fc1_4, drop_fc1_5, \
        drop_fc1_6, drop_fc1_7, drop_fc1_8, drop_fc1_9, drop_fc1_10 = nested_drop10(fc1_1, fc1_2, fc1_3, fc1_4, fc1_5, \
                                                                                    fc1_6, fc1_7, fc1_8, fc1_9, fc1_10, keep_prob)

	fc2_out = n_classes
	with tf.variable_scope('fc2_in_block'):
		logits1, logits2, logits3, logits4, logits5, \
            logits6, logits7, logits8, logits9, logits10 = output_layer10(drop_fc1_1, drop_fc1_2, drop_fc1_3, drop_fc1_4, drop_fc1_5, \
                                                                          drop_fc1_6, drop_fc1_7, drop_fc1_8, drop_fc1_9, drop_fc1_10, fc2_out)

	return logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8, logits9, logits10

def inferencen(x, n_classes, keep_prob, is_training, resize = 32):

	if resize == 32:
		resize_x = x
	else:
		temp_x = tf.image.resize_images(x, [resize, resize])
		resize_x = tf.image.resize_images(temp_x, [32,32])
    
	conv1_out = 64
	with tf.variable_scope('conv1_in_block'):
		conv1 = conv_layern(resize_x, [3,3,1,conv1_out], 1, is_training)

	relu1 = nested_relun(conv1, conv1_out)

	pool1 = nested_pooln(relu1)
	drop1 = nested_dropn(pool1, keep_prob)

	conv2_out = 128
	with tf.variable_scope('conv2_in_block'):
		conv2 = bn_relu_conv_layern(drop1, [3, 3, conv1_out, conv2_out], 1, is_training)

	relu2 = nested_relun(conv2, conv2_out)

	pool2 = nested_pooln(relu2)
	drop2 = nested_dropn(pool2, keep_prob)

	fc0 = nested_fc_concatn(drop1, drop2)

	fc1_out = 64
	with tf.variable_scope('fc1_in_block'):
		fc1 = output_layern(fc0, fc1_out)

	drop_fc1 = nested_dropn(fc1, keep_prob)

	fc2_out = n_classes
	with tf.variable_scope('fc2_in_block'):
		logits = output_layern(drop_fc1, fc2_out)

	return logits

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
