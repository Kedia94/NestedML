import tensorflow as tf
import numpy as np

n = 5                                                                                                # n-level
lnr = [1.0/16.0, 1.0/8.0, 1.0/4.0, 1.0/2.0, 1.0/1.0] # incremental order, last one is 1, n-array
#lnr = [1.0/2.0, 1.0/1.0]
BN_DECAY = 0.999 #0.999
BN_EPSILON = 0.00001 #0.00001

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

def batch_normalization_layer(name, input_layer, dimension, is_training = True):

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

def conv_relu_layern_initial(inputs, filter_shape, stride, pad = 0, bias_init = 0.0, is_training=True):
    if (pad > 0):
        paddings = [[0,0], [pad, pad], [pad,pad], [0,0]]
        inputs = tf.pad(inputs, paddings, "CONSTANT")

    biases = []
    biases.append(tf.Variable(tf.constant(bias_init, shape=[int(lnr[0] * filter_shape[3])], dtype=tf.float32), trainable=is_training, name='biases0'))
    for i in range(1, n):
        biases.append(tf.concat((biases[-1],tf.Variable(tf.constant(bias_init, shape=[int((lnr[i])*filter_shape[3]) - int((lnr[i-1])*filter_shape[3])], dtype=tf.float32), trainable=is_training, name='biases'+str(i))), 0))

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
    conv.append(tf.nn.conv2d(inputs, filters[0], strides=[1, stride, stride, 1], padding='SAME'))

    for i in range(1, n):
        temp_conv = tf.concat((conv[i-1], tf.nn.conv2d(inputs, filters[i], strides=[1, stride, stride, 1], padding='SAME')), 3)
        out = tf.nn.bias_add(temp_conv, biases[i])
        out2 = tf.nn.relu(out)
        conv.append(out2)

    return conv

def conv_relu_layern(inputs, filter_shape, stride, pad = 0, bias_init = 0.0, is_training=True):
    if (pad > 0):
        paddings = [[0,0], [pad, pad], [pad,pad], [0,0]]
        for i in range(0, n):
            inputs[i] = tf.pad(inputs[i], paddings, "CONSTANT")

    biases = []
    biases.append(tf.Variable(tf.constant(bias_init, shape=[int(lnr[0] * filter_shape[3])], dtype=tf.float32), trainable=is_training, name='biases0'))
    for i in range(1, n):
        biases.append(tf.concat((biases[-1],tf.Variable(tf.constant(bias_init, shape=[int((lnr[i]-lnr[i-1])*filter_shape[3])], dtype=tf.float32), trainable=is_training, name='biases'+str(i))), 0))

    filter1, filter2 = create_variablesn(name='conv', shape=filter_shape)

    conv = []
    conv.append(tf.nn.conv2d(inputs[0], filter1[0], strides=[1, stride, stride, 1], padding='SAME'))

    for i in range(1, n):
        temp_conv = tf.concat((tf.add(tf.nn.conv2d(inputs[i][:, :, :, :int(lnr[0] * filter_shape[2])], filter1[0], strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(inputs[i][:, :, :, int(lnr[0] * filter_shape[2]):int(lnr[1] * filter_shape[2])], filter1[1], strides=[1, stride, stride, 1], padding='SAME')),
                              tf.nn.conv2d(inputs[i][:, :, :, :int(lnr[1] * filter_shape[2])], filter2[0], strides=[1, stride, stride, 1], padding='SAME')), 3)
        for j in range(1, i):
            temp_conv = tf.concat((tf.add(temp_conv,
                                  tf.nn.conv2d(inputs[i][:, :, :, int(lnr[j] * filter_shape[2]):int(lnr[j+1] * filter_shape[2])], filter1[j+1], strides=[1, stride, stride, 1], padding='SAME')),
                                  tf.nn.conv2d(inputs[i][:, :, :, :int(lnr[j+1] * filter_shape[2])], filter2[j], strides=[1, stride, stride, 1], padding='SAME')), 3)
        out = tf.nn.bias_add(temp_conv, biases[i])
        out2 = tf.nn.relu(out)
        conv.append(out2)

    return conv

def conv_relu_layern_group2(inputs, filter_shape, stride, pad = 0, bias_init = 0.0, is_training=True):
    if (pad > 0):
        paddings = [[0,0], [pad, pad], [pad,pad], [0,0]]
        for i in range(0, n):
            inputs[i] = tf.pad(inputs[i], paddings, "CONSTANT")

    biases = []
    biases.append(tf.Variable(tf.constant(bias_init, shape=[int(lnr[0] * filter_shape[3])], dtype=tf.float32), trainable=is_training, name='biases0'))
    for i in range(1, n):
        biases.append(tf.concat((biases[-1],tf.Variable(tf.constant(bias_init, shape=[int((lnr[i]-lnr[i-1])*filter_shape[3])], dtype=tf.float32), trainable=is_training, name='biases'+str(i))), 0))

    half_filter_shape = filter_shape.copy()
    half_filter_shape[-1] = int(half_filter_shape[-1] / 2)

    filter11, filter12 = create_variablesn(name='conv1', shape=half_filter_shape)
    filter21, filter22 = create_variablesn(name='conv2', shape=half_filter_shape)

    conv = []

    inp1, inp2 = tf.split(inputs[0], num_or_size_splits=2, axis=3)
    temp_conv1 = tf.nn.conv2d(inp1, filter11[0], strides=[1, stride, stride, 1], padding='SAME')
    temp_conv2 = tf.nn.conv2d(inp2, filter21[0], strides=[1, stride, stride, 1], padding='SAME')
    temp_conv = tf.concat([temp_conv1, temp_conv2], axis=3)
    out = tf.nn.bias_add(temp_conv, biases[0])
    out2 = tf.nn.relu(out)
    conv.append(out2)

    for i in range(1, n):
        inp1, inp2 = tf.split(inputs[i], num_or_size_splits=2, axis=3)
        temp_conv1 = tf.concat((tf.add(tf.nn.conv2d(inp1[:, :, :, :int(lnr[0] * half_filter_shape[2])], filter11[0], strides=[1, stride, stride, 1], padding='SAME'),
                               tf.nn.conv2d(inp1[:, :, :, int(lnr[0] * half_filter_shape[2]):int(lnr[1] * half_filter_shape[2])], filter11[1], strides=[1, stride, stride, 1], padding='SAME')),
                               tf.nn.conv2d(inp1[:, :, :, :int(lnr[1] * half_filter_shape[2])], filter12[0], strides=[1, stride, stride, 1], padding='SAME')), 3)
        temp_conv2 = tf.concat((tf.add(tf.nn.conv2d(inp2[:, :, :, :int(lnr[0] * half_filter_shape[2])], filter21[0], strides=[1, stride, stride, 1], padding='SAME'),
                               tf.nn.conv2d(inp2[:, :, :, int(lnr[0] * half_filter_shape[2]):int(lnr[1] * half_filter_shape[2])], filter21[1], strides=[1, stride, stride, 1], padding='SAME')),
                               tf.nn.conv2d(inp2[:, :, :, :int(lnr[1] * half_filter_shape[2])], filter22[0], strides=[1, stride, stride, 1], padding='SAME')), 3)
        for j in range(1, i):
            temp_conv1 = tf.concat((tf.add(temp_conv1,
                                   tf.nn.conv2d(inp1[:, :, :, int(lnr[j] * half_filter_shape[2]):int(lnr[j+1] * half_filter_shape[2])], filter11[j+1], strides=[1, stride, stride, 1], padding='SAME')),
                                   tf.nn.conv2d(inp1[:, :, :, :int(lnr[j+1] * half_filter_shape[2])], filter12[j], strides=[1, stride, stride, 1], padding='SAME')), 3)
            temp_conv2 = tf.concat((tf.add(temp_conv2,
                                   tf.nn.conv2d(inp2[:, :, :, int(lnr[j] * half_filter_shape[2]):int(lnr[j+1] * half_filter_shape[2])], filter21[j+1], strides=[1, stride, stride, 1], padding='SAME')),
                                   tf.nn.conv2d(inp2[:, :, :, :int(lnr[j+1] * half_filter_shape[2])], filter22[j], strides=[1, stride, stride, 1], padding='SAME')), 3)
        temp_conv = tf.concat([temp_conv1, temp_conv2], axis=3)
        out = tf.nn.bias_add(temp_conv, biases[i])
        out2 = tf.nn.relu(out)
        conv.append(out2)
        
    return conv

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

def nested_pooln(inputs, kernel=2):
    pool = []
    for i in range(0, n):
        pool.append(tf.nn.max_pool(inputs[i], ksize=[1, kernel, kernel, 1], strides=[1,2,2,1], padding = 'SAME'))
    return pool

def nested_dropn(inputs, keep_prob):
    drop = []
    for i in range(0, n):
        drop.append(tf.nn.dropout(inputs[i], keep_prob=keep_prob))
    return drop

def nested_reshapen(inputs, shape):
    reshape = []
    for i in range(0, n):
        reshape.append(tf.reshape(inputs[i], shape))
    return reshape

def nested_local_response_normalization(inputs, depth_radius, alpha, beta):
    normal = []
    for i in range(0, n):
        normal.append(tf.nn.local_response_normalization(inputs[i], depth_radius, alpha, beta, name='_'+str(i)))
    return normal

def fc_layern(inputs, num_labels):
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

def fc_relu_layern(inputs, num_labels):
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

    relu = []
    for i in range(0, n):
        relu.append(tf.nn.relu(fc_h[i], name='fc_relu_1'))

    return relu

class TRACKNET: 
    def __init__(self, batch_size, train = True):
        self.parameters = {}
        self.batch_size = batch_size
        self.target = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
        self.image = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
        self.bbox = tf.placeholder(tf.float32, [batch_size, 4])
        self.train = train
        self.wd = 0.0005
    def build(self):
        ########### for target ###########
        # [filter_height, filter_width, in_channels, out_channels]
        print(self.target.shape)
        with tf.variable_scope('target_conv1_'):
            target_conv1 = conv_relu_layern_initial(self.target, [11, 11, 3, 96], 4, is_training=True)
#        self.target_conv1 = self._conv_relu_layer(bottom = self.target, filter_size = [11, 11, 3, 96],
#                                                    strides = [1,4,4,1], name = "target_conv_1")
        
        # now 55 x 55 x 96
        with tf.variable_scope('target_pool1_'):
            target_pool1 = nested_pooln(target_conv1, kernel=3) 
#        self.target_pool1 = tf.nn.max_pool(self.target_conv1, ksize = [1, 3, 3, 1], strides=[1, 2, 2, 1],
#                                                    padding='VALID', name='target_pool1')
        # now 27 x 27 x 96
        with tf.variable_scope('target_lrn1_'):
            target_lrn1 = nested_local_response_normalization(target_pool1, depth_radius = 2, alpha = 0.0001, beta = 0.75)
#        self.target_lrn1 = tf.nn.local_response_normalization(self.target_pool1, depth_radius = 2, alpha=0.0001,
#                                                    beta=0.75, name="target_lrn1")
        # now 27 x 27 x 96

        with tf.variable_scope('target_conv2_'):
            target_conv2 = conv_relu_layern_group2(target_lrn1, [5, 5, 48, 256], 1, pad=2, bias_init = 1.0, is_training=True)
#        self.target_conv2 = self._conv_relu_layer(bottom = self.target_lrn1,filter_size = [5, 5, 48, 256],
#                                                    strides = [1,1,1,1], pad = 2, bias_init = 1.0, group = 2, name="target_conv_2")
        # now 27 x 27 x 256
        with tf.variable_scope('target_pool2_'):
            target_pool2 = nested_pooln(target_conv2, kernel=3) 
#        self.target_pool2 = tf.nn.max_pool(self.target_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                                                    padding='VALID', name='target_pool2')
        # now 13 x 13 x 256
        with tf.variable_scope('target_lrn2_'):
            target_lrn2 = nested_local_response_normalization(target_pool2, depth_radius = 2, alpha = 0.0001, beta = 0.75)
#        self.target_lrn2 = tf.nn.local_response_normalization(self.target_pool2, depth_radius = 2, alpha=0.0001,
#                                                    beta=0.75, name="target_lrn2")
        # now 13 x 13 x 256
        with tf.variable_scope('target_conv3_'):
            target_conv3 = conv_relu_layern(target_lrn2, [3, 3, 256, 384], 1, pad=1, is_training=True)
#        self.target_conv3 = self._conv_relu_layer(bottom = self.target_lrn2,filter_size = [3, 3, 256, 384],
#                                                    strides = [1,1,1,1], pad = 1, name="target_conv_3")
        # now 13 x 13 x 384
        with tf.variable_scope('target_conv4_'):
            target_conv4 = conv_relu_layern_group2(target_conv3, [3, 3, 192, 384], 1, pad = 1, bias_init = 1.0, is_training=True)
#        self.target_conv4 = self._conv_relu_layer(bottom = self.target_conv3,filter_size = [3, 3, 192, 384], bias_init = 1.0, 
#                                                    strides = [1,1,1,1], pad = 1, group = 2, name="target_conv_4")
        # now 13 x 13 x 384
        with tf.variable_scope('target_conv5_'):
            target_conv5 = conv_relu_layern_group2(target_conv4, [3, 3, 192, 256], 1, pad = 1, bias_init = 1.0, is_training=True)
#        self.target_conv5 = self._conv_relu_layer(bottom = self.target_conv4,filter_size = [3, 3, 192, 256], bias_init = 1.0, 
#                                                    strides = [1,1,1,1], pad = 1, group = 2, name="target_conv_5")
        # now 13 x 13 x 256
        with tf.variable_scope('target_pool5_'):
            target_pool5 = nested_pooln(target_conv5, kernel=3) 
#        self.target_pool5 = tf.nn.max_pool(self.target_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                                                    padding='VALID', name='target_pool5')
        # now 6 x 6 x 256
        

        ########### for image ###########
        # [filter_height, filter_width, in_channels, out_channels]
        with tf.variable_scope('image_conv1_'):
            image_conv1 = conv_relu_layern_initial(self.image, [11, 11, 3, 96], 4, is_training=True)
#        self.image_conv1 = self._conv_relu_layer(bottom = self.image, filter_size = [11, 11, 3, 96],
#                                                    strides = [1,4,4,1], name = "image_conv_1")
        
        # now 55 x 55 x 96
        with tf.variable_scope('image_pool1_'):
            image_pool1 = nested_pooln(image_conv1, kernel=3) 
#        self.image_pool1 = tf.nn.max_pool(self.image_conv1, ksize = [1, 3, 3, 1], strides=[1, 2, 2, 1],
#                                                   padding='VALID', name='image_pool1')

        # now 27 x 27 x 96
        with tf.variable_scope('image_lrn1_'):
            image_lrn1 = nested_local_response_normalization(image_pool1, depth_radius = 2, alpha = 0.0001, beta = 0.75)
#        self.image_lrn1 = tf.nn.local_response_normalization(self.image_pool1, depth_radius = 2, alpha=0.0001,
#                                                    beta=0.75, name="image_lrn1")

        # now 27 x 27 x 96
        with tf.variable_scope('image_conv2_'):
            image_conv2 = conv_relu_layern_group2(image_lrn1, [5, 5, 48, 256], 1, pad=2, bias_init = 1.0, is_training=True)
#        self.image_conv2 = self._conv_relu_layer(bottom = self.image_lrn1,filter_size = [5, 5, 48, 256],
#                                                    strides = [1,1,1,1], pad = 2, bias_init = 1.0, group = 2, name="image_conv_2")

        # now 27 x 27 x 256
        with tf.variable_scope('image_pool2_'):
            image_pool2 = nested_pooln(image_conv2, kernel=3) 
#        self.image_pool2 = tf.nn.max_pool(self.image_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                                                    padding='VALID', name='image_pool2')

        # now 13 x 13 x 256
        with tf.variable_scope('image_lrn2_'):
            image_lrn2 = nested_local_response_normalization(image_pool2, depth_radius = 2, alpha = 0.0001, beta = 0.75)
#        self.image_lrn2 = tf.nn.local_response_normalization(self.image_pool2, depth_radius = 2, alpha=0.0001,
#                                                    beta=0.75, name="image_lrn2")

        # now 13 x 13 x 256
        with tf.variable_scope('image_conv3_'):
            image_conv3 = conv_relu_layern(image_lrn2, [3, 3, 256, 384], 1, pad=1, is_training=True)
#        self.image_conv3 = self._conv_relu_layer(bottom = self.image_lrn2,filter_size = [3, 3, 256, 384],
#                                                    strides = [1,1,1,1], pad = 1, name="image_conv_3")

        # now 13 x 13 x 384
        with tf.variable_scope('image_conv4_'):
            image_conv4 = conv_relu_layern_group2(image_conv3, [3, 3, 192, 384], 1, pad = 1, bias_init = 1.0, is_training=True)
#        self.image_conv4 = self._conv_relu_layer(bottom = self.image_conv3,filter_size = [3, 3, 192, 384], 
#                                                    strides = [1,1,1,1], pad = 1, group = 2, name="image_conv_4")

        # now 13 x 13 x 384
        with tf.variable_scope('image_conv5_'):
            image_conv5 = conv_relu_layern_group2(image_conv4, [3, 3, 192, 256], 1, pad = 1, bias_init = 1.0, is_training=True)
#        self.image_conv5 = self._conv_relu_layer(bottom = self.image_conv4,filter_size = [3, 3, 192, 256], bias_init = 1.0, 
#                                                    strides = [1,1,1,1], pad = 1, group = 2, name="image_conv_5")

        # now 13 x 13 x 256
        with tf.variable_scope('image_pool5_'):
            image_pool5 = nested_pooln(image_conv5, kernel=3) 
#        self.image_pool5 = tf.nn.max_pool(self.image_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                                                    padding='VALID', name='image_pool5')

        # now 6 x 6 x 256
        # tensorflow layer: n * w * h * c
        # but caffe layer is: n * c * h * w

        # tensorflow kernel: h * w * in_c * out_c
        # caffe kernel: out_c * in_c * h * w

        ########### Concatnate two layers ###########
        concat = []
        for i in range(0, n):
            concat.append(tf.concat([target_pool5[i], image_pool5[i]], axis=3))
#        self.concat = tf.concat([self.target_pool5, self.image_pool5], axis = 3) # 0, 1, 2, 3 - > 2, 3, 1, 0

        # important, since caffe has different layer dimension order
        for i in range(0, n):
            concat[i] = tf.transpose(concat[i], perm=[0,3,1,2])
#        self.concat = tf.transpose(self.concat, perm=[0,3,1,2]) 

        ########### fully connencted layers ###########
        # 6 * 6 * 256 * 2 == 18432
        # assert self.fc1.get_shape().as_list()[1:] == [6, 6, 512]
        print(concat[0].shape)
        fc1 = []
        with tf.variable_scope('fc1_'):
            for i in range(0, n):
                fc1.append(self._fc_relu_layers(concat[i], dim = 4096, name = "fc1"))
#        self.fc1 = self._fc_relu_layers(self.concat, dim = 4096, name = "fc1")
        if (self.train):
            fc1 = nested_dropn(fc1, 0.5)
#            self.fc1 = tf.nn.dropout(self.fc1, 0.5)

        fc2 = []
        with tf.variable_scope('fc2_'):
            for i in range(0, n):
                fc2.append(self._fc_relu_layers(fc1[i], dim = 4096, name = "fc2"))
#        self.fc2 = self._fc_relu_layers(self.fc1, dim = 4096, name = "fc2")
        if (self.train):
            fc2 = nested_dropn(fc2, 0.5)
#            self.fc2 = tf.nn.dropout(self.fc2, 0.5)

        fc3 = []
        with tf.variable_scope('fc3_'):
            for i in range(0, n):
                fc3.append(self._fc_relu_layers(fc2[i], dim = 4096, name = "fc3"))
#        self.fc3 = self._fc_relu_layers(self.fc2, dim = 4096, name = "fc3")
        if (self.train):    
            fc3 = nested_dropn(fc3, 0.5)
#            self.fc3 = tf.nn.dropout(self.fc3, 0.5)

        self.fc4 = []
        with tf.variable_scope('fc4_'):
            for i in range(0, n):
                self.fc4.append(self._fc_layers(fc3[i], dim = 4, name = "fc4"))
#        self.fc4 = self._fc_layers(self.fc3, dim = 4, name = "fc4")

#        self.print_shapes()
        self.loss = self._loss_layer(self.fc4, self.bbox ,name = "loss")
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_weight_loss')
        self.loss_wdecay = self.loss + l2_loss

    def _loss_layer(self, bottom, label, name = None):
        diff = []
        diff_flat = []
        loss = 0
        for i in range(n):
            diff.append(tf.subtract(self.fc4[i], self.bbox))
#        diff = tf.subtract(self.fc4, self.bbox)
            diff_flat.append(tf.abs(tf.reshape(diff[-1], [-1])))
#        diff_flat = tf.abs(tf.reshape(diff,[-1]))
            loss += tf.reduce_sum(diff_flat[-1], name = name)
        loss /= n
#        loss = tf.reduce_sum(diff_flat, name = name)
        return loss

    def _conv_relu_layer(self,bottom,filter_size, strides, pad = 0,bias_init = 0.0, group = 1, trainable = False, name = None):
        with tf.name_scope(name) as scope:

            if (pad > 0):
                paddings = [[0,0],[pad,pad],[pad,pad],[0,0]]
                bottom = tf.pad(bottom, paddings, "CONSTANT")
            kernel = tf.Variable(tf.truncated_normal(filter_size, dtype=tf.float32,
                                                     stddev=1e-2), trainable=trainable, name='weights')
            biases = tf.Variable(tf.constant(bias_init, shape=[filter_size[3]], dtype=tf.float32), trainable=trainable, name='biases')
            self.parameters[name] = [kernel, biases]
            if (group == 1):
                conv = tf.nn.conv2d(bottom, kernel, strides, padding='VALID')
                out = tf.nn.bias_add(conv, biases)
            elif (group == 2):
                kernel1, kernel2 = tf.split(kernel, num_or_size_splits=group, axis=3)
                bottom1, bottom2 = tf.split(bottom, num_or_size_splits=group, axis=3)
                conv1 = tf.nn.conv2d(bottom1, kernel1, strides, padding='VALID')
                conv2 = tf.nn.conv2d(bottom2, kernel2, strides, padding='VALID')
                conv = tf.concat([conv1, conv2], axis=3)
                out = tf.nn.bias_add(conv, biases)
            else:
                raise TypeError("number of groups not supported")

            # if not tf.get_variable_scope().reuse:
            #     weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd,
            #                            name='kernel_loss')
            #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
            #                      weight_decay)


            out2 = tf.nn.relu(out, name=scope)
            _activation_summary(out2)
            out2 = tf.Print(out2, [tf.shape(out2)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return out2

    def _fc_relu_layers(self, bottom, dim, name = None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                    dtype=tf.float32, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float32), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            fc_weights = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias)
            self.parameters[name] = [weights, bias]


            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                       name='fc_relu_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)



            top = tf.nn.relu(fc_weights, name=scope)
            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return top

    def _fc_layers(self, bottom, dim, name = None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                    dtype=tf.float32, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float32), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            top = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias, name=scope)
            self.parameters[name] = [weights, bias]

            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                       name='fc_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)

            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return top
    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var
    def print_shapes(self):
        print("%s:"%(self.image_conv1),self.image_conv1.get_shape().as_list())
        print("%s:"%(self.image_pool1),self.image_pool1.get_shape().as_list())
        print("%s:"%(self.image_lrn1),self.image_lrn1.get_shape().as_list())
        print("%s:"%(self.image_conv2),self.image_conv2.get_shape().as_list())
        print("%s:"%(self.image_pool2),self.image_pool2.get_shape().as_list())
        print("%s:"%(self.image_lrn2),self.image_lrn2.get_shape().as_list())
        print("%s:"%(self.image_conv3),self.image_conv3.get_shape().as_list())
        print("%s:"%(self.image_conv4),self.image_conv4.get_shape().as_list())
        print("%s:"%(self.image_conv5),self.image_conv5.get_shape().as_list())
        print("%s:"%(self.image_pool5),self.image_pool5.get_shape().as_list())
        print("%s:"%(self.concat),self.concat.get_shape().as_list())
        print("%s:"%(self.fc1),self.fc1.get_shape().as_list())
        print("%s:"%(self.fc2),self.fc2.get_shape().as_list())
        print("%s:"%(self.fc3),self.fc3.get_shape().as_list())
        print("%s:"%(self.fc4),self.fc4.get_shape().as_list())
        print("kernel_sizes:")
        for key in self.parameters:
            print("%s:"%(key),self.parameters[key][0].get_shape().as_list())

    def load_weight_from_dict(self,weights_dict,sess):
        # for convolutional layers
        sess.run(self.parameters['target_conv_1'][0].assign(weights_dict['conv1']['weights']))
        sess.run(self.parameters['target_conv_2'][0].assign(weights_dict['conv2']['weights']))
        sess.run(self.parameters['target_conv_3'][0].assign(weights_dict['conv3']['weights']))
        sess.run(self.parameters['target_conv_4'][0].assign(weights_dict['conv4']['weights']))
        sess.run(self.parameters['target_conv_5'][0].assign(weights_dict['conv5']['weights']))
        sess.run(self.parameters['image_conv_1'][0].assign(weights_dict['conv1_p']['weights']))
        sess.run(self.parameters['image_conv_2'][0].assign(weights_dict['conv2_p']['weights']))
        sess.run(self.parameters['image_conv_3'][0].assign(weights_dict['conv3_p']['weights']))
        sess.run(self.parameters['image_conv_4'][0].assign(weights_dict['conv4_p']['weights']))
        sess.run(self.parameters['image_conv_5'][0].assign(weights_dict['conv5_p']['weights']))

        sess.run(self.parameters['target_conv_1'][1].assign(weights_dict['conv1']['bias']))
        sess.run(self.parameters['target_conv_2'][1].assign(weights_dict['conv2']['bias']))
        sess.run(self.parameters['target_conv_3'][1].assign(weights_dict['conv3']['bias']))
        sess.run(self.parameters['target_conv_4'][1].assign(weights_dict['conv4']['bias']))
        sess.run(self.parameters['target_conv_5'][1].assign(weights_dict['conv5']['bias']))
        sess.run(self.parameters['image_conv_1'][1].assign(weights_dict['conv1_p']['bias']))
        sess.run(self.parameters['image_conv_2'][1].assign(weights_dict['conv2_p']['bias']))
        sess.run(self.parameters['image_conv_3'][1].assign(weights_dict['conv3_p']['bias']))
        sess.run(self.parameters['image_conv_4'][1].assign(weights_dict['conv4_p']['bias']))
        sess.run(self.parameters['image_conv_5'][1].assign(weights_dict['conv5_p']['bias']))

        # for fully connected layers
        sess.run(self.parameters['fc1'][0].assign(weights_dict['fc6-new']['weights']))
        sess.run(self.parameters['fc2'][0].assign(weights_dict['fc7-new']['weights']))
        sess.run(self.parameters['fc3'][0].assign(weights_dict['fc7-newb']['weights']))
        sess.run(self.parameters['fc4'][0].assign(weights_dict['fc8-shapes']['weights']))

        sess.run(self.parameters['fc1'][1].assign(weights_dict['fc6-new']['bias']))
        sess.run(self.parameters['fc2'][1].assign(weights_dict['fc7-new']['bias']))
        sess.run(self.parameters['fc3'][1].assign(weights_dict['fc7-newb']['bias']))
        sess.run(self.parameters['fc4'][1].assign(weights_dict['fc8-shapes']['bias']))


    
    def test(self):
        sess = tf.Session()
        a = np.full((self.batch_size,227,227,3), 1) # numpy.full(shape, fill_value, dtype=None, order='C')
        b = np.full((self.batch_size,227,227,3), 2)
        sess.run(tf.global_variables_initializer())

        sess.run([self.fc4],feed_dict={self.image:a, self.target:b})





def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.debug("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)

if __name__ == "__main__":
    tracknet = TRACKNET(10)
    tracknet.build()
    sess = tf.Session()
    a = np.full((tracknet.batch_size,227,227,3), 1)
    b = np.full((tracknet.batch_size,227,227,3), 2)
    sess.run(tf.global_variables_initializer())
    sess.run([tracknet.image_pool5],feed_dict={tracknet.image:a, tracknet.target:b})



