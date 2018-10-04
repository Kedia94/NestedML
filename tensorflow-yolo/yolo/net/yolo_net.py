from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from yolo.net.net import Net 

n = 5                                                                                                  # n-level
lnr = [1.0/64.0, 1.0/32.0, 1.0/16.0, 1.0/8.0, 1.0/1.0] # incremental order, last one is 1, n-array
#lnr = [1.0/2.0, 1.0/1.0]
BN_DECAY = 0.999 #0.999
BN_EPSILON = 0.00001 #0.00001

def create_variablesn(name, shape, pretrained_collection, trainable_collection):
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
  pretrained_collection.append(lv1_variables[-1])
  trainable_collection.append(lv1_variables[-1])
  for i in range(1, n):
    lv1_variables.append(tf.get_variable(name + '_l1_' + str(i), initializer=tf.random_uniform(shape1[i], -ni, ni, tf.float32, seed=None))) 
    pretrained_collection.append(lv1_variables[-1])
    trainable_collection.append(lv1_variables[-1])
    lv2_variables.append(tf.get_variable(name + '_l2_' + str(i), initializer=tf.random_uniform(shape2[i-1], -ni, ni, tf.float32, seed=None)))
    pretrained_collection.append(lv2_variables[-1])
    trainable_collection.append(lv2_variables[-1])

  return lv1_variables, lv2_variables

def batch_normalization_layer(name, input_layer, dimension, pretrained_collection, trainable_collection, is_training = True):

  beta = tf.get_variable(name + 'beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
  pretrained_collection.append(beta)
  trainable_collection.append(beta)
  gamma = tf.get_variable(name + 'gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
  pretrained_collection.append(gamma)
  trainable_collection.append(gamma)
  mu = tf.get_variable(name + 'mu', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
  pretrained_collection.append(mu)
  trainable_collection.append(mu)
  sigma = tf.get_variable(name + 'sigma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
  pretrained_collection.append(sigma)
  trainable_collection.append(sigma)

  if is_training is True:
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    train_mean = tf.assign(mu, mu * BN_DECAY + mean * (1 - BN_DECAY))
    train_var = tf.assign(sigma, sigma * BN_DECAY + variance * (1 - BN_DECAY))

    with tf.control_dependencies([train_mean, train_var]):
      return tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
  else:
    bn_layer = tf.nn.batch_normalization(input_layer, mu, sigma, beta, gamma, BN_EPSILON)

  return bn_layer

def conv_layern(input_layer, filter_shape, stride, pretrained_collection, trainable_collection, is_training):
  in_channel = input_layer.get_shape().as_list()[-1]

  bn_layer = batch_normalization_layer('l1_l2_l3', input_layer, in_channel, pretrained_collection, trainable_collection, is_training)
  bn_layer = tf.nn.relu(bn_layer)

  ni = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + filter_shape[-1])))

  filters = []
  filters.append(tf.get_variable('conv_l1',
                 initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(lnr[0]*filter_shape[3])],
                 -ni, ni,   tf.float32, seed=None)))
  pretrained_collection.append(filters[-1])
  trainable_collection.append(filters[-1])
  for i in range (1, n):
    filters.append(tf.get_variable('conv_l'+str(i+1),
                   initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(lnr[i]*filter_shape[3])-int(lnr[i-1]*filter_shape[3])],
                   -ni, ni,   tf.float32, seed=None)))
    pretrained_collection.append(filters[-1])
    trainable_collection.append(filters[-1])

  conv = []
  conv.append(tf.nn.conv2d(bn_layer, filters[0], strides=[1, stride, stride, 1], padding='SAME'))
  for i in range (1, n):
    conv.append(tf.concat((conv[i-1], tf.nn.conv2d(bn_layer, filters[i], strides=[1, stride, stride, 1], padding='SAME')), 3))

  return conv

def bn_relu_conv_layern(inputs, filter_shape, stride, pretrained_collection, trainable_collection, is_training):
  in_channel = []
  for i in range(0, n):
    in_channel.append(inputs[i].get_shape().as_list()[-1])

  bn_layer = []
  for i in range(0, n):
    temp_bn_layer = batch_normalization_layer('l'+str(i+1), inputs[i], in_channel[i], pretrained_collection, trainable_collection, is_training)
    bn_layer.append(tf.nn.relu(temp_bn_layer))

  filter1, filter2 = create_variablesn(name='conv', shape=filter_shape, pretrained_collection=pretrained_collection, trainable_collection=trainable_collection)

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

def residual_blockn(inputs, output_channel, wide_scale, pretrained_collection, trainable_collection, is_training, first_block=False):
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
      conv = conv_layern(inputs[0], [3, 3, input_channel, output_channel], stride, pretrained_collection, trainable_collection, is_training)
    else:
      conv = bn_relu_conv_layern(inputs, [3, 3, input_channel, output_channel], stride, pretrained_collection, trainable_collection, is_training)

  with tf.variable_scope('conv2_in_block'):
    conv = bn_relu_conv_layern(conv, [3, 3, output_channel, output_channel], 1, pretrained_collection, trainable_collection, is_training)

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
      conv = conv_layern(inputs[0], [3, 3, input_channel, output_channel], stride, pretrained_collection, trainable_collection, is_training)
    else:
      conv = bn_relu_conv_layern(inputs, [3, 3, input_channel, output_channel], stride, pretrained_collection, trainable_collection, is_training)

  with tf.variable_scope('conv2_in_block'):
    conv = bn_relu_conv_layern(conv, [3, 3, output_channel, output_channel], 1, pretrained_collection, trainable_collection, is_training)

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

def nested_pooln(inputs):
  pool = []
  for i in range(0, n):
    pool.append(tf.nn.max_pool(inputs[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME'))
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

def output_layern(inputs, num_labels, pretrained_collection, trainable_collection):
  input_dim = []
  for i in range(0, n):
    input_dim.append(inputs[i].get_shape().as_list()[-1])

  fc_w = []
  for i in range(0, n):
    fc_w.append(tf.get_variable('fc_weights_l'+ str(i+1), shape=[input_dim[i], num_labels], initializer=tf.initializers.variance_scaling(scale=1.0)))
    pretrained_collection.append(fc_w[-1])
    trainable_collection.append(fc_w[-1])

  fc_b = []
  for i in range(0, n):
    fc_b.append(tf.get_variable(name='fc_bias_l'+ str(i+1), shape=[num_labels], initializer=tf.zeros_initializer()))
    pretrained_collection.append(fc_b[-1])
    trainable_collection.append(fc_b[-1])

  fc_h = []
  for i in range(0, n):
    fc_h.append(tf.matmul(inputs[i], fc_w[i]) + fc_b[i])

  return fc_h

class YoloNet(Net):

  def __init__(self, common_params, net_params, test=False):
    """
    common params: a params dict
    net_params   : a params dict
    """
    super(YoloNet, self).__init__(common_params, net_params)
    #process params
    self.image_size = int(common_params['image_size'])
    self.num_classes = int(common_params['num_classes'])
    self.cell_size = int(net_params['cell_size'])
    self.boxes_per_cell = int(net_params['boxes_per_cell'])
    self.batch_size = int(common_params['batch_size'])
    self.weight_decay = float(net_params['weight_decay'])

    if not test:
      self.object_scale = float(net_params['object_scale'])
      self.noobject_scale = float(net_params['noobject_scale'])
      self.class_scale = float(net_params['class_scale'])
      self.coord_scale = float(net_params['coord_scale'])

  def inference(self, images):
    """Build the yolo model

    Args:
      images:  4-D tensor [batch_size, image_height, image_width, channels]
    Returns:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    """
    is_training = True
    conv_num = 1
    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = conv_layern(images, [7, 7, 3, 64], 2, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), images, [7, 7, 3, 64], stride=2)
    conv_num += 1

    temp_pool = nested_pooln(temp_conv)
#    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_pool, [3, 3, 64, 192], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 64, 192], stride=1)
    conv_num += 1

    temp_pool = nested_pooln(temp_conv)
#    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_pool, [1, 1, 192, 128], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [1, 1, 192, 128], stride=1)
    conv_num += 1
    
    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 128, 256], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
    conv_num += 1

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [1, 1, 256, 256], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 256, 256], stride=1)
    conv_num += 1

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 256, 512], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
    conv_num += 1

    temp_conv = nested_pooln(temp_conv)
#    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    for i in range(4):
      with tf.variable_scope('conv' + str(conv_num) + '_'):
        temp_conv = bn_relu_conv_layern(temp_conv, [1, 1, 512, 256], 1, self.pretrained_collection, self.trainable_collection, is_training)
#      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 512, 256], stride=1)
      conv_num += 1

      with tf.variable_scope('conv' + str(conv_num) + '_'):
        temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 256, 512], 1, self.pretrained_collection, self.trainable_collection, is_training)
#      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
      conv_num += 1

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [1, 1, 512, 512], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 512, 512], stride=1)
    conv_num += 1

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 512, 1024], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
    conv_num += 1

    temp_conv = nested_pooln(temp_conv)
#    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    for i in range(2):
      with tf.variable_scope('conv' + str(conv_num) + '_'):
        temp_conv = bn_relu_conv_layern(temp_conv, [1, 1, 1024, 512], 1, self.pretrained_collection, self.trainable_collection, is_training)
#      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 1024, 512], stride=1)
      conv_num += 1

      with tf.variable_scope('conv' + str(conv_num) + '_'):
        temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 512, 1024], 1, self.pretrained_collection, self.trainable_collection, is_training)
#      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
      conv_num += 1

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 1024, 1024], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1

    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 1024, 1024], 2, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=2)
    conv_num += 1

    #
    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 1024, 1024], 1, self.pretrained_collection, self.trainable_collection, is_training)
#    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1
    
    with tf.variable_scope('conv' + str(conv_num) + '_'):
      temp_conv = bn_relu_conv_layern(temp_conv, [3, 3, 1024, 1024], 1, self.pretrained_collection, self.trainable_collection, is_training)
#   temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1


    #Fully connected layer
    reshaped_conv = []
    for i in range(0, n):
      reshaped_conv.append(tf.reshape(temp_conv[i], [tf.shape(temp_conv[i])[0], -1]))

#    temp_conv = tf.reshape(temp_conv, [tf.shape(temp_conv)[0], -1])
    with tf.variable_scope('local1_'):
      local1 = output_layern(reshaped_conv, 4096, self.pretrained_collection, self.trainable_collection)
#    local1 = self.local('local1', temp_conv, 49 * 1024, 4096)


    with tf.variable_scope('local1_'):
      local1 = nested_dropn(local1, keep_prob=0.5)
#    local1 = tf.nn.dropout(local1, keep_prob=0.5)

    with tf.variable_scope('local2_'):
      local2 = output_layern(local1, self.cell_size * self.cell_size * (self.num_classes + 5 * self.boxes_per_cell), self.pretrained_collection, self.trainable_collection)
#    local2 = self.local('local2', local1, 4096, self.cell_size * self.cell_size * ( self.num_classes + 5 * self.boxes_per_cell), leaky=False)

#    local2 = nested_reshapen(local2, [tf.shape(local2[0])[0], self.cell_size, self.cell_size, self.num_classes + 5 * self.boxes_per_cell]

    reshaped_conv = []
    for i in range(0, n):
      reshaped_conv.append(tf.reshape(local2[i], [tf.shape(local2[i])[0], self.cell_size, self.cell_size, self.num_classes + 5 * self.boxes_per_cell]))
#    local2 = tf.reshape(local2, [tf.shape(local2)[0], self.cell_size, self.cell_size, self.num_classes + 5 * self.boxes_per_cell])

    return reshaped_conv

#    predicts = local2


#   return predicts

  def iou(self, boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

    #calculate the left up point
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    #intersection
    intersection = rd - lu 

    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
    
    inter_square = mask * inter_square
    
    #calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
    return inter_square/(square1 + square2 - inter_square + 1e-6)

  def cond1(self, num, object_num, loss, predict, label, nilboy):
    """
    if num < object_num
    """
    return num < object_num


  def body1(self, num, object_num, loss, predict, labels, nilboy):
    """
    calculate loss
    Args:
      predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
      labels : [max_objects, 5]  (x_center, y_center, w, h, class)
    """
    label = labels[num:num+1, :]
    label = tf.reshape(label, [-1])

    #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
    min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
    max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)

    min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
    max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

    min_x = tf.floor(min_x)
    min_y = tf.floor(min_y)

    max_x = tf.ceil(max_x)
    max_y = tf.ceil(max_y)

    temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
    objects = tf.ones(temp, tf.float32)

    temp = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    objects = tf.pad(objects, temp, "CONSTANT")

    #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
    #calculate responsible tensor [CELL_SIZE, CELL_SIZE]
    center_x = label[0] / (self.image_size / self.cell_size)
    center_x = tf.floor(center_x)

    center_y = label[1] / (self.image_size / self.cell_size)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size -center_x - 1]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    response = tf.pad(response, temp, "CONSTANT")
    #objects = response

    #calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]
    

    predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])

    predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size, self.image_size, self.image_size]

    base_boxes = np.zeros([self.cell_size, self.cell_size, 4])

    for y in range(self.cell_size):
      for x in range(self.cell_size):
        #nilboy
        base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
    base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]), [1, 1, self.boxes_per_cell, 1])

    predict_boxes = base_boxes + predict_boxes

    iou_predict_truth = self.iou(predict_boxes, label[0:4])
    #calculate C [cell_size, cell_size, boxes_per_cell]
    C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

    #calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))
    
    max_I = tf.reduce_max(I, 2, keepdims=True)

    I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

    #calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    no_I = tf.ones_like(I, dtype=tf.float32) - I 


    p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

    #calculate truth x,y,sqrt_w,sqrt_h 0-D
    x = label[0]
    y = label[1]

    sqrt_w = tf.sqrt(tf.abs(label[2]))
    sqrt_h = tf.sqrt(tf.abs(label[3]))
    #sqrt_w = tf.abs(label[2])
    #sqrt_h = tf.abs(label[3])

    #calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    p_x = predict_boxes[:, :, :, 0]
    p_y = predict_boxes[:, :, :, 1]

    #p_sqrt_w = tf.sqrt(tf.abs(predict_boxes[:, :, :, 2])) * ((tf.cast(predict_boxes[:, :, :, 2] > 0, tf.float32) * 2) - 1)
    #p_sqrt_h = tf.sqrt(tf.abs(predict_boxes[:, :, :, 3])) * ((tf.cast(predict_boxes[:, :, :, 3] > 0, tf.float32) * 2) - 1)
    #p_sqrt_w = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 2]))
    #p_sqrt_h = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 3]))
    #p_sqrt_w = predict_boxes[:, :, :, 2]
    #p_sqrt_h = predict_boxes[:, :, :, 3]
    p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
    p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
    #calculate truth p 1-D tensor [NUM_CLASSES]
    P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

    #calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    p_P = predict[:, :, 0:self.num_classes]

    #class_loss
    class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
    #class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

    #object_loss
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale
    #object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

    #noobject_loss
    #noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

    #coord_loss
    coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_y - y)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ self.image_size +
                 tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/self.image_size) * self.coord_scale

    nilboy = I

    return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss, loss[3] + coord_loss], predict, labels, nilboy


  def loss(self, predicts, labels, objects_num):
    """Add Loss to all the trainable variables

    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """
    class_loss = tf.constant(0, tf.float32)
    object_loss = tf.constant(0, tf.float32)
    noobject_loss = tf.constant(0, tf.float32)
    coord_loss = tf.constant(0, tf.float32)
    loss = [0, 0, 0, 0]
    tuple_results = []
    for nn in range(0, n):
      for i in range(self.batch_size):
        predict = predicts[nn][i, :, :, :]
        label = labels[i, :, :]
        object_num = objects_num[i]
        nilboy = tf.ones([7,7,2])
        tuple_results.append(tf.while_loop(self.cond1, self.body1, [tf.constant(0), object_num, [class_loss, object_loss, noobject_loss, coord_loss], predict, label, nilboy]))
        for j in range(4):
          loss[j] = loss[j] + tuple_results[-1][2][j]
        nilboy = tuple_results[-1][5]

    tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size/n)

    tf.summary.scalar('class_loss', loss[0]/self.batch_size)
    tf.summary.scalar('object_loss', loss[1]/self.batch_size)
    tf.summary.scalar('noobject_loss', loss[2]/self.batch_size)
    tf.summary.scalar('coord_loss', loss[3]/self.batch_size)
#    tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size/n )

    return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy
