import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


IS_TRAIN_PHASE = tf.placeholder(dtype=tf.bool, name='is_train_phase')


def bn(input, decay=0.9, epsilon=1e-5, scope='bn'):
    with tf.variable_scope(scope) as scope:
        bn = tf.cond(IS_TRAIN_PHASE,
            lambda: slim.batch_norm(input, decay=decay, epsilon=epsilon, center=True, scale=True,
                                    is_training=True, reuse=None,
                                    updates_collections=None, scope=scope),
            lambda: slim.batch_norm(input, decay=decay, epsilon=epsilon, center=True, scale=True,
                                    is_training=False, reuse=True,
                                    updates_collections=None, scope=scope))
    return bn


def relu(input, name='relu'):
    act = tf.nn.relu(input, name=name)
    return act


def conv2d(input, num_kernels=1, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, name='conv'):
    input_shape = input.get_shape().as_list()
    assert len(input_shape) == 4
    C = input_shape[3]
    H = kernel_size[0]
    W = kernel_size[1]
    K = num_kernels

    # [filter_height, filter_width, in_channels, out_channels]
    w = tf.get_variable(name=name+'_weight', shape=[H, W, C, K], initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(input, w, strides=stride, padding=padding, name=name)
    if has_bias:
        b = tf.get_variable(name=name+'_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        conv += b
    return conv


def linear(input, num_hiddens=1, has_bias=True, name='linear'):
    input_shape = input.get_shape().as_list()
    assert len(input_shape) == 2
    C = input_shape[1]
    K = num_hiddens

    w = tf.get_variable(name=name+'_weight', shape=[C, K], initializer=tf.truncated_normal_initializer(stddev=0.1))
    dense = tf.matmul(input, w, name=name)
    if has_bias:
        b = tf.get_variable(name=name+'_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        dense += b
    return dense


def avgpool(input, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, is_global_pool=False, name='avg'):
    if is_global_pool:
        input_shape = input.get_shape().as_list()
        assert len(input_shape) == 4
        H = input_shape[1]
        W = input_shape[2]
        pool = tf.nn.avg_pool(input, ksize=[1, H, W, 1], strides=[1, H, W, 1], padding='VALID', name=name)
        pool = flatten(pool)
    else:
        H = kernel_size[0]
        W = kernel_size[1]
        pool = tf.nn.avg_pool(input, ksize=[1, H, W, 1], strides=stride, padding=padding, name=name)
    return pool


def flatten(input, name='flat'):
    input_shape = input.get_shape().as_list()       # list: [None, 9, 2]
    dim = np.prod(input_shape[1:])                  # dim = prod(9,2) = 18
    flat = tf.reshape(input, [-1, dim], name=name)  # -1 means "all"
    return flat


def concat(input, axis=3, name='cat'):
    cat = tf.concat(axis=axis, values=input, name=name)
    return cat


def upsample2d(input, factor=2, has_bias=True, trainable=True, name='upsample2d'):
    def make_upsample_filter(size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

    input_shape = input.get_shape().as_list()
    assert len(input_shape) == 4
    # N = input_shape[0]
    # H = input_shape[1]
    # W = input_shape[2]
    C = input_shape[3]

    size = 2 * factor - factor % 2
    filter = make_upsample_filter(size)
    weights = np.zeros(shape=(size, size, C, C), dtype=np.float32)
    for c in range(C):
        weights[:, :, c, c] = filter
    init = tf.constant_initializer(value=weights, dtype=tf.float32)

    output_shape = tf.stack([tf.shape(input)[0], tf.shape(input)[1]*factor, tf.shape(input)[2]*factor, tf.shape(input)[3]])
    w = tf.get_variable(name=name+'_weight', shape=[size, size, C, C], initializer=init, trainable=trainable)
    deconv = tf.nn.conv2d_transpose(name=name, value=input, filter=w, output_shape=output_shape, strides=[1, factor, factor, 1], padding='SAME')
    if has_bias:
        b = tf.get_variable(name=name+'_bias', shape=[C], initializer=tf.constant_initializer(0.0))
        deconv += b
    return deconv


def conv2d_bn_relu(input, num_kernels=1, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='conv'):
    with tf.variable_scope(name):
        block = conv2d(input, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding, has_bias=False)
        block = bn(block)
        block = relu(block)
    return block


def linear_bn_relu(input, num_hiddens=1, name='conv'):
    with tf.variable_scope(name):
        block = linear(input, num_hiddens=num_hiddens, has_bias=False)
        block = bn(block)
        block = relu(block)
    return block


def mobilenet(inputs,
              down4x=False,
              width_multiplier=1,
              scope='MobileNet'):
    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  width_multiplier,
                                  sc,
                                  downsample=False):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        act1 = bn(inputs, scope=sc+'/dw_act')
        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(act1,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc+'/depthwise_conv')

        act2 = bn(depthwise_conv, scope=sc+'/pw_act')
        pointwise_conv = slim.convolution2d(act2,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc+'/pointwise_conv')
        return pointwise_conv, act1

    with tf.variable_scope(scope):
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None):
            with slim.arg_scope([slim.batch_norm],
                                activation_fn=tf.nn.relu,
                                fused=True):
                net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, scope='conv_1')
                net,   _ = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net, act = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net,   _ = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                shortcut = slim.convolution2d(act, 128, [1, 1], stride=2)
                net += shortcut
                shortcut = net
                net, act = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, downsample=not down4x, sc='conv_ds_5')
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_6')
                if not down4x:
                    shortcut = slim.convolution2d(act, 256, [1, 1], stride=2)
                net += shortcut
                shortcut = net
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_7')
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_8')
                net += shortcut
                shortcut = net
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_9')
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_10')
                net += shortcut
                shortcut = net
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_11')
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_12')
                net += shortcut
                shortcut = net
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_13')
                net,   _ = _depthwise_separable_conv(net, 128 if down4x else 256, width_multiplier, sc='conv_ds_14')
                net += shortcut
                net = bn(net, scope='act')
    return net
