from typing import List, Tuple

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import _preprocess_padding
from tensorflow.python.keras.backend_config import image_data_format
from tensorflow.python.keras.layers.pooling import Pooling2D, Pooling1D, AveragePooling1D, MaxPooling2D, \
    AveragePooling2D
from tensorflow.python.ops.nn_ops import _get_sequence
from tensorflow.python.util import deprecation
from informationPoolSupport import KL_div,KL_div2,sample_lognormal,batch_average


@tf.function
def op_pad(padding, x, ksize, strides):
    paddings = tf.constant([[0, 0], [0, 0]])
    if padding == 'VALID':
        paddings = tf.constant([[0, 0], [0, 0]])
    if padding == 'SAME':
        ih, iw = tf.shape(x)[2:]
        if ih % strides[0] == 0:
            ph = max(ksize[0] - strides[0], 0)
        else:
            ph = max(ksize[0] - (ih % strides[0]), 0)
        if iw % strides[1] == 0:
            pw = max(ksize[1] - strides[1], 0)
        else:
            pw = max(ksize[1] - (iw % strides[1]), 0)
        pl = pw // 2
        pr = pw - pl
        pt = ph // 2
        pb = ph - pt
        paddings = tf.constant([[pt, pb], [pl, pr]])
    return paddings


# @tf.function
# def op_flatten2D(X, pool_h, pool_w, c, out_h, out_w, paddings, stride=1, batch=64):
#     X_padded = tf.pad(X, [[0, 0], paddings[0], paddings[1], [0, 0]])
#
#     windows = tf.TensorArray(tf.float32, size=out_h * out_w, dynamic_size=False, clear_after_read=False)
#     i = 0
#     for y in range(out_h):
#         for x in range(out_w):
#             window = tf.slice(X_padded, [0, y * stride, x * stride, 0], [-1, pool_h, pool_w, -1])
#             windows = windows.write(i, window)
#             i += 1
#
#     stacked = windows.stack()  # shape : [out_h, out_w, n, filter_h, filter_w, c]
#     output = tf.reshape(stacked, [-1, c * pool_w * pool_h])
#     return output
#
#
# def op_entr_pool(value, ksize, strides, paddings, batch_size, out_h, out_w, mode='high',
#                  data_format="NHWC", name=None):
#     inputs = value
#     pool_h = ksize[0]
#     pool_w = ksize[1]
#     stride = strides[0]
#     # print(f"pool h {pool_h} pool w {pool_w}") # pool h 1 pool w 2
#
#     n = batch_size
#     c = tf.shape(value)[3]
#     # print(f"inputs shape {inputs.shape}") # (64, 11, 28, 256)
#
#     X_flat = op_flatten2D(inputs, pool_h, pool_w, c, out_h, out_w, paddings, stride, batch=n)
#     # print(f"X_flat shape {X_flat.shape}") #  (4480, 512)
#
#     nrows = tf.shape(X_flat)[0]
#     ncols = tf.shape(X_flat)[1]
#     size = nrows * ncols
#
#     X_flat = tf.reshape(X_flat, [-1])
#     y, idx, counts = tf.unique_with_counts(X_flat)
#
#     elements_idx = tf.meshgrid(tf.range(0, size))
#     indices_for_y = tf.gather(idx, elements_idx)
#
#     x_counts = tf.cast(tf.gather(counts, indices_for_y), tf.float32)
#     one = tf.constant(1.0, dtype=tf.float32)
#     size = tf.cast(size, dtype=tf.float32)
#     one_div_size = tf.divide(one, size)
#     x_probs = tf.scalar_mul(one_div_size, x_counts)
#     x_entropies = tf.map_fn(lambda p: -p * tf.math.log(p), x_probs)
#     x_entropies = tf.reshape(x_entropies, [out_h, out_w, -1, pool_h * pool_w, c])
#     X_flat = tf.reshape(X_flat, [out_h, out_w, -1, pool_h * pool_w, c])
#
#     if mode is 'high':
#         max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)
#     elif mode is 'low':
#         max_entropy_indices = tf.argmax(x_entropies, axis=3, output_type=tf.int32)
#     else:
#         max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)
#
#     grid = tf.meshgrid(tf.range(0, x_entropies.shape[0]), tf.range(0, x_entropies.shape[1]),
#                        tf.range(0, x_entropies.shape[2]), tf.range(0, x_entropies.shape[4]), indexing='ij')
#
#     coords = tf.stack(grid + [max_entropy_indices], axis=-1)
#     c = tf.unstack(coords, axis=-1)
#     coords = tf.stack([c[0], c[1], c[2], c[4], c[3]], axis=-1)
#     entropy_pool = tf.gather_nd(X_flat, coords)
#     # print(f"finished pooling {entropy_pool.get_shape()}")
#     return tf.transpose(entropy_pool, [2, 0, 1, 3])


@tf.function
def op_flatten1D(X, pool_h, c, out_h, paddings, stride=1, batch=64):
    # print(f"op_flatten1D paddings shape {paddings.shape}")
    # print(f"op_flatten1D X shape {X.shape}")
    # X_padded = X
    X_padded = tf.pad(X, [[0, 0], paddings[0], paddings[1], [0, 0]])
    # print(f"op_flatten1D X_padded shape {X_padded.shape}")
    windows = tf.TensorArray(tf.float32, size=out_h, dynamic_size=False, clear_after_read=False)
    i = 0
    for y in range(out_h):
        window = tf.slice(X_padded, [0, y * stride, 0, 0], [-1, pool_h, 1, -1])
        windows = windows.write(i, window)
        # print(f"op_flatten1D sliced shape {window.shape}")
        i += 1

    stacked = windows.stack()  # shape : [out_h, out_w, n, filter_h, filter_w, c]
    # print(f"op_flatten1D stacked shape {stacked.shape}")
    output = tf.reshape(stacked, [-1, c * pool_h])
    # print(f"op_flatten1D output shape {output.shape}")
    return output


def op_entr_pool1D(value, ksize, strides, paddings, batch_size, out_h, out_w, mode='high',
                   data_format="NHWC", name=None):
    inputs = value
    pool_h = ksize[0]
    pool_w = ksize[1]
    # print(f"strides = {strides}")
    # print(f"batch_size = {batch_size}")
    # print(f"pool h = {pool_h} pool w = {pool_w}")  # pool h 1 pool w 3
    stride = strides[0]

    c = tf.shape(value)[3]
    # print("op entr pool 1d")
    # print(f"inputs shape {inputs.shape}")  # (None, 250, 1, 128) (64, 11, 28, 256)
    X_flat = op_flatten1D(inputs, pool_h, c, out_h, paddings, stride, batch=batch_size)
    # print(f"X_flat shape {X_flat.shape}")  # (4480, 512) (2656, 384)
    nrows = tf.shape(X_flat)[0]
    ncols = tf.shape(X_flat)[1]
    size = nrows * ncols

    X_flat = tf.reshape(X_flat, [-1])
    y, idx, counts = tf.unique_with_counts(X_flat)

    elements_idx = tf.meshgrid(tf.range(0, size))
    indices_for_y = tf.gather(idx, elements_idx)

    x_counts = tf.cast(tf.gather(counts, indices_for_y), tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    size = tf.cast(size, dtype=tf.float32)
    one_div_size = tf.divide(one, size)
    x_probs = tf.scalar_mul(one_div_size, x_counts)
    x_entropies = tf.map_fn(lambda p: -p * tf.math.log(p), x_probs)
    x_entropies = tf.reshape(x_entropies, [out_h, out_w, -1, pool_h * pool_w, c])
    X_flat = tf.reshape(X_flat, [out_h, out_w, -1, pool_h * pool_w, c])

    if mode is 'high':
        max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)
    elif mode is 'low':
        max_entropy_indices = tf.argmax(x_entropies, axis=3, output_type=tf.int32)
    else:
        max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)

    grid = tf.meshgrid(tf.range(0, x_entropies.shape[0]), tf.range(0, x_entropies.shape[1]),
                       tf.range(0, x_entropies.shape[2]), tf.range(0, x_entropies.shape[4]), indexing='ij')

    coords = tf.stack(grid + [max_entropy_indices], axis=-1)
    c = tf.unstack(coords, axis=-1)
    coords = tf.stack([c[0], c[1], c[2], c[4], c[3]], axis=-1)
    entropy_pool = tf.gather_nd(X_flat, coords)
    # print(f"finished pooling {entropy_pool.get_shape()}")
    transposed = tf.transpose(entropy_pool, [2, 0, 1, 3])
    # print(f"finished pooling transposed {transposed.get_shape()}")
    # print(f"DImensions of transposed {[transposed.shape[0], transposed.shape[1], transposed.shape[2], transposed.shape[3]]}")
    # output = tf.squeeze(transposed)

    # output = tf.reshape(transposed, [transposed.shape[0],
    #                                  transposed.shape[1],
    #                                  transposed.shape[3]])
    return transposed


class EntropyPooling2D(Pooling2D):
    def __init__(self, pool_size, strides=None, padding='valid', data_format=None,
                 mode='high', name=None, **kwargs):
        super(EntropyPooling2D, self).__init__(
            self.entr_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, data_format=data_format, **kwargs)
        self.mode = mode

    def build(self, input_shape):
        self.built_input_shape = input_shape
        self.built_output_shape = self.compute_output_shape(input_shape)
        super(EntropyPooling2D, self).build(input_shape)

    def entr_pool(self, value, ksize, strides, padding, data_format='NHWC',
                  name=None, input=None):
        with ops.name_scope(name, "EntrPool", [value]) as name:
            value = deprecation.deprecated_argument_lookup(
                "input", input, "value", value)

            if data_format is None:
                data_format = "NHWC"
            channel_index = 1 if data_format.startswith("NC") else 3

            ksize = _get_sequence(ksize, 2, channel_index, "ksize")
            strides = _get_sequence(strides, 2, channel_index, "strides")
            paddings = op_pad(padding, value, ksize, strides)

            _, out_h, out_w, _ = self.built_output_shape

            return self.op_entr_pool(value,
                                ksize=ksize,
                                strides=strides,
                                paddings=paddings,
                                batch_size=self.built_input_shape[0],
                                out_h=out_h,
                                out_w=out_w,
                                mode='high',
                                data_format=data_format,
                                name=name)

    def op_flatten2D(self, X, pool_h, pool_w, c, out_h, out_w, paddings, stride=1, batch=64):
        X_padded = tf.pad(X, [[0, 0], paddings[0], paddings[1], [0, 0]])
        windows = tf.TensorArray(tf.float32, size=out_h * out_w, dynamic_size=False, clear_after_read=False)
        i = 0
        for y in range(out_h):
            for x in range(out_w):
                window = tf.slice(X_padded, [0, y * stride, x * stride, 0], [-1, pool_h, pool_w, -1])
                windows = windows.write(i, window)
                i += 1

        stacked = windows.stack()  # shape : [out_h, out_w, n, filter_h, filter_w, c]
        output = tf.reshape(stacked, [-1, c * pool_w * pool_h])
        return output

    def op_entr_pool(self, value, ksize, strides, paddings, batch_size, out_h, out_w, mode='high',
                     data_format="NHWC", name=None):
        inputs = value
        pool_h = ksize[1]
        pool_w = ksize[2]
        stride = strides[1]
        # print(f"pool h {pool_h} pool w {pool_w}") # pool h 1 pool w 2

        n = batch_size
        c = tf.shape(value)[3]
        X_flat = self.op_flatten2D(inputs, pool_h, pool_w, c, out_h, out_w, paddings, stride, batch=n)

        nrows = tf.shape(X_flat)[0]
        ncols = tf.shape(X_flat)[1]
        size = nrows * ncols

        X_flat = tf.reshape(X_flat, [-1])
        y, idx, counts = tf.unique_with_counts(X_flat)

        elements_idx = tf.meshgrid(tf.range(0, size))
        indices_for_y = tf.gather(idx, elements_idx)

        x_counts = tf.cast(tf.gather(counts, indices_for_y), tf.float32)
        one = tf.constant(1.0, dtype=tf.float32)
        size = tf.cast(size, dtype=tf.float32)
        one_div_size = tf.divide(one, size)
        x_probs = tf.scalar_mul(one_div_size, x_counts)
        x_entropies = tf.map_fn(lambda p: -p * tf.math.log(p), x_probs)
        x_entropies = tf.reshape(x_entropies, [out_h, out_w, -1, pool_h * pool_w, c])
        X_flat = tf.reshape(X_flat, [out_h, out_w, -1, pool_h * pool_w, c])

        if mode is 'high':
            max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)
        elif mode is 'low':
            max_entropy_indices = tf.argmax(x_entropies, axis=3, output_type=tf.int32)
        else:
            max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)

        grid = tf.meshgrid(tf.range(0, x_entropies.shape[0]), tf.range(0, x_entropies.shape[1]),
                           tf.range(0, x_entropies.shape[2]), tf.range(0, x_entropies.shape[4]), indexing='ij')

        coords = tf.stack(grid + [max_entropy_indices], axis=-1)
        c = tf.unstack(coords, axis=-1)
        coords = tf.stack([c[0], c[1], c[2], c[4], c[3]], axis=-1)
        entropy_pool = tf.gather_nd(X_flat, coords)
        return tf.transpose(entropy_pool, [2, 0, 1, 3])


class EntropyPooling1D(Pooling1D):
    def __init__(self, pool_size, strides=None, padding='valid', data_format=None,
                 mode='high', name=None, **kwargs):
        super(EntropyPooling1D, self).__init__(
            self.entr_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, data_format=data_format, **kwargs)
        self.mode = mode

    def build(self, input_shape):
        self.built_input_shape = input_shape
        # print(f"self.built_input_shape {self.built_input_shape}")
        self.built_output_shape = self.compute_output_shape(input_shape)
        super(EntropyPooling1D, self).build(input_shape)

    # def call(self, inputs):
    #     pad_axis = 2 if self.data_format == 'channels_last' else 3
    #     inputs = array_ops.expand_dims(inputs, pad_axis)
    #     outputs = self.entr_pool(
    #         inputs,
    #         self.pool_size + (1,),
    #         strides=self.strides + (1,),
    #         padding=self.padding,
    #         data_format=self.data_format)
    #     # return array_ops.squeeze(outputs, pad_axis)
    #     return outputs

    def entr_pool(self, x, pool_size, strides, padding='valid', data_format=None,
                  name=None, input=None):
        # data_format should be 'channels_last' by default.
        if data_format is None:
            data_format = image_data_format()  # returns 'channels_last'
            # print(f"data format {data_format}")
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ' + str(data_format))
        if len(pool_size) != 2:
            raise ValueError('`pool_size` must be a tuple of 2 integers.')
        if len(strides) != 2:
            raise ValueError('`strides` must be a tuple of 2 integers.')

        # print(f"entr_pool x {x.get_shape()}")
        # x, tf_data_format = _preprocess_conv2d_input(x, data_format)
        # print(f"entr_pool after preprocess x {x.get_shape()}")
        padding = _preprocess_padding(padding)

        # print(f"self pool size {self.pool_size}")
        # print(f"pool size {pool_size}")
        # print(f"self strides size {self.strides}")
        # print(f"strides {strides}")
        #
        # print(f"pool size {pool_size}")
        # print(f"strides {strides}")
        #
        # print(f"self.built_output_shape {self.built_output_shape}")  # (None, 250, 128)
        _, out_h, _ = self.built_output_shape  # (None, 83, 128)
        out_w = 1
        paddings = op_pad(padding, x, pool_size, strides)

        # print(f"padding {paddings.shape}")
        # print(f"padding {paddings}")
        # print(f"self.padding {self.padding}")

        x = op_entr_pool1D(x,
                           ksize=pool_size,
                           strides=strides,
                           paddings=paddings,
                           batch_size=self.built_input_shape[0],
                           out_h=out_h,
                           out_w=out_w,
                           mode='high',
                           data_format=data_format,
                           name=name)
        # if data_format == 'channels_first' and tf_data_format == 'NHWC':
        #     x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return x


class EntropyPoolLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EntropyPoolLayer, self).__init__(**kwargs)
        self.n = 0
        self.h = 0
        self.w = 0
        self.c = 0

    def get_config(self):
        config = {
            'n': self.n,
            'h': self.h,
            'w': self.w,
            'c': self.c
        }
        base_config = super(EntropyPoolLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.n = input_shape[0]  # tf.constant(64, dtype=tf.int32)
        self.h = input_shape[1]
        self.w = input_shape[2]
        self.c = input_shape[3]

        self.batch_size = input_shape[0]
        # self.x_shape = self.x_shape[0].assign(64)
        # print(f"batch_size {self.batch_size}")
        super(EntropyPoolLayer, self).build(input_shape)

    def call(self, inputs, pool_h=2, pool_w=2, padding=0, stride=2, mode='high', batch_size=64, **kwargs):
        # inputs = tf.reshape(inputs, self.x_shape)
        # n, h, w, c = [d for d in inputs.shape]
        n = self.n  # tf.shape(inputs)[0]
        h = self.h  # tf.shape(inputs)[1]
        w = self.w  # tf.shape(inputs)[2]
        c = self.c  # tf.shape(inputs)[3]
        # print(f"n={n}, h={h}, w={w}, c={c}") # n = None, h = 98, w = 257, c = 32
        out_h = (h + 2 * padding - pool_h) // stride + 1
        out_w = (w + 2 * padding - pool_w) // stride + 1
        #pool_h=2, pool_w=2, c=256, out_h=5, out_w=14, stride=2, padding=0, n = 64

        X_flat = self.flatten2(inputs, pool_h, pool_w, c, out_h, out_w, stride, padding, batch=n)
        # nrows, ncols = [d for d in X_flat.get_shape()]
        nrows = tf.shape(X_flat)[0]
        ncols = tf.shape(X_flat)[1]
        size = nrows * ncols

        X_flat = tf.reshape(X_flat, [-1])
        y, idx, counts = tf.unique_with_counts(X_flat)

        elements_idx = tf.meshgrid(tf.range(0, size))
        indices_for_y = tf.gather(idx, elements_idx)

        x_counts = tf.cast(tf.gather(counts, indices_for_y), tf.float32)
        one = tf.constant(1.0, dtype=tf.float32)
        # print(size)
        # print(type(size))
        size = tf.cast(size, dtype=tf.float32)
        # size = tf.float32(size)
        one_div_size = tf.divide(one, size)
        x_probs = tf.scalar_mul(one_div_size, x_counts)
        x_entropies = tf.map_fn(lambda p: -p * tf.math.log(p), x_probs)
        # print(f"x_entropies {x_entropies}")
        # print(f"out_h : {out_h}, out_w {out_w}, n {n}, pool_h {pool_h}, pool_w {pool_w}, c {c}")

        x_entropies = tf.reshape(x_entropies, [out_h, out_w, -1, pool_h * pool_w, c])

        X_flat = tf.reshape(X_flat, [out_h, out_w, -1, pool_h * pool_w, c])

        if mode is 'high':
            max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)
        elif mode is 'low':
            max_entropy_indices = tf.argmax(x_entropies, axis=3, output_type=tf.int32)
        else:
            max_entropy_indices = tf.argmin(x_entropies, axis=3, output_type=tf.int32)

        grid = tf.meshgrid(tf.range(0, x_entropies.shape[0]), tf.range(0, x_entropies.shape[1]),
                           tf.range(0, x_entropies.shape[2]), tf.range(0, x_entropies.shape[4]), indexing='ij')

        coords = tf.stack(grid + [max_entropy_indices], axis=-1)
        c = tf.unstack(coords, axis=-1)
        coords = tf.stack([c[0], c[1], c[2], c[4], c[3]], axis=-1)
        entropy_pool = tf.gather_nd(X_flat, coords)
        # print(f"finished pooling {entropy_pool.get_shape()}")
        return tf.transpose(entropy_pool, [2, 0, 1, 3])

    def flatten2(self, X, pool_h, pool_w, c, out_h, out_w, stride=1, padding=0, batch=64):
        # batch = tf.shape(X)[0]
        print(f"padding {padding}")
        X_padded = tf.pad(X, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        print(f"X_padded shape: {X_padded.get_shape()}")  # X_padded    shape: (None, 98, 257, 32)
        # windows = []
        windows = tf.TensorArray(tf.float32, size=out_h * out_w, dynamic_size=False, clear_after_read=False)
        i = 0
        for y in range(out_h):
            for x in range(out_w):
                window = tf.slice(X_padded, [0, y * stride, x * stride, 0], [-1, pool_h, pool_w, -1])
                windows = windows.write(i, window)
                i += 1
                # print(f"Sliced window shape: {window}")
        # Sliced window shape: Tensor("keras_model/entropy_pool_layer/Slice_6270:0", shape=(None, 2, 2, 32), dtype=float32)

        # stacked = tf.stack(windows)  # shape : [out_h, out_w, n, filter_h, filter_w, c]
        # print(f"stacked: {windows.get_shape()}")
        stacked = windows.stack()  # shape : [out_h, out_w, n, filter_h, filter_w, c]
        # print(f"stacked: {stacked.get_shape()}")
        # output=tf.reshape(stacked, [-1, window_c * window_w * window_h])

        # output = tf.reshape(stacked, [batch * tf.shape(stacked)[0], c * pool_w * pool_h])
        output = tf.reshape(stacked, [-1, c * pool_w * pool_h])
        # print(f"output: {output.get_shape()}")
        return output


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters, conv_num=3, activation="relu", pool_type=None):
        super(ResidualBlock, self).__init__(name='')
        self.conv_num = conv_num
        self.filters = filters
        self.activation = activation
        self.pool_type = pool_type

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv1D(self.filters, 1, padding="same")

        self.convolutions = []
        self.activations = []

        for i in range(self.conv_num - 1):
            self.convolutions.append(tf.keras.layers.Conv1D(self.filters, 3, padding="same"))
            self.activations.append(tf.keras.layers.Activation(self.activation))

        self.conv3 = tf.keras.layers.Conv1D(self.filters, 3, padding="same")
        self.add_layers = tf.keras.layers.Add()
        self.last_activation = tf.keras.layers.Activation(self.activation)
        if self.pool_type == PoolingLayerFactory.ENTR:
            self.pool = EntropyPooling1D(pool_size=2, strides=2)
        elif self.pool_type == PoolingLayerFactory.AVG:
            self.pool = AveragePooling1D(pool_size=2, strides=2)
        else:
            self.pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        super(ResidualBlock, self).build(input_shape)

    def call(self, input_tensor, training=False):
        s = self.conv1(input_tensor)
        x = input_tensor
        for conv, activation in zip(self.convolutions, self.activations):
            x = conv(x)
            x = activation(x)
        x = self.conv3(x)
        x = self.add_layers([x, s])
        x = self.last_activation(x)
        x = self.pool(x)
        return x

class InformationPool(tf.keras.layers.Layer):

    def __init__(self,filters,kernel,stride):
        super(InformationPool, self).__init__()
        self.conv2d=tf.keras.layers.Conv2D(filters,kernel,stride, activation='relu',padding='same')
        self.conv2d2=tf.keras.layers.Conv2D(filters,kernel,stride, activation='sigmoid',padding='same',trainable=False)
        self.mu=tf.Variable(initial_value=0,dtype='float32',trainable=True)
        self.sigma=tf.Variable(initial_value=1,dtype='float32',trainable=True)
        #self.keep_prob = tf.placeholder(tf.float32, shape=[]) 
       # self.initial_keep_prob = tf.placeholder(tf.float32, shape=[]) 
      #  self.sigma0 = tf.placeholder(tf.float32, shape=[])
    
   # def build(self):
    #    '''Creates the placeholders for this model'''
     #   self.keep_prob = tf.placeholder(tf.float32, shape=[]) 
      #  self.initial_keep_prob = tf.placeholder(tf.float32, shape=[]) 
       # self.sigma0 = tf.placeholder(tf.float32, shape=[])
        
        

   ## @ex.capture
    def conv(self, inputs, num_outputs, activations='relu', kernel_size=3, stride=1, scope=None):
        '''Creates a convolutional layer with default arguments'''
        if activations == 'relu':
            activation_fn = tf.nn.relu
        elif activations == 'softplus':
            activation_fn = tf.nn.softplus
        else:
            raise ValueError("Invalid activation function.")
        return self.conv2d( inputs = inputs
            #num_outputs = num_outputs,
            #kernel_size = kernel_size,
            #stride = stride,
            #padding = 'SAME',
            #activation = activation_fn,
            #normalizer = BatchNormalization,
            #scope=scope )
            )

  ##  @ex.capture
    def information_pool(self, inputs, max_alpha=1, alpha_mode='information', lognorm_prior=True, num_outputs=None, stride=2, scope=None):
          
        # Creates the output convolutional layer
        network = self.conv(inputs, num_outputs=int(num_outputs), stride=stride)
        with tf.compat.v1.variable_scope(scope,'information_dropout'):
            # Computes the noise parameter alpha for the output
            #K.print_tensor(tf.constant(2))
            alpha = self.conv2d2(inputs
                                 #num_outputs=int(num_outputs),
                                 #kernel_size=3,
               # stride=stride,
               # activation=tf.sigmoid,
                #scope='alpha')
                )
            # Rescale alpha in the allowed range and add a small value for numerical stability
            alpha = 0.001 + max_alpha * alpha
            # Computes the KL divergence using either log-uniform or log-normal prior
            if not lognorm_prior:
                kl = - tf.math.log(alpha/(max_alpha + 0.001))
            else:
                #mu1 = tf.compat.v1.get_variable('mu1', [], initializer=tf.constant_initializer(0.))
                #sigma1 = tf.compat.v1.get_variable('sigma1', [], initializer=tf.constant_initializer(1.))
                
                kl = KL_div2(tf.math.log(tf.maximum(network,1e-4)), alpha, self.mu, self.sigma)
            tf.compat.v1.add_to_collection('kl_terms', kl)
        # Samples the noise with the given parameter
        e = sample_lognormal(mean=tf.zeros_like(network), sigma = alpha )#sigma0 = self.sigma0)
        # Returns the noisy output of the dropout
        return network * e

    ##@ex.capture
    def conv_dropout(self, inputs, num_outputs, dropout):
        if dropout == 'information':
            network = self.information_pool(inputs, num_outputs=num_outputs)
        elif dropout == 'binary':
            network = self.conv(inputs, num_outputs, stride=2)
            network = tf.nn.dropout(network, self.keep_prob)
        elif dropout == 'none':
            network = self.conv2d(inputs)
        else:
            raise ValueError("Invalid dropout value")
        return network
    
    def call(self,inputs,dropout='information'):
      return  self.conv_dropout(inputs,inputs.shape[0],dropout)

information_pool_custom_loss_basic_keras=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
def information_pool_custom_loss(y_true, y_pred):
            
     loss=information_pool_custom_loss_basic_keras(y_true,y_pred)
     kl_terms = [ batch_average(kl) for kl in tf.compat.v1.get_collection('kl_terms') ]
     kl_terms=tf.math.add_n(kl_terms)/(257*98*32*2)
     loss=loss + 0.5*kl_terms
     

        
     return loss
 

class PoolingLayerFactory():
    MAX = "MAX"
    AVG = "AVG"
    ENTR = "ENTR"

    @staticmethod
    def create_pooling_layers(types_of_poolings: List[str], ksizes: List[Tuple]) -> List:
        pools = []
        for p, k in zip(types_of_poolings, ksizes):
            if p == PoolingLayerFactory.MAX:
                pool = MaxPooling2D(pool_size=k)
            elif p == PoolingLayerFactory.AVG:
                pool = AveragePooling2D(pool_size=k)
            elif p == PoolingLayerFactory.ENTR:
                pool = EntropyPooling2D(pool_size=k)
                # pool = EntropyPoolLayer()
            else:
                pool = MaxPooling2D(pool_size=k)
            pools.append(pool)
        return pools
