from typing import List, Tuple, Optional

import tensorflow as tf
from tensorflow.python.keras.layers import MaxPooling2D, AveragePooling2D, AveragePooling1D, MaxPooling1D

from components import EntropyPoolLayer, EntropyPooling2D, ResidualBlock, EntropyPooling1D, PoolingLayerFactory,InformationPooling2D,InformationPooling1D,InformationPooling1DPure


class AlexNet(tf.keras.Model):

    def __init__(self, labels, types_of_poolings: Optional[List[str]], ksizes: Optional[List[Tuple]]):
        super(AlexNet, self).__init__(self)
        self.labels = labels
        self.types_of_poolings = types_of_poolings
        self.ksizes = ksizes
        self.log_pool_info = False
        self.pool_info = []
        if not types_of_poolings:
            self.types_of_poolings = [PoolingLayerFactory.MAX,
                                      PoolingLayerFactory.MAX,
                                      PoolingLayerFactory.MAX,
                                      PoolingLayerFactory.ENTR]
        if not ksizes:
            self.ksizes = [(2, 2) for i in range(len(self.types_of_poolings))]

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(122, 257, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, ))

    def get_config(self):
        config = super(AlexNet, self).get_config()
        config.update({"labels": self.labels})
        config.update({"types_of_poolings": self.types_of_poolings})
        config.update({"ksizes": self.ksizes})
        return config

    def build(self, input_shape):
        print(f"build is called with input shape {input_shape}")
        self.normalization0 = tf.keras.layers.BatchNormalization(trainable=True, input_shape=input_shape[:1])
        self.conv1 = tf.keras.layers.Conv2D(32, 3, 1, activation='relu', padding='same', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(32, 3, 1, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu')  # 128
        self.conv4 = tf.keras.layers.Conv2D(256, 3, 1, activation='relu')  # 256
        self.conv5 = tf.keras.layers.Conv2D(128, 1, 1, activation='relu')  # 128
        self.conv6 = tf.keras.layers.Conv2D(64, 1, 1, activation='relu')  # 128

        self.normalization1 = tf.keras.layers.BatchNormalization()
        self.normalization2 = tf.keras.layers.BatchNormalization()
        self.normalization3 = tf.keras.layers.BatchNormalization()
        self.normalization4 = tf.keras.layers.BatchNormalization()
        self.normalization5 = tf.keras.layers.BatchNormalization()
        self.normalization6 = tf.keras.layers.BatchNormalization()
        self.normalization7 = tf.keras.layers.BatchNormalization()
        
        self.conv=[32,32,128,256]        
        poolings=PoolingLayerFactory.create_pooling_layers(self.types_of_poolings, self.ksizes,self.conv)
    
            
        self.pool1 = poolings[0]
        self.pool2 = poolings[1]
        self.pool3 = poolings[2]
        self.pool4 = poolings[3]
        #self.pool1=InformationPooling2(pool_size=(2,2),conv=32)
        #self.pool2=InformationPooling2(pool_size=(2,2),conv=32)
        #self.pool3=MaxPooling2D(pool_size=(2,2))
        #self.pool4=MaxPooling2D(pool_size=(2,2))
     

        self.dense1 = tf.keras.layers.Dense(self.labels.__len__(), activation=None)
        self.flatten = tf.keras.layers.Flatten()

        super(AlexNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.log_pool_info:
            self.pool_info = []

        x = self.normalization0(inputs)
        x = self.conv1(x)
        x = self.normalization1(x)

        if self.log_pool_info:
            xin = x
        x = self.pool1(x)
        if self.log_pool_info:
            self.pool_info.append((self.types_of_poolings[0], xin, x))

        x = self.conv2(x)
        x = self.normalization2(x)

        if self.log_pool_info:
            xin = x
        x = self.pool2(x)
        if self.log_pool_info:
            self.pool_info.append((self.types_of_poolings[1], xin, x))

        x = self.conv3(x)
        x = self.normalization3(x)
        if self.log_pool_info:
            xin = x
        x = self.pool3(x)
        if self.log_pool_info:
            self.pool_info.append((self.types_of_poolings[2], xin, x))

        x = self.conv4(x)
        x = self.normalization4(x)
        if self.log_pool_info:
            xin = x
        x = self.pool4(x)
        if self.log_pool_info:
            self.pool_info.append((self.types_of_poolings[3], xin, x))

        x = self.conv5(x)

        x = self.conv6(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.normalization7(x)
        
        return x


class ResidualModel(tf.keras.Model):

    def __init__(self, labels, last_pool=None, resblock_pool=None):
        super(ResidualModel, self).__init__(self)
        self.labels = labels
        self.last_pool = last_pool
        self.resblock_pool = resblock_pool

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(8000,1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, ))

    def get_config(self):
        config = super(ResidualModel, self).get_config()
        config.update({"labels": self.labels})
        return config

    def build(self, input_shape):
        print(f"build is called with input shape {input_shape}")
        multitude=5
        if self.last_pool == PoolingLayerFactory.INFO and self.resblock_pool == PoolingLayerFactory.INFO :
            multitude=6
        elif self.last_pool == PoolingLayerFactory.INFO :
            multitude=1
        else :
            multitude=5
        self.residual_block1 = ResidualBlock(16, 2,multitude=multitude)
        self.residual_block2 = ResidualBlock(32, 2,multitude=multitude)
        self.residual_block3 = ResidualBlock(64, 3,multitude=multitude)
        self.residual_block4 = ResidualBlock(128, 3,multitude=multitude)
        self.residual_block5 = ResidualBlock(128, 3,multitude=multitude)

        if self.last_pool == PoolingLayerFactory.ENTR:
            self.pool = EntropyPooling1D(pool_size=3, strides=3)
        elif self.last_pool == PoolingLayerFactory.AVG:
            self.pool = AveragePooling1D(pool_size=3, strides=3)
        elif self.last_pool == PoolingLayerFactory.MAX:
            self.pool = MaxPooling1D(pool_size=3, strides=3)
        elif self.last_pool == PoolingLayerFactory.INFOP:
            self.pool = InformationPooling1DPure(pool_size=3,conv=128)
        else:
            self.pool = InformationPooling1D(pool_size=3,conv=128,multitude_regularizer=multitude)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(self.labels.__len__(), activation="softmax", name="output")
        super(ResidualModel, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.residual_block1(inputs)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)

        # print(f"Before pooling {x.shape}")  # (None, 250, 128)
        x = self.pool(x)
        # print(f"After pooling {x.shape}")  # (None, 83, 128)  (0, 83, 0)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

