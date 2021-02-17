from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

# 训练参数
batch_size = 32
epochs = 200
data_augmentation = True
num_classes = 10


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D 卷积批量标准化 - 激活栈构建器

    # 参数
        inputs (tensor): 从输入图像或前一层来的输入张量
        num_filters (int): Conv2D 过滤器数量
        kernel_size (int): Conv2D 方形核维度
        strides (int): Conv2D 方形步幅维度
        activation (string): 激活函数名
        batch_normalization (bool): 是否包含批标准化
        conv_first (bool): conv-bn-activation (True) 或
            bn-activation-conv (False)

    # 返回
        x (tensor): 作为下一层输入的张量
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet 版本 1 模型构建器 [a]

    2 x (3 x 3) Conv2D-BN-ReLU 的堆栈
    最后一个 ReLU 在快捷连接之后。
    在每个阶段的开始，特征图大小由具有 strides=2 的卷积层减半（下采样），
    而滤波器的数量加倍。在每个阶段中，这些层具有相同数量的过滤器和相同的特征图尺寸。
    特征图尺寸:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    参数数量与 [a] 中表 6 接近:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # 参数
        input_shape (tensor): 输入图像张量的尺寸
        depth (int): 核心卷积层的数量
        num_classes (int): 类别数 (CIFAR10 为 10)

    # 返回
        model (Model): Keras 模型实例
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # 开始模型定义
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # 实例化残差单元的堆栈
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # 第一层但不是第一个栈
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # 线性投影残差快捷键连接，以匹配更改的 dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # 在顶层加分类器。
    # v1 不在最后一个快捷连接 ReLU 后使用 BN
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 实例化模型。
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_resnet_20(num_classes=10):
    model = resnet_v1(input_shape=(32,32,3), depth=20, num_classes=num_classes)
    # optimizer = keras.optimizers.SGD(learning_rate=0.01)
    # optimizer = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-6, decay=0.000005)
    optimizer = keras.optimizers.Adadelta(lr=1.0)
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model

def get_resnet_32(num_classes=10):
    model = resnet_v1(input_shape=(32,32,3), depth=32, num_classes=num_classes)
    # optimizer = keras.optimizers.SGD(learning_rate=0.01)
    # optimizer = keras.optimizers.Adagrad(lr=1.0, epsilon=1e-6)
    optimizer = keras.optimizers.Adadelta(lr=1.0)
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_resnet_20_uncompiled(num_classes=10):
    model = resnet_v1(input_shape=(32,32,3), depth=20, num_classes=num_classes)
    # optimizer = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-6, decay=0.000005)
    # optimizer = keras.optimizers.SGD(learning_rate=0.01)
    optimizer = keras.optimizers.Adadelta(lr=1.0)
    return model, optimizer

def get_resnet_32_uncompiled(num_classes=10):
    model = resnet_v1(input_shape=(32,32,3), depth=32, num_classes=num_classes)
    # optimizer = keras.optimizers.SGD(learning_rate=0.01)
    # optimizer = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-6, decay=0.000005)
    optimizer = keras.optimizers.Adadelta(lr=1.0)
    return model, optimizer