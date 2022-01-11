from keras import backend as K
from keras.layers import (Activation, Conv2D, Dense, DepthwiseConv2D, Dropout,
                          GlobalAveragePooling2D)
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Model


# 卷积标准化+激活
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    # 深度可分离卷积，对输入的特征层进行特征提取
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    # 标准化和激活函数
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # 利用一个1*1的卷积进行通道数的调整
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)

    # 进行标准化和激活
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


def MobileNet(inputs, embedding_size=128, dropout_keep_prob=0.4, alpha=1.0, depth_multiplier=1):
    # 160,160,3 -> 80,80,32 32通道
    x = _conv_block(inputs, 32, strides=(2, 2))

    # 80,80,32 -> 80,80,64 通道调整为64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 80,80,64 -> 40,40,128 高和宽压缩，通道调整为128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 40,40,128 -> 20,20,256 高和宽压缩，通道调整为256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 20,20,256 -> 10,10,512 高和宽压缩，通道调整为512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 10,10,512 -> 5,5,1024 高和宽压缩，通道调整为1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 1024 全局平均值化
    x = GlobalAveragePooling2D()(x)

    # 防止网络过拟合，训练的时候起作用
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)

    # 全连接层到128
    # 128
    x = Dense(embedding_size, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
                           name='BatchNorm_Bottleneck')(x)

    #   创建模型
    model = Model(inputs, x, name='mobilenet')

    return model
