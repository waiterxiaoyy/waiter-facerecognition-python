import keras.backend as K
from keras.layers import Activation, Dense, Input, Lambda
from keras.models import Model

from nets.inception_resnetv1 import InceptionResNetV1
from nets.mobilenet import MobileNet


def facenet(input_shape, num_classes=None, backbone="mobilenet", mode="train"):
    inputs = Input(shape=input_shape)
    # --------------------------------------------#
    #   利用主干网络进行特征提取
    # --------------------------------------------#
    if backbone == "mobilenet":
        model = MobileNet(inputs, dropout_keep_prob=0.4)
    elif backbone == "inception_resnetv1":
        model = InceptionResNetV1(inputs, dropout_keep_prob=0.4)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))

    if mode == "train":
        # --------------------------------------------#
        #   训练的话利用交叉熵和triplet_loss
        #   结合一起训练
        # --------------------------------------------#
        logits = Dense(num_classes)(model.output)
        softmax = Activation("softmax", name="Softmax")(logits)

        normalize = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        combine_model = Model(inputs, [softmax, normalize])
        return combine_model
    elif mode == "predict":
        # --------------------------------------------#
        #   预测的时候只需要考虑人脸的特征向量就行了
        # --------------------------------------------#
        x = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        model = Model(inputs, x)
        return model
    else:
        raise ValueError('Unsupported mode - `{}`, Use train, predict.'.format(mode))
