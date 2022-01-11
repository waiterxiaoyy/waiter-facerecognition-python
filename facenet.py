import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nets.facenet import facenet


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
# --------------------------------------------#
class Facenet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        # --------------------------------------------------------------------------#
        "model_path": "model_data/facenet_mobilenet.h5",
        # --------------------------------------------------------------------------#
        #   输入图片的大小。
        # --------------------------------------------------------------------------#
        "input_shape": [160, 160, 3],
        # --------------------------------------------------------------------------#
        #   所使用到的主干特征提取网络
        # --------------------------------------------------------------------------#
        "backbone": "mobilenet"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Facenet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        self.model = facenet(self.input_shape, backbone=self.backbone, mode="predict")
        self.model.load_weights(self.model_path, by_name=True)
        print('{} model loaded.'.format(model_path))

    def letterbox_image(self, image, size):
        if self.input_shape[-1] == 1:
            image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # ---------------------------------------------------#
        #   对输入图片进行不失真的resize
        # ---------------------------------------------------#
        image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
        image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])
        # ---------------------------------------------------#
        #   进行图片的归一化
        # ---------------------------------------------------#
        image_1 = np.asarray(image_1).astype(np.float64) / 255
        image_2 = np.asarray(image_2).astype(np.float64) / 255
        # ---------------------------------------------------#
        #   添加上batch维度，才可以放入网络当中预测
        # ---------------------------------------------------#
        photo1 = np.expand_dims(image_1, 0)
        photo2 = np.expand_dims(image_2, 0)
        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        output1 = self.model.predict(photo1)
        output2 = self.model.predict(photo2)
        # ---------------------------------------------------#
        #   计算二者之间的距离（欧几里得距离）
        # ---------------------------------------------------#
        l1 = np.sqrt(np.sum(np.square(output1 - output2), axis=-1))

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va='bottom', fontsize=11)
        plt.show()
        return l1
