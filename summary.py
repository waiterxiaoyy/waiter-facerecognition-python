#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import os

from nets.facenet import facenet

if __name__ == "__main__":
    input_shape = [160, 160, 3]
    model = facenet(input_shape, len(os.listdir("./datasets")), backbone="mobilenet", mode="train")
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
