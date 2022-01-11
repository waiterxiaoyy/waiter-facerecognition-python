import math
import os
import random

import cv2
import tensorflow
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from PIL import Image


def triplet_loss(alpha=0.2, batch_size=32):
    def _triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:int(2 * batch_size)], y_pred[-batch_size:]

        pos_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        basic_loss = pos_dist - neg_dist + alpha

        idxs = tf.where(basic_loss > 0)
        select_loss = tf.gather_nd(basic_loss, idxs)

        loss = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        return loss

    return _triplet_loss


# ------------------------------------#
#   数据加载器
# ------------------------------------#
class FacenetDataset(tensorflow.keras.utils.Sequence):
    def __init__(self, input_shape, dataset_path, num_train, num_classes, batch_size):
        super(FacenetDataset, self).__init__()
        self.dataset_path = dataset_path
        self.num_train = num_train
        self.num_classes = num_classes
        self.batch_size = batch_size

        # ------------------------------------#
        #   输入图片的高宽和通道数
        # ------------------------------------#
        self.image_height = input_shape[0]
        self.image_width = input_shape[1]
        self.channel = input_shape[2]

        # ------------------------------------#
        #   路径和标签
        # ------------------------------------#
        self.paths = []
        self.labels = []

        self.load_dataset()

    def __len__(self):
        return math.ceil(self.num_train / float(self.batch_size))

    def __getitem__(self, index):
        # ------------------------------------#
        #   创建全为零的矩阵
        # ------------------------------------#
        images = np.zeros((self.batch_size, 3, self.image_height, self.image_width, self.channel))
        labels = np.zeros((self.batch_size, 3))

        for i in range(self.batch_size):
            # ------------------------------------#
            #   先获得两张同一个人的人脸
            #   用来作为anchor和positive
            # ------------------------------------#
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]
            while len(selected_path) < 2:
                c = random.randint(0, self.num_classes - 1)
                selected_path = self.paths[self.labels[:] == c]

            # ------------------------------------#
            #   随机选择两张
            # ------------------------------------#
            image_indexes = np.random.choice(range(0, len(selected_path)), 2)
            # ------------------------------------#
            #   打开图片并放入矩阵
            # ------------------------------------#
            image = Image.open(selected_path[image_indexes[0]])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64) / 255
            if self.channel == 1:
                images[i, 0, :, :, 0] = image
            else:
                images[i, 0, :, :, :] = image
            labels[i, 0] = c

            image = Image.open(selected_path[image_indexes[1]])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64) / 255
            if self.channel == 1:
                images[i, 1, :, :, 0] = image
            else:
                images[i, 1, :, :, :] = image
            labels[i, 1] = c

            # ------------------------------#
            #   取出另外一个人的人脸
            # ------------------------------#
            different_c = list(range(self.num_classes))
            different_c.pop(c)

            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = different_c[different_c_index[0]]
            selected_path = self.paths[self.labels == current_c]
            while len(selected_path) < 1:
                different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
                current_c = different_c[different_c_index[0]]
                selected_path = self.paths[self.labels == current_c]

            # ------------------------------#
            #   随机选择一张
            # ------------------------------#
            image_indexes = np.random.choice(range(0, len(selected_path)), 1)
            image = Image.open(selected_path[image_indexes[0]])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64) / 255
            if self.channel == 1:
                images[i, 2, :, :, 0] = image
            else:
                images[i, 2, :, :, :] = image
            labels[i, 2] = current_c

        # --------------------------------------------------------------#
        #   假设batch为32
        #   0,32,64   属于一个组合 0和32是同一个人，0和64不是同一个人
        # --------------------------------------------------------------#
        images1 = np.array(images)[:, 0, :, :, :]
        images2 = np.array(images)[:, 1, :, :, :]
        images3 = np.array(images)[:, 2, :, :, :]
        images = np.concatenate([images1, images2, images3], 0)

        labels1 = np.array(labels)[:, 0]
        labels2 = np.array(labels)[:, 1]
        labels3 = np.array(labels)[:, 2]
        labels = np.concatenate([labels1, labels2, labels3], 0)

        labels = np_utils.to_categorical(np.array(labels), num_classes=self.num_classes)

        return images, {'Embedding': np.zeros_like(labels), 'Softmax': labels}

    def load_dataset(self):
        for path in self.dataset_path:
            path_split = path.split(";")
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[0]))
        self.paths = np.array(self.paths, dtype=np.object)
        self.labels = np.array(self.labels)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, jitter=.1, hue=.1, sat=1.3, val=1.3, flip_signal=True):
        image = image.convert("RGB")

        h, w = input_shape
        rand_jit1 = self.rand(1 - jitter, 1 + jitter)
        rand_jit2 = self.rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = self.rand(0.85, 1.15)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        flip = self.rand() < .5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))

        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-10, 10)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        return image_data


class LFWDataset():
    def __init__(self, dir, pairs_path, input_shape, batch_size):
        super(LFWDataset, self).__init__()
        self.input_shape = input_shape
        self.pairs_path = pairs_path
        self.batch_size = batch_size
        self.validation_images = self.get_lfw_paths(dir)

    def generate(self):
        imgs1 = []
        imgs2 = []
        issames = []
        for annotation_line in self.validation_images:
            (path_1, path_2, issame) = annotation_line
            img1, img2 = Image.open(path_1), Image.open(path_2)
            img1 = self.letterbox_image(img1, [self.input_shape[1], self.input_shape[0]])
            img2 = self.letterbox_image(img2, [self.input_shape[1], self.input_shape[0]])

            img1, img2 = np.array(img1) / 255., np.array(img2) / 255.

            imgs1.append(img1)
            imgs2.append(img2)
            issames.append(issame)
            if len(imgs1) == self.batch_size:
                imgs1 = np.array(imgs1)
                imgs2 = np.array(imgs2)
                issames = np.array(issames)
                yield imgs1, imgs2, issames
                imgs1 = []
                imgs2 = []
                issames = []

        imgs1 = np.array(imgs1)
        imgs2 = np.array(imgs2)
        issames = np.array(issames)
        yield imgs1, imgs2, issames

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

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self, lfw_dir, file_ext="jpg"):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list
