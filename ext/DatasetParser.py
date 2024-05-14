import pathlib
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from ext.ImageExt import negative


def get_image_num(image_name):
    return int(image_name.split('_')[0]) - 1


def parseDataSet(path):
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                img_name = str(os.path.join(dirname, filename))
                img_num = get_image_num(filename)

                img = cv2.imread(img_name)[:, :, :]
                img = cv2.resize(img, (28, 28))
                img = negative(img)

                if "train" in dirname:
                    y_train.append(img_num)
                    x_train.append(img[0])
                elif "test" in dirname:
                    y_test.append(img_num)
                    x_test.append(img[0])
                elif "val" in dirname:
                    y_val.append(img_num)
                    x_val.append(img[0])
            except:
                continue

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_val), np.array(y_val)


# test
#
# path = "/Users/imalinowski/PycharmProjects/NumsML/roman_nums"
#
# x_train, y_train, x_test, y_test, x_val, y_val = parseDataSet(path + '/dataset')
#
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)
#
# for i in range(200, 210):
#     print(y_train[i])
#     plt.imshow(x_train[i], cmap='gray')
#     plt.show()
