import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pathlib
import cv2

from ext.ImageExt import preprocess_img


def parseDataSet(path):
    x_train, y_train = [], []

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                img_name = str(os.path.join(dirname, filename))

                img = cv2.imread(img_name)[:, :, :]
                img = cv2.resize(img, (28, 28))
                img = preprocess_img(img)
                x_train.append(img[0])

                y_train.append(int(filename.split('_')[0]) - 1)
            except:
                continue

    return np.array(x_train), np.array(y_train)
