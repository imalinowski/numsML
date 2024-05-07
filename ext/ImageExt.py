import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as ski


def blur(img):  # [ 1, 28, 28 ]
    return cv2.GaussianBlur(img, (5, 5), 0.7)


def rgb2gray(rgb):  # [ 1, 28, 28 ]
    gray = rgb[:, :, 0]
    height, width = gray.shape
    for i in range(0, height):
        for j in range(0, width):
            gray[i, j] = np.dot(rgb[i, j], [0.2989, 0.5870, 0.1140])
            # print(str(gray[i, j]) + " ", end="")
        # print("\n")
    return np.array([gray])


def negative(img):  # [ 1, 28, 28 ]
    img_neg = rgb2gray(img)
    height, width, _ = img.shape
    for i in range(0, height):
        for j in range(0, width):
            img_neg[0, i, j] = 254 - img_neg[0, i, j]
            # print(str(img_neg[i, j]) + " ", end="")
        # print("\n")
    return img_neg


def preprocess_img(img):
    blured = blur(img)
    negatived = negative(blured)
    return negatived


# test
# image_name = 'digits/digit-' + str(5) + '.png'
# img = cv2.imread(image_name)[:, :, :]
# img = preprocess_img(img)
# print(img.shape)
# plt.imshow(img[0], cmap=plt.get_cmap('gray'))
# # plt.imshow(img[0], cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
# plt.show()
