import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ext.ImageExt import preprocess_img

path = 'arabic_nums/digits/'

model = tf.keras.models.load_model('model.keras')


def get_image_name(image_num):
    return path + 'digit-' + str(image_num) + '.png'


image_number = 0
while os.path.isfile(get_image_name(image_number)):
    image_name = get_image_name(image_number)
    try:
        img = cv2.imread(image_name)[:, :, :]
        img = cv2.resize(img, (28, 28))
        print(img.shape)
        # img = np.invert(np.array([img]))
        img = preprocess_img(img)
        print(img.shape)
        prediction = model.predict(img)
        print(f"This digit {image_name} is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.get_cmap('gray'))
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
