import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ext.ImageExt import preprocess_img

path = 'digits/'

# model = tf.keras.models.load_model('arabic_nums/best_model.keras') # acc 91.67 loss 8.33
model = tf.keras.models.load_model('model.keras')


def get_image_name(image_num):
    return path + 'digit-' + str(image_num) + '.png'


def predict_number(img_name):
    img = cv2.imread(img_name)[:, :, :]
    img = cv2.resize(img, (28, 28))
    img = preprocess_img(img)

    # show img
    #plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    #plt.show()

    return model.predict(img)


def get_img_num(image_num):
    return image_num % 10


def test():
    image_number = 0

    correct_predictions = 0
    loss = 0

    while os.path.isfile(get_image_name(image_number)):
        image_name = get_image_name(image_number)
        try:
            prediction = predict_number(image_name)
            predicted_num = np.argmax(prediction)
            print(f"This digit {image_name} is probably a {predicted_num}")

            if predicted_num == get_img_num(image_number):
                correct_predictions += 1
            else:
                loss += 1  # maybe incorrect
        except:
            print("Error!")
        finally:
            image_number += 1

    accuracy = correct_predictions / image_number
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    mse_loss = loss / image_number
    print("Loss: {:.2f}% # maybe incorrect".format(mse_loss * 100))


# test
test()
