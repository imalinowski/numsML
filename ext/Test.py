import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ext.ImageExt import preprocess_img


def get_image_name(dirname, filename):
    return str(os.path.join(dirname, filename))


def predict_number(model, img_name):
    img = cv2.imread(img_name)[:, :, :]
    img = cv2.resize(img, (28, 28))
    img = preprocess_img(img)

    # show img
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))
    plt.show()

    return model.predict(img)


def get_img_num(image_name):
    return int(image_name.split('-')[1].split('.')[0])


def testModel(test_data_path, model):
    image_number = 0

    correct_predictions = 0
    loss = 0

    for dirname, _, filenames in os.walk(test_data_path):
        for filename in filenames:
            try:
                image_name = get_image_name(dirname, filename)
                prediction = predict_number(model, image_name)
                predicted_num = np.argmax(prediction) + 1
                print(f"This digit {image_name} is probably a {predicted_num}")

                if predicted_num == get_img_num(image_name):
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
