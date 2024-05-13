import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

img = x_train[0]
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()
