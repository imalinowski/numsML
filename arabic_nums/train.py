import tensorflow as tf
from arabic_nums.model import create_model
from ext.ChartsExt import draw_charts

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = create_model()

# compile and fit
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(x_train, y_train, epochs=10)

# evaluate
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

model.save('model.keras')

draw_charts(history)

# todo римский датасет с цифрами
# todo подстроить тестовые данные
# todo acc loss автоматизация

# done сверточная сеть
# done добавить валидацию test
# done добавить графики
# done идея ч/б или наоборот
