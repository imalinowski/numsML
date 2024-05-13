import pathlib
from ext.ChartsExt import draw_charts
from arabic_nums.model import create_model
from ext.DatasetParser import parseDataSet
import tensorflow as tf

path = str(pathlib.Path().resolve())

x_train, y_train, x_test, y_test, x_val, y_val = parseDataSet(path + '/dataset')

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_val = tf.keras.utils.normalize(x_val, axis=1)


model = create_model()

# compile and fit
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
model.save('model.keras')

# evaluate and draw stats
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)
draw_charts(history)
