import pathlib

from ext.ChartsExt import draw_charts
from ext.model import create_model
from ext.DatasetParser import parseDataSet
import tensorflow as tf

path = str(pathlib.Path().resolve())

x_train, y_train = parseDataSet(path + '/dataset')

x_train = tf.keras.utils.normalize(x_train, axis=1)

print(x_train.shape)
print(y_train.shape)

model = create_model()

# compile and fit
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(x_train, y_train, epochs=10)
model.save('model.keras')

# evaluate and draw stats
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
draw_charts(history)
