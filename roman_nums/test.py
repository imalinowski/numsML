import pathlib
import tensorflow as tf
from ext.Test import testModel

test_data_path = str(pathlib.Path().resolve()) + '/digits'

model = tf.keras.models.load_model('model.keras')

testModel(test_data_path, model)
