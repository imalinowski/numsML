import pathlib
import tensorflow as tf
from ext.Test import testModel

test_data_path = str(pathlib.Path().resolve()) + '/digits'

model = tf.keras.models.load_model('best_model_4.keras')  # acc 70% loss 30%
# model = tf.keras.models.load_model('best_model_3.keras')  # acc 50% loss 50%
# model = tf.keras.models.load_model('best_model_2.keras')  # acc 45% loss 45%
# model = tf.keras.models.load_model('best_model.keras')  # acc 36% loss 54%
# model = tf.keras.models.load_model('model.keras')

testModel(test_data_path, model)
