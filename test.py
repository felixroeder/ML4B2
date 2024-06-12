import tensorflow as tf

model = tf.keras.models.load_model('trained_model.h5', safe_mode=False, compile=False)


