import tensorflow as tf

model = tf.keras.models.load_model('retrained_model.keras', safe_mode=False, compile=False)
