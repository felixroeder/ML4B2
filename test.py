import tensorflow as tf

model = tf.keras.models.load_model('trained_model.keras', safe_mode=False, compile=False)
