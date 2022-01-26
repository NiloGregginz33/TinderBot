import tensorflow as tf
import keras

converter = tf.lite.TFLiteConverter.from_saved_model("facial_attractiveness.h5")
tflite_model = converter.convert()

# Save the model.
with open('facial_attractiveness_model.tflite', 'wb') as f:
  f.write(tflite_model)
