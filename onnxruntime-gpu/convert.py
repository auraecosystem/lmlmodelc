import tensorflow as tf
import tf2onnx

# Build and convert a Keras model to ONNX with a dynamic batch size
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,), name="input"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3),
])
input_signature = [tf.TensorSpec([None, 4], tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=17)
tf.io.write_file("keras_model.onnx", onnx_model.SerializeToString())
