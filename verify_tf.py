
import tensorflow as tf
import tensorflow_addons as tfa
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Addons version: {tfa.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs Available: {len(gpus)}")
for gpu in gpus:
    print(f" - {gpu}")

try:
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    z = tf.matmul(x, y)
    print("Basic matmul test: SUCCESS")
except Exception as e:
    print(f"Basic matmul test: FAILED - {e}")
