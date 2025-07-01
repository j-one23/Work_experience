import tensorflow as tf
import numpy as np

def representative_dataset_gen():
    for _ in range(100):
        data = np.random.rand(1, 3, 256, 256).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_saved_model("unet_tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

converter.experimental_new_converter = True

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open("unet_tpu_int8_2.tflite", "wb") as f:
    f.write(tflite_model)

print("int8 양자화")
