import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("unet_tpu_int8.onnx")

tf_rep = prepare(onnx_model)

tf_rep.export_graph("unet_tf_model")
