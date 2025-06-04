import cv2
import numpy as np
import scipy.special
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import time

def main():
    model_path = "final_int8_edgetpu.tflite"
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    scale, zero_point = output_details["quantization"]

    _, channel, height, width = input_details["shape"]
    input_dtype = input_details["dtype"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Preprocess
        img = cv2.resize(frame, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if input_dtype == np.float32:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.uint8)
        img = np.transpose(img, (2, 0, 1))  # HWC 占쏙옙 CHW
        input_data = np.expand_dims(img, axis=0)  # 1占쏙옙C占쏙옙H占쏙옙W

        common.set_input(interpreter, input_data)
        interpreter.invoke()

        # 2) Get raw uint8 output and squeeze
        output_uint8 = interpreter.get_tensor(output_details["index"])[0]
        if output_uint8.ndim == 3 and output_uint8.shape[0] == 1:
            mask_uint8 = np.squeeze(output_uint8, axis=0)
        else:
            mask_uint8 = output_uint8

        # 3) Dequantize to float
        mask_float = (mask_uint8.astype(np.float32) - zero_point) * scale

        # 4) Sigmoid activation
        mask_sigmoid = scipy.special.expit(mask_float)

        # 5) Threshold: use >= 0.5 so pixels with sigmoid==0.5 are included
        mask_bin = (mask_sigmoid >= 0.5).astype(np.uint8) * 255

        # 6) Resize mask back to original frame size
        mask_resized = cv2.resize(
            mask_bin,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # 7) Create red overlay where mask==255
        mask_color = np.zeros_like(frame)
        mask_color[mask_resized == 255] = (255, 0, 0)  # BGR: red

        # 8) Overlay red mask onto original frame
        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        # 9) Compute & display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time
        cv2.putText(
            overlay,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # 10) Show result
        cv2.imshow("Lane Segmentation (Edge TPU)", overlay)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
