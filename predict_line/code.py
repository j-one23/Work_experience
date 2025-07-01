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
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(img, axis=0)  # 1,C,H,W


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

        
        # 6-1) Crop mask to desired region (1/2 to 5/4 of the height)
        h = mask_resized.shape[0]
        start_y = h // 2
        end_y = min(h * 5 // 4, h)  # prevent going beyond frame height
        mask_cropped = mask_resized[start_y:end_y, :]

        # 6-2) Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cropped)

        # 6-3) Print object info
        print(f"Detected Objects: {num_labels - 1}")  # exclude background
        object_centroids = []
        for i in range(1, num_labels):  # label 0 is background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            object_centroids.append((i, cx, cy))
            if area > 500:
                print(f"Object {i}: x={x}, y={y}, w={w}, h={h}, area={area}, centroid=({cx:.1f}, {cy + start_y:.1f})")
  
            
        # 7) Create blue overlay where mask==255
        mask_color = np.zeros_like(frame)
        mask_color[mask_resized == 255] = (255, 0, 0)  # BGR: blue

        # 8) Overlay blue mask onto original frame
        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.9, 0)

        # 6-4) closest x, in the middle line
        crop_h, crop_w = mask_cropped.shape
        center_x = crop_w // 2
        target_y = crop_h // 2

        mid_line = mask_cropped[target_y]  # 1D array

        white_indices = np.where(mid_line == 255)[0]

        if len(white_indices) >= 2:
            left_candidates = white_indices[white_indices < center_x]
            right_candidates = white_indices[white_indices >= center_x]

            if len(left_candidates) > 0 and len(right_candidates) > 0:
                # find 2 point, each side
                x1 = left_candidates[np.argmax(left_candidates)] 
                x2 = right_candidates[np.argmin(right_candidates)]  

                pt1 = (int(x1), int(target_y + start_y))
                pt2 = (int(x2), int(target_y + start_y))
                
                cv2.circle(overlay, pt1, 5, (0, 255, 255), -1)  
                cv2.circle(overlay, pt2, 5, (0, 255, 255), -1)
                cv2.line(overlay, pt1, pt2, (0, 0, 255), 2) 

                dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                print(f"Distance between closest white pixels at middle row: {dist:.2f} pixels")


        # 9) Compute & display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time
        cv2.putText( overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 10) Show result
        cv2.imshow("Lane Segmentation (Edge TPU)", overlay)
        cv2.imshow("mask", mask_cropped)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
