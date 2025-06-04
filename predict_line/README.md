### 추론 코드 설명
 - ##### 모델 로딩 및 인터프리터 초기화
```
    model_path = "final_int8_edgetpu.tflite"
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
```
Edge TPU용 양자화 모델 로딩

make_interpreter는 Coral에서 제공하는 함수 자동 최적화

allocate_tensors() 로 메모리 할당

 - ##### 입출력 텐서 정보 추출
```
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    scale, zero_point = output_details["quantization"]
```
입출력 텐서 구조와 데이터 타입 확인

scale, zero_point: 양자화 모델에서 uint8->float형으로 되돌릴 때 사용

 - ##### 웹캠 연결
```
    cap = cv2.VideoCapture(0)
```
 - ##### 프레임 루프 시작

```
    while True:
      ret, frame = cap.read()
```

 - ##### 전처리

```
    img = cv2.resize(frame, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    input_data = np.expand_dims(img, axis=0)
```

모델 입력 크기 리사이즈

RGB로 변환

TFLite 모델의 입력 형식에 맞게 채널 우선 CHW로 변환 후 배치 차원 추가

 - ##### 추론 실행

```
    common.set_input(interpreter, input_data)
    interpreter.invoke()
```

입력 데이터 전달 후 interpreter.invock()로 추론 실행

 - ##### 모델 출력 후처리

```
mask_float = (mask_uint8.astype(np.float32) - zero_point) * scale
mask_sigmoid = scipy.special.expit(mask_float)
mask_bin = (mask_sigmoid >= 0.5).astype(np.uint8) * 255
```
양자화 출력-> float으로 변경

Sigmoid 활성화

값이 0.5 이상이면 차선으로 판단

 - ##### 마스크 리사이징 및 컬러 오버레이

```
mask_resized = cv2.resize(mask_bin, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

mask_color = np.zeros_like(frame)
mask_color[mask_resized == 255] = (255, 0, 0)  # Red for lane

overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
```
마스크를 원본 프레임 크기로 확대

파란색 (BGR)으로 차선 부분 표시

addWeighted로 원본과 오버레이 혼합

 - ##### FPS 계산 및 표시

```
fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
cv2.putText(overlay, f"FPS: {fps:.2f}", ...)
```
이전 프레임과의 시간 차를 기반으로 FPS 계산

 - ##### 결과 출력 및 종료 조건

```
cv2.imshow("Lane Segmentation (Edge TPU)", overlay)

if cv2.waitKey(1) & 0xFF in (27, ord("q")):
    break
```
종료
