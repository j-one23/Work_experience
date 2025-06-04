### 추론 코드 설명
 - ##### 모델 로딩 및 인터프리터 초기화

    model_path = "final_int8_edgetpu.tflite"
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
   
Edge TPU용 양자화 모델 로딩

make_interpreter는 Coral에서 제공하는 함수 자동 최적화

allocate_tensors() 로 메모리 할당

 - ##### 입출력 텐서 정보 추출

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    scale, zero_point = output_details["quantization"]

입출력 텐서 구조와 데이터 타입 확인

scale, zero_point: 양자화 모델에서 uint8->float형으로 되돌릴 때 사용

 - ##### 웹캠 연결

    cap = cv2.VideoCapture(0)

 - ##### 프레임 루프 시작
 - 
