# Work_experience
청년 일경험 사업 정리

### 인공지능 모델
UNet MobileNet v2
 - Edge TPU TFLite 모델 로딩
 - 이미지 전처리 후 모델 추론
 - 모델 출력: 디퀀타이즈, 시그모이드, threshold
 - 차선 마스크 시각화 및 원본 영상에 오버레이
 - FPS 표시 및 실시간 출력
### 모델 변환



![image](https://github.com/user-attachments/assets/401144fb-dc1b-4048-bbbc-d82a92a37637)  


```
https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb#scrollTo=license
```
 - tflite to edgetpu.tflite
tflite 모델을 컴파일러 웹을 통해 변환
