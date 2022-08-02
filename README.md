# CSRC_harmful
harmful site detection

### Usage
```
$ python3 main2.py --gpu [gpu_id]
```


## Gyumin
### input: dataset/week 데이터셋 csv 파일
### output: model, shap 기여도 excel 파일

### python main.py

#### 필수: week_ 내 csv 파일 존재

#### 순서: 1) 주간 model 저장, shap 기여도 저장
####     : 2) shap 전처리 Lite 모델 저장
####     : 3) week 파일 -> total 이동
####     : 4) total model 저장, shap 기여도 저장
####     : 5) shap 전저리 Lite 모델 저장