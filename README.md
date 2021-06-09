# Pytorch를 이용한 딥러닝(DeepLearning_pytorch)


## linear_regression

- 개요
  + Pytorch로 선형회귀(Linear Regression) 모델 구현
  + 선형회귀 : 1개의 x로부터 y를 예측

- 구현 사항
  + 학습 데이터 선언
  + 가설(H(x) = Wx + b) 정의
  + 가중치(W)와 편향(b) 선언
  + 비용 함수 정의 : 평균 제곱 오차
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)

- 모델 학습
```
  Epoch    0/2000 W: 0.187, b: 0.080 Cost: 18.666666
  Epoch  100/2000 W: 1.746, b: 0.578 Cost: 0.048171
  Epoch  200/2000 W: 1.800, b: 0.454 Cost: 0.029767
  Epoch  300/2000 W: 1.843, b: 0.357 Cost: 0.018394
  Epoch  400/2000 W: 1.876, b: 0.281 Cost: 0.011366
  Epoch  500/2000 W: 1.903, b: 0.221 Cost: 0.007024
  Epoch  600/2000 W: 1.924, b: 0.174 Cost: 0.004340
  Epoch  700/2000 W: 1.940, b: 0.136 Cost: 0.002682
  Epoch  800/2000 W: 1.953, b: 0.107 Cost: 0.001657
  Epoch  900/2000 W: 1.963, b: 0.084 Cost: 0.001024
  Epoch 1000/2000 W: 1.971, b: 0.066 Cost: 0.000633
  Epoch 1100/2000 W: 1.977, b: 0.052 Cost: 0.000391
  Epoch 1200/2000 W: 1.982, b: 0.041 Cost: 0.000242
  Epoch 1300/2000 W: 1.986, b: 0.032 Cost: 0.000149
  Epoch 1400/2000 W: 1.989, b: 0.025 Cost: 0.000092
  Epoch 1500/2000 W: 1.991, b: 0.020 Cost: 0.000057
  Epoch 1600/2000 W: 1.993, b: 0.016 Cost: 0.000035
  Epoch 1700/2000 W: 1.995, b: 0.012 Cost: 0.000022
  Epoch 1800/2000 W: 1.996, b: 0.010 Cost: 0.000013
  Epoch 1900/2000 W: 1.997, b: 0.008 Cost: 0.000008
  Epoch 2000/2000 W: 1.997, b: 0.006 Cost: 0.000005
```

---

## multivariable_linear_regression

- 개요
  + Pytorch를 이용해 다중 선형 회귀(Multivariable Linear Regression) 구현
  + 다중 선형 회귀 : 다수의 x로부터 y를 예측

- 구현 사항
  + 학습 데이터(행렬) 선언
  + 가설(H(x) = Wx + b) 정의
  + 가중치(W)와 편향(b) 설정  
  + 비용 함수 정의 : 평균 제곱 오차
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)

- 모델 학습
```
  Epoch    0/20 hypothesis: tensor([0., 0., 0., 0., 0.]) Cost: 29661.800781
  Epoch    1/20 hypothesis: tensor([67.2578, 80.8397, 79.6523, 86.7394, 61.6605]) Cost: 9298.520508
  Epoch    2/20 hypothesis: tensor([104.9128, 126.0990, 124.2466, 135.3015,  96.1821]) Cost: 2915.712402
  Epoch    3/20 hypothesis: tensor([125.9942, 151.4381, 149.2133, 162.4896, 115.5097]) Cost: 915.040527
  Epoch    4/20 hypothesis: tensor([137.7967, 165.6247, 163.1911, 177.7112, 126.3307]) Cost: 287.936096
  Epoch    5/20 hypothesis: tensor([144.4044, 173.5674, 171.0168, 186.2332, 132.3891]) Cost: 91.371063
  Epoch    6/20 hypothesis: tensor([148.1035, 178.0143, 175.3980, 191.0042, 135.7812]) Cost: 29.758249
  Epoch    7/20 hypothesis: tensor([150.1744, 180.5042, 177.8509, 193.6753, 137.6805]) Cost: 10.445267
  Epoch    8/20 hypothesis: tensor([151.3336, 181.8983, 179.2240, 195.1707, 138.7440]) Cost: 4.391237
  Epoch    9/20 hypothesis: tensor([151.9824, 182.6789, 179.9928, 196.0079, 139.3396]) Cost: 2.493121
  Epoch   10/20 hypothesis: tensor([152.3454, 183.1161, 180.4231, 196.4765, 139.6732]) Cost: 1.897688
  Epoch   11/20 hypothesis: tensor([152.5485, 183.3609, 180.6640, 196.7389, 139.8602]) Cost: 1.710552
  Epoch   12/20 hypothesis: tensor([152.6620, 183.4982, 180.7988, 196.8857, 139.9651]) Cost: 1.651416
  Epoch   13/20 hypothesis: tensor([152.7253, 183.5752, 180.8742, 196.9678, 140.0240]) Cost: 1.632369
  Epoch   14/20 hypothesis: tensor([152.7606, 183.6184, 180.9164, 197.0138, 140.0571]) Cost: 1.625924
  Epoch   15/20 hypothesis: tensor([152.7802, 183.6427, 180.9399, 197.0395, 140.0759]) Cost: 1.623420
  Epoch   16/20 hypothesis: tensor([152.7909, 183.6565, 180.9530, 197.0538, 140.0865]) Cost: 1.622152
  Epoch   17/20 hypothesis: tensor([152.7968, 183.6643, 180.9603, 197.0618, 140.0927]) Cost: 1.621262
  Epoch   18/20 hypothesis: tensor([152.7999, 183.6688, 180.9644, 197.0661, 140.0963]) Cost: 1.620501
  Epoch   19/20 hypothesis: tensor([152.8014, 183.6715, 180.9665, 197.0686, 140.0985]) Cost: 1.619757
  Epoch   20/20 hypothesis: tensor([152.8020, 183.6731, 180.9677, 197.0699, 140.0999]) Cost: 1.619046
```

---

## nn_Module
- 개요
  + nn.Module을 이용해 다중 선형 회귀 클래스 구현

- 구현 사항
  + 학습 데이터(행렬) 선언
  + nn.Linear() 모듈을 이용해 다중 선형 회귀 클래스 모델 선언(MultivariateLinearRegressionModel())
  + 비용 함수 정의 : 평균 제곱 오차 함수(F.mse_loss() 사용)
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)
 
- 모델 학습
```
  Epoch    0/2000 Cost: 31667.597656
  Epoch  100/2000 Cost: 0.225993
  Epoch  200/2000 Cost: 0.223911
  Epoch  300/2000 Cost: 0.221941
  Epoch  400/2000 Cost: 0.220059
  Epoch  500/2000 Cost: 0.218271
  Epoch  600/2000 Cost: 0.216575
  Epoch  700/2000 Cost: 0.214950
  Epoch  800/2000 Cost: 0.213413
  Epoch  900/2000 Cost: 0.211952
  Epoch 1000/2000 Cost: 0.210560
  Epoch 1100/2000 Cost: 0.209232
  Epoch 1200/2000 Cost: 0.207967
  Epoch 1300/2000 Cost: 0.206761
  Epoch 1400/2000 Cost: 0.205619
  Epoch 1500/2000 Cost: 0.204522
  Epoch 1600/2000 Cost: 0.203484
  Epoch 1700/2000 Cost: 0.202485
  Epoch 1800/2000 Cost: 0.201542
  Epoch 1900/2000 Cost: 0.200635
  Epoch 2000/2000 Cost: 0.199769
```

---

## dataLoader

- 개요
  + Data Loader을 이용해 학습 데이터 관리 및 선형 회귀 구현
  + torch.utils.data.DataLoader() 사용

- 구현 사항
  + 학습 데이터 선언 및 Data Loader에 저장
  + 선형 회귀 클래스 모델 선언(nn.Linear())
  + 비용 함수 정의 : 평균 제곱 오차 함수(F.mse_loss() 사용)
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)
 
- 모델 학습
```
  Epoch    0/20 Batch 1/3 Cost: 37210.574219
  Epoch    0/20 Batch 2/3 Cost: 16463.394531
  Epoch    0/20 Batch 3/3 Cost: 2130.966309
  ...중략...
  Epoch   20/20 Batch 1/3 Cost: 12.638277
  Epoch   20/20 Batch 2/3 Cost: 7.614962
  Epoch   20/20 Batch 3/3 Cost: 35.864082
```

---

## customDataset

- 개요
  + torch.utils.data.Dataset()을 상속받아 직접 커스텀 데이터셋 만들어 선형 회귀 구현

- 구현 사항
  + torch.utils.data.Dataset()을 상속받아 CustomDataset() 구현
  + 학습 데이터를 커스텀 데이터셋에 저장
  + torch.utils.data.DataLoader()을 이용해 학습 데이터 로드
  + nn.Linear()을 이용해 선형 회귀 클래스 모델 선언
  + 비용 함수 정의 : 평균 제곱 오차 함수(F.mse_loss() 사용)
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)

- 모델 학습
```
  Epoch    0/20 Batch 1/3 Cost: 29410.156250
  Epoch    0/20 Batch 2/3 Cost: 7150.685059
  Epoch    0/20 Batch 3/3 Cost: 3482.803467
  ... 중략 ...
  Epoch   20/20 Batch 1/3 Cost: 0.350531
  Epoch   20/20 Batch 2/3 Cost: 0.653316
  Epoch   20/20 Batch 3/3 Cost: 0.010318
```

---

## logisticRegression

- 개요
  + 클래스를 이용해 로지스틱 회귀(Logistic Regression) 구현

- 구현 사항
  + 학습 데이터 선언
  + nn.Linear()과 nn.Sigmoid()를 이용해 로지스틱 회귀 클래스 모델 선언(BinaryClassifier())
  + 비용 함수 정의 : 평균 제곱 오차(F.mse_loss() 사용)
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)
 
- 모델 학습
```
  Epoch    0/1000 Cost: 0.539713 Accuracy 83.33%
  Epoch   10/1000 Cost: 0.614853 Accuracy 66.67%
  Epoch   20/1000 Cost: 0.441875 Accuracy 66.67%
  ...중략...
  Epoch  980/1000 Cost: 0.020219 Accuracy 100.00%
  Epoch  990/1000 Cost: 0.020029 Accuracy 100.00%
  Epoch 1000/1000 Cost: 0.019843 Accuracy 100.00%
```

---

## softmaxRegression

- 개요
  + 소프트맥스 회귀(Softmax Regression) 구현

- 구현 사항
  + 학습 데이터 선언
  + 가설(H(x) = Wx + b) 정의
  + 가중치(W)와 편향(b) 선언
  + 비용 함수 정의 : 크로스 엔트로피 함수(F.cross_entropy() 사용)
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)

- 모델 학습
```
  Epoch    0/1000 Cost: 1.098612
  Epoch  100/1000 Cost: 0.761050
  Epoch  200/1000 Cost: 0.689991
  Epoch  300/1000 Cost: 0.643229
  Epoch  400/1000 Cost: 0.604117
  Epoch  500/1000 Cost: 0.568256
  Epoch  600/1000 Cost: 0.533922
  Epoch  700/1000 Cost: 0.500291
  Epoch  800/1000 Cost: 0.466908
  Epoch  900/1000 Cost: 0.433507
  Epoch 1000/1000 Cost: 0.399962
```

---

## softMax_MNIST

- 개요
  + 소프트맥스 회귀(Softmax Regression)를 이용해 MNIST 데이터 분류하는 프로그램

- 구현 사항
  + 학습 데이터 MNIST 로드(torchvision.datasets.dsets.MNIST() 사용)
  + nn.Linear()을 이용해 input_dim은 784, output_dim은 10(MNIST : 0~9)인 선형 회귀 모델 선언
  + 비용 함수 정의 : 크로스 엔트로피 함수(F.cross_entropy() 사용)
  + 옵티마이저 정의 : 경사 하강법(optim.SGD() 사용)

- 모델 학습
```
  Epoch: 0001 cost = 0.535468459
  Epoch: 0002 cost = 0.359274209
  Epoch: 0003 cost = 0.331187516
  Epoch: 0004 cost = 0.316578060
  Epoch: 0005 cost = 0.307158142
  Epoch: 0006 cost = 0.300180763
  Epoch: 0007 cost = 0.295130193
  Epoch: 0008 cost = 0.290851474
  Epoch: 0009 cost = 0.287417054
  Epoch: 0010 cost = 0.284379572
  Epoch: 0011 cost = 0.281825274
  Epoch: 0012 cost = 0.279800713
  Epoch: 0013 cost = 0.277808994
  Epoch: 0014 cost = 0.276154339
  Epoch: 0015 cost = 0.274440885
  Learning finished
```

---

## perceptron_MNIST

- 개요
  + 다층 퍼셉트론을 구현하고, 학습을 통해 MNIST 분류하는 프로그램

- 구현 사항
  + 학습 데이터 MNIST 로드(sklearn.datasets.load_digits() 사용)
  + nn.Sequential(), nn.Linear(), nn.ReLU()를 이용해 input_dim은 64, output_dim은 10(MNIST : 0~9)인 다층 퍼셉트론 모델 선언
  + 비용 함수 정의 : 크로스 엔트로피 함수(F.cross_entropy() 사용)
  + 옵티마이저 정의 : Adam(optim.Adam() 사용)

- 모델 학습
```
  Epoch    0/100 Cost: 2.380815
  Epoch   10/100 Cost: 2.059323
  ... 중략 ...
  Epoch   90/100 Cost: 0.205398
```

---

## cnn_MNIST

- 개요
  + CNN 모델을 구현하고, 학습을 통해 MNIST 분류하는 프로그램

- 구현 사항
  + 학습 데이터 MNIST 로드(torchvision.datasets.dsets.MNIST() 사용)
  + nn.Conv2d(), nn.ReLU(), nn.MaxPool2d(), view(), nn.Linear()를 이용해 output_dim은 10(MNIST : 0~9)인 CNN 모델 선언
  + 비용 함수 정의 : 크로스 엔트로피 함수(F.cross_entropy() 사용)
  + 옵티마이저 정의 : Adam(optim.Adam() 사용)

- 모델 학습 
```
  [Epoch:    1] cost = 0.224006683
  [Epoch:    2] cost = 0.062186949
  [Epoch:    3] cost = 0.0449030139
  [Epoch:    4] cost = 0.0355709828
  [Epoch:    5] cost = 0.0290450025
  [Epoch:    6] cost = 0.0248527844
  [Epoch:    7] cost = 0.0207189098
  [Epoch:    8] cost = 0.0181982815
  [Epoch:    9] cost = 0.0153046707
  [Epoch:   10] cost = 0.0124179339
  [Epoch:   11] cost = 0.0105423154
  [Epoch:   12] cost = 0.00991860125
  [Epoch:   13] cost = 0.00894770492
  [Epoch:   14] cost = 0.0071221008
  [Epoch:   15] cost = 0.00588585297
```

- 정확도
```
  Accuracy: 0.9883000254631042
```
