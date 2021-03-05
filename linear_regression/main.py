import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# 데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
# 모델 초기화
w = torch.zeros(1, requires_grad=True) # 가중치 w를 0으로 초기화, 학습을 통해 값이 변경되는 변수
b = torch.zeros(1, requires_grad=True) # 편향 b를 0으로 초기화, 학습을 통해 값이 변경되는 변수
# optimizer 설정 : SGD(경사하강법)
optimizer = optim.SGD([w,b], lr=0.01)

nb_epochs = 2000 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * w + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad() # gradient를 0으로 초기화
    cost.backward() # 비용 함수를 미분하여 gradient 계산
    optimizer.step() # w와 b를 업데이트

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w.item(), b.item(), cost.item()
        ))


