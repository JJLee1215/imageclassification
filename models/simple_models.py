'''
Purpose : building fully connected layers
input size : 28*28, 500
500 -> 400 -> 300 -> 200 -> 100 -> 50 -> output 

1. BatchNorm - normalize for mini-batches 

Regularization = minimize training error 
1) Weight Decay
2) Dropout
=> training time is delayed


input tensor shape = |X| = (batch_size, vector_size)
mu = x.mean(dim = 0), sigma = x.std (dim = 0)
|mu| = |sigma| = (vs,) <- the same dimension as vector_size

-> Unit Gaussaian 

    y = gamma (x - mu)/(sigma^2 + e)^0.5 + beta

    gamma, beta <- Learning parameters updated by backpropagation!!

"How much should we increase and shift to find favorable parameters for learning?"
 * gamma -> increase
 * shift -> beta

first step: regularization
second step: applying gamma and shift

Caution!

Learning: calculate mu and sigma within Mini-batches.
Inference: calculate mu and sigma within Mini-batches -> cheating => calculate the average and std from the accumulated input.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        super.__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.logsoftmax(dim = -1)
        )
        
    def forward(self,x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)   
        # |y} = (batch_size, output_size) 
        return y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    
    

class CustomResNet50(nn.Module):
    '''
        입력 이미지 (x): (batch_size, 3, 224, 224)
        ↓ ResNet50 Convolution + Pooling
        → (batch_size, 2048)
        ↓ FC 레이어 (ResNet50의 마지막 단계)
        → img_features: (batch_size, 512)

        파일명 접두사 (file_prefix): (batch_size,)
        ↓ 차원 추가 (unsqueeze)
        → (batch_size, 1)
        ↓ 파일명 처리 (file_prefix_fc)
        → file_features: (batch_size, 32)

        결합 (torch.cat):
        img_features: (batch_size, 512)
        file_features: (batch_size, 32)
        → combined_features: (batch_size, 544)

        최종 FC 레이어 (final_fc):
        combined_features: (batch_size, 544)
        → 최종 출력: (batch_size, num_classes)
    '''
    def __init__(self, num_classes=5):
        super().__init__()
        # Pre-trained ResNet50 사용
        self.model = models.resnet50(pretrained=True)
        
        # ResNet50의 Fully Connected (FC) 레이어 수정
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 파일명 접두사(4자리) 추가 처리
        self.file_prefix_fc = nn.Sequential(
            nn.Linear(1, 32),  # 파일명 접두사 입력: 스칼라(1)
            nn.ReLU()
        )

        self.final_fc = nn.Sequential(
            nn.Linear(512 + 32, 256),  # 중간 레이어로 256차원으로 줄임
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),       # 다시 128차원으로 줄임
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # 최종 3차원으로 줄임
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, file_prefix):
        # 이미지 특징 추출
        img_features = self.model(x)
        #print(f"[DEBUG] img_features.shape: {img_features.shape}")  # 이미지 특징 확인

        # 파일명 접두사 처리
        file_prefix = file_prefix.unsqueeze(1)  # 추가 차원 삽입 (batch_size, 1)
        file_features = self.file_prefix_fc(file_prefix)
        #print(f"[DEBUG] file_features.shape before squeeze: {file_features.shape}")  # file_features 크기 확인

        # file_features의 차원을 2차원으로 맞춤
        file_features = file_features.squeeze(1)  # (batch_size, 1, 32) → (batch_size, 32)
        #print(f"[DEBUG] file_features.shape after squeeze: {file_features.shape}")

        # 이미지와 파일명 특징 결합
        combined_features = torch.cat((img_features, file_features), dim=1)
        #print(f"[DEBUG] combined_features.shape: {combined_features.shape}")  # 결합된 크기 확인

        # 최종 출력
        output = self.final_fc(combined_features)
        return output