import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchsummary import summary
import torch.multiprocessing as mp

import argparse

from data_loader import MyDataset, get_loaders
from models.simple_models import FullyConnectedClassifier, Net, CustomResNet50
from trainer import Trainer
from config import CustomResNet50Wrapper
#from evnconfirm import measure_batch_time

# 1) 학습 중간에 pth 저장 코드 삽입ok
# 2) GPU 메모리 계산
# 3) 예상 속도 계산
# 4) Tensor 차원 계산
# 5) 고정메모리영역과 속도
# 6) 비슷한 이미지 삭제 코드 테스트 -> GPU 학습 끝나면 일단 저장 후 multi processing 검토 -> 비슷한 이미지 삭제 코드 테스트
# 7) profiler 활용 방법 자세히 알기 (많이 유용하다)
# 8) 메모리 충돌 : spawn, forkserver, fork 방식 차이 알기
# 9) cross entropy 가중치 적용하는 방법

def define_argparser():
    p = argparse.ArgumentParser()
    
    #p.add_argument('--model_fn', required = True)
    p.add_argument('--gpu_id', type = int, default = 0 if torch.cuda.is_available() else - 1)

    p.add_argument('--train_ratio', type = float, default = .8)
    p.add_argument('--batch_size', type = int, default = 128) # batch size 256에서 돌아가지 않음 GPU 메모리와, 내가 올린 메모리의 계산 확인이 필요함 
    p.add_argument('--n_epochs', type = int, default = 90)
    p.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    p.add_argument('--verbose', type = int, default = 1)
    
    p.add_argument('--model', type = str, default = 'resnet50')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to save/load checkpoint')

    config = p.parse_args()
    
    # GPU 또는 CPU 장치 설정
    if config.gpu_id >= 0 and torch.cuda.is_available():
        config.device = torch.device(f'cuda:{config.gpu_id}')
    else:
        config.device = torch.device('cpu')
    
    print(f"Using device: {config.device}")
    return config


def get_model(config):
    if config.model == 'fc':
        model = FullyConnectedClassifier(28**2, 10)
    elif config.model == 'resnet50':
        model = CustomResNet50(num_classes=5)
    else:
        raise NotImplementedError('You need to specify model name.')
    
    return model

def load_checkpoint(model, optimizer, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{path}' at epoch {epoch} with loss {loss:.4f}")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{path}'")
        return None, None



def main(config):
    print("device: ", config.device)
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # 1. Image Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Grayscale을 3채널로 확장
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    

    # 2. Dataset Loading
    train_loader, valid_loader, test_loader = get_loaders(config, transform)
    print("Train: ", len(train_loader.dataset))
    print("Valid: ", len(valid_loader.dataset))
    print("Test: ", len(test_loader.dataset))
        

    # 3. Model Definition
    model = get_model(config).to(config.device)
    
    # 4. Optimzier and Loss Function.
    #optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    crit = nn.NLLLoss()

    # Confirm of Dimention
    images, file_prefixes, _ = next(iter(train_loader))
    _, channels, height, width = images.shape
    input_size = (channels, height, width)
    print(f"Detected input_size from train_loader: {input_size}")
    
    if config.verbose >= 1:
        #print(model)
        wrapped_model = CustomResNet50Wrapper(model)
        summary(wrapped_model, input_size=input_size)
        print(optimizer)
        print(crit)
    
    ## 배치 처리 시간 측정 실행
    # avg_time_per_batch = measure_batch_time(train_loader, model, crit, optimizer)
    # # 1 Epoch 당 예상 처리 시간 계산
    # num_batches = len(train_loader)
    # epoch_time_estimate = avg_time_per_batch * num_batches
    # print(f"\n1 Epoch 예상 처리 시간: {epoch_time_estimate:.2f}초")
    # sys.exit()

    # 5. Checkpoint 로드 (모델과 옵티마이저가 정의된 이후)
    last_epoch = 0
    if config.checkpoint is not None and os.path.exists(config.checkpoint):
        last_epoch, _ = load_checkpoint(model, optimizer, config.checkpoint, config.device)
        print(f"Continuing training from checkpoint at epoch {last_epoch + 1}")
    else:
        print("No checkpoint found, starting training from scratch.")

    # 5. Create of Trainer Object and Start Training
    trainer = Trainer(model, optimizer, crit, config)
    trainer.train(train_loader, valid_loader, start_epoch=last_epoch + 1)

    
    # 7. 학습 종료 후 가중치 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': last_epoch + config.n_epochs,
        'loss': crit,
    }, 'resnet50_checkpoint.pth')
    

if __name__ == '__main__':
    config = define_argparser()
    #torch.multiprocessing.set_start_method('spawn', force =True)
    main(config)
    