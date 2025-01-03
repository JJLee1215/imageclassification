from copy import deepcopy
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import sys

class Trainer:
    def __init__(self, model, optimizer, crit, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config
        self.train_losses = []  # Train loss를 저장할 리스트
        self.valid_losses = []  # Validation loss를 저장할 리스트

    def _train(self, train_loader):
        '''
            crit에 집어넣기 위해서는 y_i와 y_hat_i의 차원, tensor type = long tensor로 일치해야함.
            * long tensor: 정수형 타입의 텐서
            * 클래스의 손실 함수에서는 정수형 타입인 long tensor만 허용된다.
        '''
        self.model.train()
        total_loss = 0
    
        for i, (x_i, file_prefix, y_i) in enumerate(train_loader):
            # GPU로 데이터 이동
            x_i = x_i.to(self.config.device)
            file_prefix = file_prefix.to(self.config.device)
            y_i = y_i.to(self.config.device)


            # if self.config.verbose >= 2:
            #     print(f"[DEBUG] Batch {i+1}/{len(train_loader)}:")
            #     print(f" - x_i.shape: {x_i.shape}")  # 이미지 텐서 크기
            #     print(f" - file_prefix.shape: {file_prefix.shape}")  # 파일명 접두사 크기
            #     print(f" - y_i.shape: {y_i.shape}")  # 레이블 크기

            # 모델 예측
            y_hat_i = self.model(x_i, file_prefix)

            # 손실 계산
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # 역전파 및 옵티마이저 업데이트
            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            # 디버깅 출력
            if self.config.verbose >= 2:
                print("Train Iteration (%d/%d): loss = %.4e" % (i + 1, len(train_loader), float(loss_i)))

            total_loss += float(loss_i)

        avg_loss = total_loss / len(train_loader)  # Train loss 평균 계산
        self.train_losses.append(avg_loss)  # Train loss 저장    

        return avg_loss

    def _validate(self, valid_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for i, (x_i, file_prefix, y_i) in enumerate(valid_loader):
                # GPU로 데이터 이동
                x_i = x_i.to(self.config.device)
                file_prefix = file_prefix.to(self.config.device)
                y_i = y_i.to(self.config.device)

                # 디버깅 출력
                # if self.config.verbose >= 2:
                #     print(f"[DEBUG] Validation Batch {i+1}/{len(valid_loader)}:")
                #     print(f" - x_i.shape: {x_i.shape}")  # 이미지 텐서 크기
                #     print(f" - file_prefix.shape: {file_prefix.shape}")  # 파일명 접두사 크기
                #     print(f" - y_i.shape: {y_i.shape}")  # 레이블 크기

                # 모델 예측
                y_hat_i = self.model(x_i, file_prefix)

                # 손실 계산
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                # 디버깅 출력
                if self.config.verbose >= 2:
                    print("Valid Iteration (%d/%d): loss = %.4e" % (i + 1, len(valid_loader), float(loss_i)))

                total_loss += float(loss_i)

        avg_loss = total_loss / len(valid_loader)  # Validation loss 평균 계산
        self.valid_losses.append(avg_loss)  # Validation loss 저장

        return avg_loss

    def save_checkpoint(self, epoch_index, valid_loss):
        """모델 가중치 저장 함수"""
        # 현재 날짜와 시간 추가
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 체크포인트 파일명 생성 (날짜, epoch, loss 포함)
        checkpoint_path = f"checkpoint_{current_time}_epoch_{epoch_index}_loss_{valid_loss:.4e}.pth"

        # 모델 상태 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch_index,
            'loss': valid_loss,
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self, train_loader, valid_loader, start_epoch=1):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(start_epoch, self.config.n_epochs + 1):
            train_loss = self._train(train_loader)
            valid_loss = self._validate(valid_loader)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch (%d/%d): train_loss = %.4e   valid_loss = %.4e   lowest_loss = %.4e" % (
                epoch_index,  # 현재 에포크 번호
                self.config.n_epochs,  # 총 에포크 수
                train_loss,
                valid_loss,
                lowest_loss,
            ))

            # 매 5 에포크마다 가중치 저장
            if epoch_index % 5 == 0:
                self.save_checkpoint(epoch_index, valid_loss)

        # 최종적으로 최적의 모델 로드
        self.model.load_state_dict(best_model)

        # 학습 종료 후 loss 그래프 그리기
        self.plot_losses()

    def plot_losses(self):
        """Train loss와 Validation loss를 그래프로 시각화"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')  # Train Loss 곡선
        plt.plot(self.valid_losses, label='Validation Loss')  # Validation Loss 곡선
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.grid()
        plt.savefig('loss_curve.png')  # 그래프를 이미지 파일로 저장
        plt.show()