import torch
import torch.nn as nn

# 래퍼 클래스 정의
class CustomResNet50Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 더미 file_prefix 생성 (배치 크기와 동일한 0 텐서)
        batch_size = x.shape[0]
        file_prefix = torch.zeros(batch_size, 1, device=x.device)
        return self.model(x, file_prefix)