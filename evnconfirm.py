import torch
import subprocess
import time
import os
import psutil

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version() if torch.cuda.is_available() else "Not Available")
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# nvidia-smi 명령어 실행 및 출력
print("\nRunning nvidia-smi command:")
try:
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
except FileNotFoundError:
    print("nvidia-smi command not found. Make sure NVIDIA drivers are installed.")

total_cores = os.cpu_count()
print(f"Total CPU cores (including hyper-threading): {total_cores}")

# 논리적 코어 수 (전체 스레드 포함)
logical_cores = psutil.cpu_count(logical=True)
print(f"Logical CPU cores: {logical_cores}")

# 물리적 코어 수
physical_cores = psutil.cpu_count(logical=False)
print(f"Physical CPU cores: {physical_cores}")

def measure_batch_time(loader, model, criterion, optimizer):
    model.train()
    total_time = 0.0
    num_batches = len(loader)
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        start_time = time.time()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        
        # 배치 처리 시간 계산
        batch_time = end_time - start_time
        total_time += batch_time

        if i % 10 == 0:
            print(f"Batch {i}/{num_batches} - 처리 시간: {batch_time:.4f}초")

    avg_batch_time = total_time / num_batches
    print(f"\n평균 배치 처리 시간: {avg_batch_time:.4f}초")
    return avg_batch_time