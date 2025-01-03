import os
import shutil
import torch
from torchvision import transforms
from PIL import Image
from models.simple_models import CustomResNet50
from datetime import datetime
from tqdm import tqdm

# 1. 설정
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")  

mvpc_list = ["mvpc_1001","mvpc_1002","mvpc_1003","mvpc_1004","mvpc_1005","mvpc_1006","mvpc_1007","mvpc_1008","mvpc_1009","mvpc_1010","mvpc_1011",
               "mvpc_2002","mvpc_2003","mvpc_2004","mvpc_2005","mvpc_2006","mvpc_2007","mvpc_2008","mvpc_2009","mvpc_2010","mvpc_2011",
               "mvpc_2013","mvpc_2014","mvpc_2015","mvpc_2016","mvpc_2017","mvpc_2018","mvpc_2019","mvpc_2020","mvpc_2021","mvpc_2022","mvpc_3001"] 

for mvpc_num in mvpc_list:

    input_folder = f"data/Test/{mvpc_num}"
    output_folder = f"sorted_{mvpc_num}_{current_date}"  # 날짜가 포함된 폴더명으로 설정
    os.makedirs(output_folder, exist_ok=True)

    # 분류 폴더 생성
    categories = ['object_blank', 'object_err', 'object_lying', 'object_sitting', 'object_standing']
    for category in categories:
        os.makedirs(os.path.join(output_folder, category), exist_ok=True)

    # 2. 장치 설정 (GPU 또는 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. 모델 로드
    model = CustomResNet50(num_classes=5).to(device)
    #checkpoint_path = 'checkpoint_2024-11-20_18-49-29_epoch_25_loss_1.0715e-01.pth'
    checkpoint_path = 'checkpoint_2024-12-19_06-18-46_epoch_35_loss_9.9942e-02.pth'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        #model.load_state_dict(checkpoint)
        model.eval()
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")

    # 4. 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # IR 이미지를 3채널로 확장
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 5. 이미지 분류 및 이동 함수
    def classify_and_move_image(image_path):
        """이미지를 분류하고 예측된 폴더로 이동"""
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

        # file_prefix 생성 및 텐서 변환
        file_prefix = int(os.path.basename(image_path)[:4])  # 파일명 앞 4자리
        file_prefix = torch.tensor([file_prefix], dtype=torch.float32).to(device)  # 텐서 변환

        # 예측
        with torch.no_grad():
            outputs = model(image, file_prefix=file_prefix)  # file_prefix 전달
            _, predicted = torch.max(outputs, 1)
            category_index = predicted.item()
            category_name = categories[category_index]

        # 예측된 폴더로 파일 이동
        destination_folder = os.path.join(output_folder, category_name)
        shutil.move(image_path, os.path.join(destination_folder, os.path.basename(image_path)))
        #print(f"Moved {os.path.basename(image_path)} to {category_name}")

    # 6. Test_image 폴더 내의 모든 이미지 분류 및 이동
    print(input_folder)
    all_files = [os.path.join(root, file)
                 for root, _, files in os.walk(input_folder)
                 for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # tqdm을 사용하여 프로그레스 바 표시
    for file_path in tqdm(all_files, desc=f"Processing {mvpc_num}", unit="image"):
        classify_and_move_image(file_path)

    print(f"{mvpc_num} - Image classification and sorting completed.")