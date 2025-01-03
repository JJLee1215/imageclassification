'''
    Code to copy test images from the original images to the test folder
'''

import os
import shutil
from tqdm import tqdm  # tqdm 임포트

# 현재 작업 디렉터리 확인
current_dir = os.getcwd()

# 변수 설정
#mvpc_list = ["mvpc_1001","mvpc_1002","mvpc_1003","mvpc_1004","mvpc_1005","mvpc_1006","mvpc_1007","mvpc_1008","mvpc_1009","mvpc_1010","mvpc_1011",
#                "mvpc_2002","mvpc_2003","mvpc_2004","mvpc_2005","mvpc_2006","mvpc_2007","mvpc_2008","mvpc_2009","mvpc_2010","mvpc_2011",
#                "mvpc_2013","mvpc_2014","mvpc_2015","mvpc_2016","mvpc_2017","mvpc_2018","mvpc_2019","mvpc_2020","mvpc_2021","mvpc_2022","mvpc_3001"] 
mvpc_list = ["mvpc_3001"] 

for mvpc_num in mvpc_list:
    source_path = os.path.join("..", "mvpc_원본", mvpc_num)
    destination_path = os.path.join("data", "Test", mvpc_num)

    # 경로 확인
    print(f"Source Path: {source_path}")
    print(f"Destination Path: {destination_path}")

    # 대상 폴더가 존재하지 않으면 생성
    os.makedirs(destination_path, exist_ok=True)

    # 소스 경로가 존재하는지 확인
    if os.path.exists(source_path):
        # 이미지 파일 리스트 수집
        image_files = []
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    image_files.append(os.path.join(root, file))

        # tqdm을 사용하여 파일 복사 진행률 표시
        for file_path in tqdm(image_files, desc="Copying files", unit="file"):
            shutil.copy2(file_path, destination_path)

        print("\nAll image files have been copied successfully.")
    else:
        print(f"Source path does not exist: {source_path}")