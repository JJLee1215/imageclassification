'''
    Code to check how many image files are in a folder
'''

import os

folder_path = "data/Train"

# 메인 딕셔너리 생성
device_dics = {}

# 폴더 내 모든 하위 폴더를 순회
for root, dirs, _ in os.walk(folder_path):
    for dir_name in dirs:
        image_folder = os.path.join(root, dir_name)
        device_dic = {}  # 각 폴더마다 새로운 딕셔너리 생성

        # 하위 폴더 내 모든 파일을 순회
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith('.png'):
                    # 파일 이름의 앞 4자리를 추출하여 key로 사용
                    file_name = file[:4]
                    
                    # 해당 key가 이미 존재하면 카운트 증가, 아니면 초기값 1로 설정
                    if file_name in device_dic:
                        device_dic[file_name] += 1
                    else:
                        device_dic[file_name] = 1

        # 메인 딕셔너리에 하위 딕셔너리 저장
        device_dics[dir_name] = device_dic

# 결과 출력
for dir_name, device_dic in device_dics.items():
    print(f"Directory: {dir_name}")
    print(device_dic)
    print("-" * 40)


 

