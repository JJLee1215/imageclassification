import os
import shutil

def organize_images_by_prefix(source_dir, target_dir):
    """
    디렉토리 내 모든 PNG 파일을 앞 네 자리 숫자에 따라 폴더를 생성하고 이동.
    
    Args:
    - source_dir (str): 원본 파일들이 있는 디렉토리 (상대 경로).
    - target_dir (str): 파일을 이동할 대상 디렉토리 (상대 경로).
    """
    # 소스 디렉토리에서 모든 PNG 파일 검색
    png_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]

    if not png_files:
        print("이동할 PNG 파일이 없습니다.")
        return

    for file_name in png_files:
        # 앞 네 자리 숫자 추출
        prefix = file_name[:4]
        if not prefix.isdigit():
            print(f"잘못된 파일 이름 형식 (숫자 추출 불가): {file_name}")
            continue

        # 대상 폴더 경로 생성
        target_folder = os.path.join(target_dir, f"mvpc_{prefix}")
        os.makedirs(target_folder, exist_ok=True)  # 폴더가 없으면 생성

        # 파일 이동 경로 설정
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_folder, file_name)

        try:
            shutil.move(source_file, target_file)
            print(f"파일 이동: {source_file} -> {target_file}")
        except Exception as e:
            print(f"파일 이동 실패: {source_file}, 에러: {e}")


# 상대 경로 설정
source_directory = "./data/ir_total"  # 원본 파일 경로
target_directory = "./data/ir_images"  # 이동 대상 경로

# 함수 실행
organize_images_by_prefix(source_directory, target_directory)
