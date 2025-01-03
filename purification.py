'''
   Code for collecting training data by removing duplicate or similar images during image collection 
'''

import cv2
import os
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageFilter
import imagehash
from tqdm import tqdm

def imread_with_denoise(image_path, kernel_size=(3, 3)):
    """이미지를 그레이스케일로 읽고 Gaussian Blur로 노이즈 제거"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.GaussianBlur(img, kernel_size, 0)  # Gaussian Blur 적용
    return img

def get_image_hash(image_path):
    """노이즈를 제거한 이미지의 해시값을 반환"""
    img = Image.open(image_path).convert('L')  # 그레이스케일로 변환
    img = img.filter(ImageFilter.GaussianBlur(radius=1))  # Pillow를 사용해 Gaussian Blur 적용
    return imagehash.average_hash(img)

def compare_images_ssim(image1_path, image2_path):
    """노이즈를 제거한 두 이미지의 SSIM을 비교하고 유사도 점수를 반환"""
    img1 = imread_with_denoise(image1_path)
    img2 = imread_with_denoise(image2_path)

    if img1 is None or img2 is None:
        raise ValueError("One of the images couldn't be loaded.")

    # 이미지 크기 맞추기
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # SSIM 계산
    score, _ = ssim(img1, img2, full=True)
    return score

def find_and_move_duplicates(source_folder, target_folder, hash_threshold=5, ssim_threshold=0.9):
    """폴더 내 비슷한 이미지들을 찾아 다른 폴더로 옮기기"""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    images = [os.path.join(source_folder, img) for img in os.listdir(source_folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    hashes = {}
    moved_files = []

    for i, image_path in enumerate(tqdm(images, desc="Processing images")):
        try:
            img_hash = get_image_hash(image_path)
            duplicate_found = False

            for stored_path, stored_hash in hashes.items():
                # 해시 값 비교
                if abs(img_hash - stored_hash) <= hash_threshold:
                    # SSIM 비교로 확인
                    ssim_score = compare_images_ssim(image_path, stored_path)
                    if ssim_score >= ssim_threshold:
                        #print(f"Duplicate found: {image_path} is similar to {stored_path} (SSIM: {ssim_score:.2f})")
                        shutil.move(image_path, os.path.join(target_folder, os.path.basename(image_path)))
                        moved_files.append(image_path)
                        duplicate_found = True
                        break

            if not duplicate_found:
                hashes[image_path] = img_hash

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"\nTotal duplicates moved: {len(moved_files)}")

# 소스 폴더와 대상 폴더 경로 설정
source_folder = "data/Train_2/object_sitting"           # IR 이미지가 저장된 소스 폴더 경로로 변경하세요
target_folder = "data/Train_2/trash_sitting"            # 유사한 이미지를 옮길 대상 폴더 경로로 변경하세요

# 함수 실행
find_and_move_duplicates(source_folder, target_folder)
