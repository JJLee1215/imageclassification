import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from PIL import Image
import os
from torchvision import transforms
import glob
import sys
from collections import Counter
import multiprocessing


class MyCollator:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        images, file_prefixes, labels = zip(*batch)
        images = torch.stack(images).to(self.device)
        file_prefixes = torch.tensor(file_prefixes, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        #print(f"[DEBUG] Batch Labels: {labels}, Min Label: {labels.min().item()}, Max Label: {labels.max().item()}")
        return images, file_prefixes, labels


class MyDataset(Dataset):
    '''
     image dim : [1,244,244] (gray scale) -> [3,244,244]
     Resnet50 : basically 3 channels (model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
    '''
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*/*.png')) #png or jgg
        self.label_dict = {'object_blank': 0, 'object_err': 1, 'object_lying': 2, 'object_sitting': 3, 'object_standing': 4}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label_dict = {'object_blank': 0, 'object_err': 1, 'object_lying': 2, 'object_sitting': 3, 'object_standing': 4}
        
        # 이미지 파일 경로와 레이블 가져오기
        image_filepath = self.image_paths[idx]
        folder_name = os.path.basename(os.path.dirname(image_filepath))
        label = label_dict.get(folder_name, -1)
        
        # 이미지 로드 및 전처리
        image = Image.open(image_filepath).convert('L')
        if self.transform:
            transformed_image = self.transform(image)
        
        file_prefix = int(os.path.basename(image_filepath)[:4])
        
        #print(f"[DEBUG] File: {image_filepath}, Folder: {folder_name}, Label: {label}")
        return transformed_image, file_prefix, label
        
    def __repr__(self):
        return (f"MyDataset(root_dir={self.root_dir}, "
                f"num_samples={len(self.image_paths)}, "
                f"labels={self.label_dict})")


def get_loaders(config, transform):
    '''
        1. 전체 데이터셋 로드
        2. train, valid 갯수를 설정
        3. dataset shuffling
        4. Dataloader에 적재 : train, valid, test로 나눔
    '''
    # 1. Loading Total Dataset
    dataset = MyDataset(root_dir='./data/Train_2/', transform=transform)
    test_dataset = MyDataset(root_dir='./data/Test/', transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    # 2. Split Train/Valid
    train_cnt = int(len(dataset) * config.train_ratio)
    valid_cnt = len(dataset) - train_cnt
    
    # 3. Shuffled dataset
    train_dataset, valid_dataset = random_split(dataset, [train_cnt, valid_cnt])

    # 4. Collecting class label
    train_labels = [label for _, _, label in train_dataset]
    class_counts = Counter(train_labels)
    print(f"Class Counts: {class_counts}")

    # 5. Class Label
    class_weights = 1.0 / torch.tensor([class_counts[i] for i in sorted(class_counts)], dtype=torch.float)
    sample_weights = torch.tensor([class_weights[label] for _, _, label in train_dataset])

    # 6. WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    collator = MyCollator(config.device)


    # Get the number of CPU cores available
    #num_workers = multiprocessing.cpu_count() // 4
    #num_workers = 1
    #print(f"Using num_workers={num_workers}")
    
    collator = MyCollator(config.device)

    # 4. Train, Validation DataLoader 생성 (collate_fn을 사용하여 데이터를 자동으로 GPU로 이동)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        #shuffle=True, # sampler와 shuffle을 동시에 사용할 수 없음
        drop_last=True,
        collate_fn=collator,
        #num_workers=num_workers,  # 동적 num_workers 사용
        #persistent_workers=False,
        #pin_memory_device='cuda',
        #pin_memory=False  # GPU 사용 시 속도 향상
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collator,
        #num_workers=num_workers,
        #persistent_workers=False,
        #pin_memory_device='cuda',
        #pin_memory=False  # GPU 사용 시 속도 향상
    )

    # 5. Test DataLoader 생성 (collate_fn 사용)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        #num_workers=num_workers,
        #persistent_workers=False,
        #pin_memory_device='cuda',
        #pin_memory=False  # GPU 사용 시 속도 향상
    )
    
    return train_loader, valid_loader, test_loader


