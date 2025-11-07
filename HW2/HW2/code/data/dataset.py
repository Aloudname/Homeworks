"""数据集管理模块"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config.settings import Config


class MedicalImageDataset(Dataset):
    """医学影像数据集"""
    
    def __init__(self, data_directory, transform=None):
        self.data_directory = data_directory
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.image_names = []
        
        self._load_dataset()
        
    def _load_dataset(self):
        """加载数据集"""
        for category, label in Config.CLASS_MAPPING.items():
            category_path = os.path.join(self.data_directory, category)
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if self._is_valid_image(filename):
                        full_path = os.path.join(category_path, filename)
                        self.image_paths.append(full_path)
                        self.labels.append(label)
                        self.image_names.append(filename)
        
        print(f"数据集加载: {self.data_directory} - {len(self.image_paths)} 张影像")
    
    @staticmethod
    def _is_valid_image(filename):
        """验证图像文件"""
        return filename.lower().endswith(('.jpeg', '.jpg', '.png'))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image_name = self.image_names[index]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"图像读取失败 {image_path}: {e}")
            image = Image.new('RGB', Config.IMAGE_SIZE, color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, image_name


class TransformManager:
    """图像变换管理器"""
    
    @staticmethod
    def get_train_transforms():
        """获取训练集变换"""
        aug_config = Config.AUGMENTATION_CONFIG
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=aug_config['random_flip_prob']),
            transforms.RandomRotation(aug_config['random_rotation']),
            transforms.RandomAffine(degrees=0, translate=aug_config['translate']),
            transforms.ColorJitter(
                brightness=aug_config['brightness_jitter'],
                contrast=aug_config['contrast_jitter']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
        ])
    
    @staticmethod
    def get_val_transforms():
        """获取验证集变换"""
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
        ])


class DataLoaderManager:
    """数据加载器管理器"""
    
    @staticmethod
    def create_data_loaders(data_root):
        """创建数据加载器"""
        train_transform = TransformManager.get_train_transforms()
        val_transform = TransformManager.get_val_transforms()
        
        # 创建数据集
        train_dataset = MedicalImageDataset(
            os.path.join(data_root, 'train'), 
            train_transform
        )
        val_dataset = MedicalImageDataset(
            os.path.join(data_root, 'val'), 
            val_transform
        )
        test_dataset = MedicalImageDataset(
            os.path.join(data_root, 'test'), 
            val_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset