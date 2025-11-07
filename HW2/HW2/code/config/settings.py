"""配置文件管理"""
import os
import torch

class Config:
    """配置类"""
    
    # 路径配置
    DATA_ROOT = r"E:\\HW2\\X-ray肺炎检测数据集"
    OUTPUT_DIR = "./output"
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 训练配置
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # 模型配置
    MODEL_TYPE = 'resnet50'
    NUM_CLASSES = 2
    PRETRAINED = True
    
    # 图像处理配置
    IMAGE_SIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # 数据增强配置
    AUGMENTATION_CONFIG = {
        'random_flip_prob': 0.5,
        'random_rotation': 10,
        'translate': (0.1, 0.1),
        'brightness_jitter': 0.2,
        'contrast_jitter': 0.2
    }
    
    # 类别配置
    CLASS_NAMES = ['正常', '肺炎']
    CLASS_MAPPING = {'normal': 0, 'pneumonia': 1}
    CLASS_WEIGHTS = [2.0, 1.0]  # 用于处理类别不平衡
    
    @classmethod
    def setup_environment(cls):
        """设置运行环境"""
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        import warnings
        import matplotlib.pyplot as plt
        warnings.filterwarnings('ignore')
        plt.rcParams['font.family'] = 'SimHei'
        
        # 创建输出目录
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        print(f"运行设备: {cls.DEVICE}")
        print(f"输出目录: {cls.OUTPUT_DIR}")