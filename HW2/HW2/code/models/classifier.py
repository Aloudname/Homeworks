"""模型定义模块"""
import torch.nn as nn
from torchvision import models

from config.settings import Config


class MedicalImageClassifier(nn.Module):
    """医学影像分类器"""
    
    MODEL_CONFIGS = {
        'resnet18': (models.resnet18, 512),
        'resnet50': (models.resnet50, 2048),
        'resnet34': (models.resnet34, 512)
    }
    
    def __init__(self, model_type=Config.MODEL_TYPE, num_classes=Config.NUM_CLASSES):
        super(MedicalImageClassifier, self).__init__()
        
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_constructor, feature_dim = self.MODEL_CONFIGS[model_type]
        self.backbone = model_constructor(pretrained=Config.PRETRAINED)
        self.feature_layer = self._get_feature_layer(model_type)
        
        # 特征图记录
        self.activation_maps = None
        self.gradient_maps = None
        
        # 注册钩子
        self._register_hooks()
        
        # 重构分类器
        self.backbone.fc = self._build_classifier(feature_dim, num_classes)
        
    def _get_feature_layer(self, model_type):
        """获取特征层"""
        if model_type.startswith('resnet'):
            return self.backbone.layer4
        else:
            raise ValueError(f"未知模型类型的特征层: {model_type}")
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        self.feature_layer.register_forward_hook(self._forward_hook)
        self.feature_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """前向钩子"""
        self.activation_maps = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        """反向钩子"""
        self.gradient_maps = grad_output[0]
    
    @staticmethod
    def _build_classifier(input_dim, output_dim):
        """构建分类器"""
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        """获取特征图"""
        _ = self.forward(x)
        return self.activation_maps
    
    def get_gradient_maps(self):
        """获取梯度图"""
        return self.gradient_maps


class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_model(model_type=Config.MODEL_TYPE, num_classes=Config.NUM_CLASSES):
        """创建模型实例"""
        model = MedicalImageClassifier(model_type, num_classes)
        model = model.to(Config.DEVICE)
        print(f"创建模型: {model_type}, 类别数: {num_classes}")
        return model