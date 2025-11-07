"""工具函数模块"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config.settings import Config


class RandomStateManager:
    """随机状态管理器"""
    
    @staticmethod
    def set_seed(seed=42):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"随机种子设置为: {seed}")

    @staticmethod
    def get_random_state():
        """获取当前随机状态"""
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }

    @staticmethod
    def set_random_state(state):
        """设置随机状态"""
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda'])


class FileManager:
    """文件管理器"""
    
    @staticmethod
    def ensure_directory(path):
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_file_extension(filename):
        """获取文件扩展名"""
        return os.path.splitext(filename)[1].lower()

    @staticmethod
    def is_image_file(filename):
        """检查是否为图像文件"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return FileManager.get_file_extension(filename) in valid_extensions

    @staticmethod
    def find_image_files(directory):
        """查找目录中的所有图像文件"""
        image_files = []
        for filename in os.listdir(directory):
            if FileManager.is_image_file(filename):
                image_files.append(os.path.join(directory, filename))
        return sorted(image_files)

    @staticmethod
    def save_tensor_as_image(tensor, filepath, denormalize=True):
        """将张量保存为图像"""
        # 将张量转换为numpy数组
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        if denormalize:
            # 反标准化
            mean = torch.tensor(Config.IMAGENET_MEAN).view(-1, 1, 1)
            std = torch.tensor(Config.IMAGENET_STD).view(-1, 1, 1)
            tensor = tensor * std + mean
            tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像并保存
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一个样本
        if tensor.dim() == 3 and tensor.size(0) == 3:
            tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
        
        tensor_np = (tensor.numpy() * 255).astype(np.uint8)
        image = Image.fromarray(tensor_np)
        image.save(filepath)
        return filepath


class ImageProcessor:
    """图像处理器"""
    
    @staticmethod
    def pil_to_tensor(pil_image, normalize=True):
        """将PIL图像转换为张量"""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD) 
            if normalize else transforms.Lambda(lambda x: x)
        ])
        return transform(pil_image).unsqueeze(0)  # 添加batch维度

    @staticmethod
    def tensor_to_pil(tensor, denormalize=True):
        """将张量转换为PIL图像"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        if denormalize:
            # 反标准化
            mean = torch.tensor(Config.IMAGENET_MEAN).view(-1, 1, 1)
            std = torch.tensor(Config.IMAGENET_STD).view(-1, 1, 1)
            tensor = tensor * std + mean
            tensor = torch.clamp(tensor, 0, 1)
        
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一个样本
        
        tensor_np = tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        tensor_np = (tensor_np * 255).astype(np.uint8)
        return Image.fromarray(tensor_np)

    @staticmethod
    def resize_image(image, size=Config.IMAGE_SIZE):
        """调整图像尺寸"""
        if isinstance(image, torch.Tensor):
            image = ImageProcessor.tensor_to_pil(image)
        
        return image.resize(size, Image.BILINEAR)

    @staticmethod
    def calculate_image_statistics(dataset):
        """计算图像数据集的统计信息"""
        pixel_values = []
        for image, _, _ in dataset:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            pixel_values.append(image.flatten())
        
        all_pixels = np.concatenate(pixel_values)
        return {
            'mean': np.mean(all_pixels),
            'std': np.std(all_pixels),
            'min': np.min(all_pixels),
            'max': np.max(all_pixels),
            'shape': dataset[0][0].shape if len(dataset) > 0 else None
        }


class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def count_parameters(model):
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    @staticmethod
    def print_model_summary(model, input_size=(3, 224, 224)):
        """打印模型摘要"""
        try:
            from torchsummary import summary
            summary(model, input_size=input_size, device=Config.DEVICE.type)
        except ImportError:
            print("torchsummary未安装，使用简化版摘要")
            total_params, trainable_params = ModelUtils.count_parameters(model)
            print(f"总参数: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            print(f"模型结构: {model}")

    @staticmethod
    def save_model_checkpoint(model, optimizer, epoch, metrics, filepath):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'model_type': Config.MODEL_TYPE,
                'num_classes': Config.NUM_CLASSES,
                'image_size': Config.IMAGE_SIZE
            }
        }
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath} (周期 {epoch})")

    @staticmethod
    def load_model_checkpoint(model, optimizer=None, filepath='checkpoint.pth'):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"检查点已加载: {filepath} (周期 {checkpoint['epoch']})")
        return checkpoint['epoch'], checkpoint['metrics']


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_classification_metrics(true_labels, predictions, probabilities=None):
        """计算分类指标"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(true_labels, predictions)
        }
        
        # 计算AUC（如果有概率值）
        if probabilities is not None and len(np.unique(true_labels)) == 2:
            try:
                metrics['auc'] = roc_auc_score(true_labels, probabilities[:, 1])
            except Exception as e:
                print(f"无法计算AUC: {e}")
                metrics['auc'] = 0.0
        
        return metrics

    @staticmethod
    def print_detailed_metrics(metrics, class_names=Config.CLASS_NAMES):
        """打印详细指标"""
        print("=" * 60)
        print("详细性能指标")
        print("=" * 60)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        if 'auc' in metrics:
            print(f"AUC面积: {metrics['auc']:.4f}")
        
        print("\n混淆矩阵:")
        print(metrics['confusion_matrix'])

    @staticmethod
    def calculate_model_efficiency(model, input_size=(1, 3, 224, 224)):
        """计算模型效率"""
        import time
        
        # 移动到设备
        model = model.to(Config.DEVICE)
        model.eval()
        
        # 创建虚拟输入
        dummy_input = torch.randn(input_size).to(Config.DEVICE)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        
        # 计算内存使用（近似）
        if Config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_allocated = -1  # CPU模式下无法准确测量
        
        return {
            'avg_inference_time_ms': avg_inference_time,
            'memory_usage_mb': memory_allocated,
            'fps': 1000 / avg_inference_time if avg_inference_time > 0 else 0
        }


class VisualizationUtils:
    """可视化工具"""
    
    @staticmethod
    def create_comparison_figure(images, titles, figsize=(15, 5)):
        """创建对比图"""
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
        
        for ax, image, title in zip(axes, images, titles):
            if isinstance(image, torch.Tensor):
                image = ImageProcessor.tensor_to_pil(image)
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confidence_distribution(probabilities, true_labels, class_names=Config.CLASS_NAMES):
        """绘制置信度分布"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 正确预测的置信度
        correct_probs = []
        incorrect_probs = []
        
        for prob, true_label, pred_label in zip(probabilities, true_labels, 
                                               np.argmax(probabilities, axis=1)):
            confidence = prob[pred_label]
            if pred_label == true_label:
                correct_probs.append(confidence)
            else:
                incorrect_probs.append(confidence)
        
        # 绘制直方图
        axes[0].hist([correct_probs, incorrect_probs], bins=20, alpha=0.7, 
                    label=['正确预测', '错误预测'], color=['green', 'red'])
        axes[0].set_xlabel('置信度')
        axes[0].set_ylabel('频次')
        axes[0].set_title('预测置信度分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制箱线图
        axes[1].boxplot([correct_probs, incorrect_probs], labels=['正确', '错误'])
        axes[1].set_ylabel('置信度')
        axes[1].set_title('置信度统计比较')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def create_metric_radar_chart(metrics, max_values=None):
        """创建指标雷达图"""
        import matplotlib.pyplot as plt
        from math import pi
        
        if max_values is None:
            max_values = {
                'accuracy': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'auc': 1.0
            }
        
        # 筛选可用的指标
        available_metrics = {k: v for k, v in metrics.items() 
                           if k in max_values and v is not None}
        
        if not available_metrics:
            print("没有可用的指标数据")
            return None
        
        categories = list(available_metrics.keys())
        values = list(available_metrics.values())
        max_vals = [max_values[k] for k in categories]
        
        # 角度计算
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]  # 闭合图形
        values += values[:1]
        max_vals += max_vals[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # 绘制最大值的参考线
        ax.plot(angles, max_vals, 'o-', linewidth=2, label='最大值', color='gray', alpha=0.3)
        ax.fill(angles, max_vals, alpha=0.1, color='gray')
        
        # 绘制实际值
        ax.plot(angles, values, 'o-', linewidth=2, label='模型性能', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', size=16, y=1.1)
        ax.legend(loc='upper right')
        
        return fig


class Timer:
    """计时器类"""
    
    def __init__(self, name="操作"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available() and hasattr(self.start_time, 'record'):
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = self.start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
        else:
            import time
            elapsed_time = time.time() - self.start_time
        
        print(f"{self.name} 耗时: {elapsed_time:.4f} 秒")