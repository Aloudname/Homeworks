"""类激活图可视化模块"""
import numpy as np
import torch
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

from config.settings import Config


class ActivationMapGenerator:
    """类激活图生成器"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate_activation_map(self, input_tensor, target_class=None):
        """生成类激活图"""
        model_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        # 创建目标梯度
        target_gradient = torch.zeros_like(model_output)
        target_gradient[0][target_class] = 1
        
        # 反向传播
        model_output.backward(gradient=target_gradient)
        
        # 获取特征和梯度
        feature_maps = self.model.activation_maps
        gradient_maps = self.model.gradient_maps
        
        # 计算权重
        weights = torch.mean(gradient_maps, dim=(2, 3), keepdim=True)
        
        # 生成激活图
        activation_map = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        activation_map = torch.relu(activation_map)
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        
        return activation_map.squeeze().detach().cpu().numpy(), model_output, target_class
    
    def overlay_activation_map(self, original_image, activation_map, alpha=0.5):
        """叠加激活图到原图"""
        # 调整激活图尺寸
        activation_pil = Image.fromarray(np.uint8(255 * activation_map))
        resized_activation = activation_pil.resize(original_image.size, Image.BILINEAR)
        resized_activation = np.array(resized_activation) / 255.0
        
        # 处理原图
        original_array = np.array(original_image) / 255.0
        
        # 创建热力图
        heatmap = cm.jet(resized_activation)[:, :, :3]
        
        # 叠加图像
        blended_image = (1 - alpha) * original_array + alpha * heatmap
        blended_image = np.clip(blended_image, 0, 1)
        
        return blended_image, heatmap


class CAMVisualizer:
    """CAM可视化器"""
    
    def __init__(self, model):
        self.generator = ActivationMapGenerator(model)
        self.class_names = Config.CLASS_NAMES
    
    def visualize_samples(self, test_dataset, predictions, true_labels, image_names, num_samples=6):
        """可视化样本的CAM"""
        correct_indices = [i for i, (pred, label) in enumerate(zip(predictions, true_labels)) 
                          if pred == label]
        incorrect_indices = [i for i, (pred, label) in enumerate(zip(predictions, true_labels)) 
                           if pred != label]

        print(f"正确分类样本: {len(correct_indices)}")
        print(f"错误分类样本: {len(incorrect_indices)}")

        selected_indices = self._select_samples(correct_indices, incorrect_indices, num_samples)
        
        for idx in selected_indices:
            self._visualize_single_sample(test_dataset, idx, predictions, true_labels, image_names)
    
    def _select_samples(self, correct_indices, incorrect_indices, num_samples):
        """选择可视化样本"""
        selected = []
        if correct_indices:
            correct_count = min(num_samples // 2, len(correct_indices))
            selected.extend(np.random.choice(correct_indices, correct_count, replace=False))
        if incorrect_indices:
            incorrect_count = min(num_samples // 2, len(incorrect_indices))
            selected.extend(np.random.choice(incorrect_indices, incorrect_count, replace=False))
        return selected
    
    def _visualize_single_sample(self, test_dataset, index, predictions, true_labels, image_names):
        """可视化单个样本"""
        image_path = test_dataset.image_paths[index]
        original_image = Image.open(image_path).convert('RGB')
        
        # 预处理图像
        from data.dataset import TransformManager
        transform = TransformManager.get_val_transforms()
        processed_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(processed_image).unsqueeze(0).to(Config.DEVICE)
        
        # 生成激活图
        activation_map, output, pred_class = self.generator.generate_activation_map(input_tensor)
        blended_image, heatmap = self.generator.overlay_activation_map(original_image, activation_map)
        
        true_label = true_labels[index]
        pred_label = predictions[index]
        image_id = image_names[index]
        
        self._create_visualization(original_image, heatmap, blended_image, 
                                 activation_map, true_label, pred_label, image_id)
        self._analyze_activation_pattern(activation_map, pred_label, image_id)
    
    def _create_visualization(self, original, heatmap, blended, activation_map, 
                            true_label, pred_label, image_id):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        visualization_data = [
            (original, f'原始医学影像\n真实类别: {self.class_names[true_label]}'),
            (heatmap, '类激活热力图'),
            (blended, f'叠加可视化\n预测类别: {self.class_names[pred_label]}'),
            (activation_map, '原始激活图', 'jet')
        ]
        
        for idx, (data, title, *cmap) in enumerate(visualization_data):
            ax = axes[idx//2, idx%2]
            if cmap:
                ax.imshow(data, cmap=cmap[0])
            else:
                ax.imshow(data)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        title_color = 'green' if true_label == pred_label else 'red'
        fig.suptitle(f'影像分析: {image_id}\n真实: {self.class_names[true_label]} | 预测: {self.class_names[pred_label]}',
                    fontsize=14, color=title_color)
        
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}/cam_{image_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_activation_pattern(self, activation_map, prediction, image_id):
        """分析激活模式"""
        flat_activation = activation_map.flatten()
        high_activation_threshold = np.percentile(flat_activation, 90)
        high_activation_ratio = np.sum(flat_activation > high_activation_threshold) / len(flat_activation)
        
        print(f"\n影像分析: {image_id}")
        print(f"高激活区域比例: {high_activation_ratio:.4f}")
        
        if prediction == 1:  # 肺炎
            if high_activation_ratio > 0.05:
                print("诊断分析: 检测到明显的肺炎特征区域")
            else:
                print("诊断分析: 特征激活分散，肺炎特征不明显")
        else:  # 正常
            if high_activation_ratio > 0.05:
                print("诊断分析: 发现局部激活，建议进一步检查")
            else:
                print("诊断分析: 激活分布正常，符合健康特征")
        print("-" * 50)