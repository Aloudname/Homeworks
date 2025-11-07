"""绘图可视化模块"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config.settings import Config


class PlotManager:
    """绘图管理器"""
    
    @staticmethod
    def plot_confusion_matrix(true_labels, predictions, class_names=Config.CLASS_NAMES):
        """绘制混淆矩阵"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, 
                   yticklabels=class_names,
                   cbar_kws={'label': '样本数量'})
        plt.title('模型混淆矩阵分析')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    @staticmethod
    def plot_class_distribution(train_dataset, val_dataset, test_dataset):
        """绘制类别分布"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        datasets = [train_dataset, val_dataset, test_dataset]
        dataset_names = ['训练数据集', '验证数据集', '测试数据集']
        
        for idx, (dataset, name) in enumerate(zip(datasets, dataset_names)):
            if len(dataset) == 0:
                continue
                
            class_counts = [dataset.labels.count(0), dataset.labels.count(1)]
            
            bars = axes[idx].bar(Config.CLASS_NAMES, class_counts, 
                               color=['#66b3ff', '#ff9999'], alpha=0.8)
            axes[idx].set_title(f'{name}分布情况')
            axes[idx].set_ylabel('样本数量')
            
            # 添加数值标签
            for bar, count in zip(bars, class_counts):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                              str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_sample_predictions(test_loader, predictions, true_labels, image_names, num_samples=12):
        """绘制样本预测结果"""
        correct_indices = [i for i, (pred, label) in enumerate(zip(predictions, true_labels)) 
                          if pred == label]
        incorrect_indices = [i for i, (pred, label) in enumerate(zip(predictions, true_labels)) 
                           if pred != label]
        
        print(f"正确分类样本数: {len(correct_indices)}")
        print(f"错误分类样本数: {len(incorrect_indices)}")
        
        # 可视化正确分类的样本
        if correct_indices:
            PlotManager._plot_samples(test_loader, correct_indices, predictions, 
                                    true_labels, image_names, "正确分类样本", 
                                    min(num_samples, len(correct_indices)))
        
        # 可视化错误分类的样本
        if incorrect_indices:
            PlotManager._plot_samples(test_loader, incorrect_indices, predictions, 
                                    true_labels, image_names, "错误分类样本", 
                                    min(num_samples, len(incorrect_indices)))
    
    @staticmethod
    def _plot_samples(test_loader, indices, predictions, true_labels, image_names, title, num_samples):
        """绘制样本"""
        if len(indices) > num_samples:
            selected_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            selected_indices = indices
        
        # 获取原始图像
        original_images = []
        for idx in selected_indices:
            img_path = test_loader.dataset.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            original_images.append(image)
        
        # 创建可视化
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(selected_indices):
            if i >= len(axes):
                break
            
            image = original_images[i]
            true_label = true_labels[idx]
            pred_label = predictions[idx]
            img_name = image_names[idx]
            
            # 显示图像
            axes[i].imshow(image)
            
            # 设置标题颜色
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'{img_name}\n真实: {Config.CLASS_NAMES[true_label]}\n预测: {Config.CLASS_NAMES[pred_label]}',
                              color=color, fontsize=10)
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(selected_indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}/{title}.png', dpi=300, bbox_inches='tight')
        plt.show()