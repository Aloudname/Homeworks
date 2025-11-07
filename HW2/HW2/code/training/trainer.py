"""训练模块"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from config.settings import Config


class TrainingMetrics:
    """训练指标记录器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
    
    def update(self, train_metrics, val_metrics):
        """更新指标"""
        self.train_loss.append(train_metrics['loss'])
        self.train_accuracy.append(train_metrics['accuracy'])
        self.val_loss.append(val_metrics['loss'])
        self.val_accuracy.append(val_metrics['accuracy'])
        self.val_precision.append(val_metrics['precision'])
        self.val_recall.append(val_metrics['recall'])
        self.val_f1.append(val_metrics['f1'])


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = TrainingMetrics()
        
        # 优化器和损失函数
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(Config.CLASS_WEIGHTS).to(Config.DEVICE))
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        self.best_accuracy = 0.0
    
    def train_epoch(self):
        """训练一个周期"""
        self.model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_data, batch_labels, _ in self.train_loader:
            batch_data = batch_data.to(Config.DEVICE)
            batch_labels = batch_labels.to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_data)
            _, predictions = torch.max(outputs, 1)
            loss = self.criterion(outputs, batch_labels)
            
            loss.backward()
            self.optimizer.step()
            
            batch_size = batch_data.size(0)
            epoch_loss += loss.item() * batch_size
            correct_predictions += torch.sum(predictions == batch_labels.data)
            total_samples += batch_size
        
        return {
            'loss': epoch_loss / total_samples,
            'accuracy': correct_predictions.double() / total_samples
        }
    
    def validate_epoch(self):
        """验证一个周期"""
        self.model.eval()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels, _ in self.val_loader:
                batch_data = batch_data.to(Config.DEVICE)
                batch_labels = batch_labels.to(Config.DEVICE)
                
                outputs = self.model(batch_data)
                _, predictions = torch.max(outputs, 1)
                loss = self.criterion(outputs, batch_labels)
                
                batch_size = batch_data.size(0)
                epoch_loss += loss.item() * batch_size
                correct_predictions += torch.sum(predictions == batch_labels.data)
                total_samples += batch_size
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        accuracy = correct_predictions.double() / total_samples
        
        return {
            'loss': epoch_loss / total_samples,
            'accuracy': accuracy,
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0)
        }
    
    def train(self, num_epochs=Config.NUM_EPOCHS):
        """执行训练"""
        print("开始模型训练...")
        
        for epoch in range(num_epochs):
            print(f'训练周期 {epoch + 1}/{num_epochs}')
            print('-' * 50)
            
            # 训练和验证
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            # 更新指标
            self.metrics.update(train_metrics, val_metrics)
            
            # 打印结果
            self._print_epoch_results(train_metrics, val_metrics)
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self._save_best_model()
        
        # 可视化训练过程
        self._visualize_training()
        
        return self.model, self.metrics
    
    def _print_epoch_results(self, train_metrics, val_metrics):
        """打印周期结果"""
        print(f'训练损失: {train_metrics["loss"]:.4f} 准确率: {train_metrics["accuracy"]:.4f}')
        print(f'验证损失: {val_metrics["loss"]:.4f} 准确率: {val_metrics["accuracy"]:.4f}')
        print(f'验证精确率: {val_metrics["precision"]:.4f} 召回率: {val_metrics["recall"]:.4f} F1: {val_metrics["f1"]:.4f}')
        print()
    
    def _save_best_model(self):
        """保存最佳模型"""
        model_path = f"{Config.OUTPUT_DIR}/best_model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"保存最佳模型: {model_path}")
    
    def _visualize_training(self):
        """可视化训练过程"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.metrics.train_loss, label='训练损失')
        axes[0, 0].plot(self.metrics.val_loss, label='验证损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('周期')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot([acc.cpu() for acc in self.metrics.train_accuracy], label='训练准确率')
        axes[0, 1].plot([acc.cpu() for acc in self.metrics.val_accuracy], label='验证准确率')
        axes[0, 1].set_title('训练和验证准确率')
        axes[0, 1].set_xlabel('周期')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 精确率曲线
        axes[0, 2].plot(self.metrics.val_precision, label='验证精确率', color='orange')
        axes[0, 2].set_title('验证精确率')
        axes[0, 2].set_xlabel('周期')
        axes[0, 2].set_ylabel('精确率')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 召回率曲线
        axes[1, 0].plot(self.metrics.val_recall, label='验证召回率', color='green')
        axes[1, 0].set_title('验证召回率')
        axes[1, 0].set_xlabel('周期')
        axes[1, 0].set_ylabel('召回率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1分数曲线
        axes[1, 1].plot(self.metrics.val_f1, label='验证F1分数', color='red')
        axes[1, 1].set_title('验证F1分数')
        axes[1, 1].set_xlabel('周期')
        axes[1, 1].set_ylabel('F1分数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 隐藏最后一个子图
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()