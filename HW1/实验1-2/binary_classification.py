import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

class BinaryDatasetGenerator:
    """二分类数据集生成器"""
    def __init__(self, total_samples=10000, train_ratio=0.7, feature_dim=200, std=0.28**0.5):
        self.total_samples = total_samples
        self.train_size = int(total_samples * train_ratio)
        self.test_size = total_samples - self.train_size
        self.feature_dim = feature_dim
        self.std = std
    
    def create_class_data(self, mean_value, label_value):
        """创建单类别数据集"""
        features = torch.normal(mean_value, self.std, 
                               size=(self.total_samples, self.feature_dim), 
                               dtype=torch.float32)
        labels = torch.full((self.total_samples,), label_value, dtype=torch.float32)
        
        # 随机打乱
        indices = torch.randperm(self.total_samples)
        features, labels = features[indices], labels[indices]
        
        # 分割训练测试集
        train_data = (features[:self.train_size], labels[:self.train_size])
        test_data = (features[self.train_size:], labels[self.train_size:])
        
        return train_data, test_data
    
    def generate_binary_data(self):
        """生成二分类数据集"""
        # 类别0: 均值-0.56
        class0_train, class0_test = self.create_class_data(-0.56, 0.0)
        # 类别1: 均值0.56
        class1_train, class1_test = self.create_class_data(0.56, 1.0)
        
        # 合并数据集
        X_train = torch.cat([class0_train[0], class1_train[0]], 0)
        y_train = torch.cat([class0_train[1], class1_train[1]], 0)
        X_test = torch.cat([class0_test[0], class1_test[0]], 0)
        y_test = torch.cat([class0_test[1], class1_test[1]], 0)
        
        return (X_train, y_train), (X_test, y_test)

class BinaryClassifier(nn.Module):
    """二分类神经网络"""
    def __init__(self, input_dim=200, hidden_dim=64, dropout_rate=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

class BinaryClassificationTrainer:
    """二分类训练器"""
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def configure_optimization(self, lr=0.1, weight_decay=1e-3):
        """配置优化策略"""
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    def train_single_epoch(self):
        """单轮训练"""
        self.model.train()
        epoch_loss, processed_samples = 0.0, 0
        
        for features, targets in self.train_loader:
            features, targets = features.to(self.device), targets.float().to(self.device)
            
            logits = self.model(features)
            loss = self.loss_fn(logits, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_size = targets.shape[0]
            epoch_loss += loss.item() * batch_size
            processed_samples += batch_size
        
        return epoch_loss / processed_samples
    
    def validate_model(self):
        """模型验证"""
        self.model.eval()
        all_targets, all_scores = [], []
        total_loss = 0.0
        
        with torch.no_grad():
            for features, targets in self.test_loader:
                features, targets = features.to(self.device), targets.float().to(self.device)
                logits = self.model(features)
                total_loss += self.loss_fn(logits, targets).item() * targets.shape[0]
                
                all_targets.append(targets.cpu())
                all_scores.append(torch.sigmoid(logits).cpu())
        
        # 计算指标
        true_labels = torch.cat(all_targets).numpy()
        predicted_scores = torch.cat(all_scores).numpy()
        predicted_labels = (predicted_scores > 0.5).astype(int)
        
        avg_loss = total_loss / len(self.test_loader.dataset)
        auc_score = roc_auc_score(true_labels, predicted_scores)
        f1 = f1_score(true_labels, predicted_labels)
        
        return avg_loss, auc_score, f1
    
    def run_training(self, epochs=150, lr=0.1, weight_decay=1e-3, patience=15, plot=True):
        """执行训练流程"""
        self.configure_optimization(lr, weight_decay)
        
        train_losses, test_losses = [], []
        best_auc, stale_count = 0.0, 0
        best_weights = None
        
        for epoch_idx in range(epochs):
            # 训练和验证
            train_loss = self.train_single_epoch()
            test_loss, auc, f1 = self.validate_model()
            
            # 记录损失
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # 学习率调整
            self.lr_scheduler.step(test_loss)
            
            # 早停判断
            if auc > best_auc:
                best_auc, stale_count = auc, 0
                best_weights = self.model.state_dict().copy()
            else:
                stale_count += 1
                if stale_count >= patience:
                    print(f'训练提前终止于第 {epoch_idx + 1} 轮')
                    break
            
            # 训练信息输出
            if (epoch_idx + 1) % 10 == 0 or epoch_idx < 10:
                print(f'轮次 {epoch_idx + 1:>3}: 训练损失={train_loss:.4f}, '
                      f'测试损失={test_loss:.4f}, AUC={auc:.4f}, F1={f1:.4f}')
        
        # 加载最佳权重
        self.model.load_state_dict(best_weights)
        
        if plot:
            self._visualize_progress(train_losses, test_losses)
        
        return train_losses, test_losses, best_weights
    
    def _visualize_progress(self, train_losses, test_losses):
        """可视化训练进度"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', linewidth=2)
        plt.plot(test_losses, label='验证损失', linewidth=2)
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('二元交叉熵损失', fontsize=12)
        plt.title('二分类模型训练过程', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def execute_binary_classification():
    """执行二分类任务"""
    # 设置随机种子
    torch.manual_seed(42)
    
    # 生成数据
    data_generator = BinaryDatasetGenerator()
    (X_train, y_train), (X_test, y_test) = data_generator.generate_binary_data()
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # 初始化模型和训练器
    classifier = BinaryClassifier(input_dim=200, hidden_dim=64, dropout_rate=0.2)
    trainer = BinaryClassificationTrainer(classifier, train_loader, test_loader)
    
    # 开始训练
    results = trainer.run_training(epochs=150, lr=0.1, weight_decay=1e-3, patience=15)
    
    return results

if __name__ == "__main__":
    execute_binary_classification()