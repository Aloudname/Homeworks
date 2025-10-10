import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class RegressionDatasetGenerator:
    """回归数据集生成器"""
    def __init__(self, n_train=7000, n_test=3000, num_inputs=500, noise_std=0.056):
        self.n_train = n_train
        self.n_test = n_test
        self.num_inputs = num_inputs
        self.noise_std = noise_std
        self.true_w = torch.ones(num_inputs, 1) * 0.056
        self.true_b = 0.028
    
    def generate_data(self):
        """生成回归数据"""
        total_samples = self.n_train + self.n_test
        features = torch.randn((total_samples, self.num_inputs))
        labels = torch.matmul(features, self.true_w) + self.true_b
        noise = torch.tensor(np.random.normal(0, self.noise_std, size=labels.size()), dtype=torch.float)
        labels += noise
        
        # 分割数据集
        train_data = (features[:self.n_train], labels[:self.n_train])
        test_data = (features[self.n_train:], labels[self.n_train:])
        
        return train_data, test_data
    
    def save_data(self, filename='regression_data.pt'):
        """保存生成的数据"""
        train_data, test_data = self.generate_data()
        data_dict = {
            'train_features': train_data[0],
            'train_labels': train_data[1],
            'test_features': test_data[0],
            'test_labels': test_data[1]
        }
        torch.save(data_dict, filename)
        return train_data, test_data

class RegressionNetwork(nn.Module):
    """回归任务神经网络"""
    def __init__(self, input_dim=500, hidden_dim=256, dropout_rate=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden_layer(x))
        x = self.dropout(x)
        return self.output_layer(x).squeeze(-1)

class RegressionTrainer:
    """回归模型训练器"""
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
        self.loss_function = nn.MSELoss()
    
    def setup_optimizer(self, learning_rate=0.1, weight_decay=1e-3):
        """配置优化器和学习率调度器"""
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss, sample_count = 0.0, 0
        
        for batch_features, batch_labels in self.train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.squeeze().to(self.device)
            
            predictions = self.model(batch_features)
            batch_loss = self.loss_function(predictions, batch_labels)
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item() * batch_labels.shape[0]
            sample_count += batch_labels.shape[0]
        
        return total_loss / sample_count
    
    def evaluate(self):
        """在测试集上评估模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_labels in self.test_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                predictions = self.model(batch_features)
                total_loss += self.loss_function(predictions, batch_labels).item() * batch_labels.shape[0]
        
        return total_loss / len(self.test_loader.dataset)
    
    def execute_training(self, epochs=300, learning_rate=0.1, weight_decay=1e-3, patience=15, plot_results=True):
        """执行完整训练过程"""
        self.setup_optimizer(learning_rate, weight_decay)
        
        train_losses, test_losses = [], []
        best_loss, stale_epochs = float('inf'), 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练和评估
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            
            # 记录损失
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # 学习率调整
            self.scheduler.step(test_loss)
            
            # 早停机制
            if test_loss < best_loss:
                best_loss, stale_epochs = test_loss, 0
                best_model_state = self.model.state_dict().copy()
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f'训练提前停止于第 {epoch + 1} 轮')
                    break
            
            # 进度输出
            if (epoch + 1) % 10 == 0 or epoch < 10:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'轮次 {epoch + 1:>3}: 训练损失={train_loss:.6f}, '
                      f'测试损失={test_loss:.6f}, 学习率={current_lr:.4f}')
        
        # 恢复最佳模型
        self.model.load_state_dict(best_model_state)
        
        if plot_results:
            self._plot_training_curve(train_losses, test_losses)
        
        return train_losses, test_losses, best_model_state
    
    def _plot_training_curve(self, train_losses, test_losses):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', linewidth=2)
        plt.plot(test_losses, label='测试损失', linewidth=2)
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('均方误差损失', fontsize=12)
        plt.title('回归模型训练过程', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def run_regression_experiment():
    """运行回归实验"""
    # 生成数据
    generator = RegressionDatasetGenerator()
    train_data, test_data = generator.save_data('regression_data.pt')
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_data[0], train_data[1])
    test_dataset = TensorDataset(test_data[0], test_data[1])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # 初始化模型和训练器
    regression_model = RegressionNetwork(input_dim=500)
    trainer = RegressionTrainer(regression_model, train_loader, test_loader)
    
    # 开始训练
    training_results = trainer.execute_training(epochs=50, learning_rate=0.1)
    
    return training_results

if __name__ == "__main__":
    results = run_regression_experiment()