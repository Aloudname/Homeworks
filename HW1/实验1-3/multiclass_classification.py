# [file name]: multiclass_activation_comparison.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import time
import pandas as pd

class MultiClassNetwork(nn.Module):
    """多分类神经网络模型（支持不同激活函数）"""
    def __init__(self, input_dim=784, hidden_dim=256, dropout_rate=0.2, 
                 num_classes=10, activation_fn='relu'):
        super().__init__()
        
        self.activation_name = activation_fn
        
        # 定义激活函数
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation_fn}")
        
        # 使用Sequential容器定义分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            self.activation,  # 使用指定的激活函数
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化 - 根据激活函数调整初始化策略"""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                # 根据激活函数选择不同的初始化方法
                if self.activation_name == 'relu':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif self.activation_name == 'sigmoid':
                    nn.init.xavier_uniform_(layer.weight)
                elif self.activation_name == 'tanh':
                    nn.init.xavier_uniform_(layer.weight)
                
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.classifier(x)

class MultiClassTrainer:
    """增强版多分类训练器（支持多种评估指标）"""
    def __init__(self, model, train_loader, test_loader, activation_name):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
        self.criterion = nn.CrossEntropyLoss()
        self.activation_name = activation_name
        
        # 记录训练历史
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.f1_scores = []
        self.learning_rates = []
    
    def setup_optimizer(self, lr=0.1, weight_decay=1e-3):
        """配置优化器"""
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=3, verbose=False)
    
    def train_epoch(self):
        """单轮训练"""
        self.model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = correct_predictions / total_samples
        average_loss = running_loss / total_samples
        
        return average_loss, accuracy
    
    def evaluate_model(self):
        """模型评估"""
        self.model.eval()
        all_labels, all_predictions = [], []
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                all_labels.append(labels.cpu())
                all_predictions.append(predicted.cpu())
        
        # 计算指标
        test_loss = running_loss / total_samples
        test_accuracy = correct_predictions / total_samples
        
        true_labels = torch.cat(all_labels)
        predicted_labels = torch.cat(all_predictions)
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        
        return test_loss, test_accuracy, f1, true_labels, predicted_labels
    
    def conduct_training(self, epochs=50, lr=0.1, weight_decay=1e-3, patience=10):
        """执行训练过程"""
        self.setup_optimizer(lr, weight_decay)
        
        best_accuracy, stale_count = 0.0, 0
        best_model_state = None
        training_start_time = time.time()
        
        print(f"\n开始训练 {self.activation_name.upper()} 激活函数模型...")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练和评估
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, f1, _, _ = self.evaluate_model()
            
            # 记录指标
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.f1_scores.append(f1)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # 学习率调整
            self.scheduler.step(test_loss)
            
            # 早停机制
            if test_acc > best_accuracy:
                best_accuracy, stale_count = test_acc, 0
                best_model_state = self.model.state_dict().copy()
            else:
                stale_count += 1
                if stale_count >= patience:
                    print(f'训练提前终止于第 {epoch + 1} 轮')
                    break
            
            # 输出训练信息
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % 10 == 0 or epoch < 5:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'轮次 {epoch + 1:>3}: 训练损失={train_loss:.4f}, '
                      f'训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}, '
                      f'F1分数={f1:.4f}, 学习率={current_lr:.4f}, 时间={epoch_time:.2f}s')
        
        # 恢复最佳模型
        self.model.load_state_dict(best_model_state)
        
        # 最终评估
        final_test_loss, final_test_acc, final_f1, true_labels, predicted_labels = self.evaluate_model()
        total_training_time = time.time() - training_start_time
        
        print(f"\n{self.activation_name.upper()} 模型训练完成!")
        print(f"最终测试准确率: {final_test_acc:.4f}")
        print(f"最终F1分数: {final_f1:.4f}")
        print(f"总训练时间: {total_training_time:.2f}秒")
        
        return {
            'activation': self.activation_name,
            'final_test_accuracy': final_test_acc,
            'final_f1_score': final_f1,
            'training_time': total_training_time,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'f1_scores': self.f1_scores,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }

def load_mnist_data(batch_size=64):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(
        root='./Datasets/MNIST', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./Datasets/MNIST', train=False, download=True, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 检查数据形状
    sample_batch = next(iter(train_loader))
    print(f"输入数据形状: {sample_batch[0].shape}")
    print(f"标签形状: {sample_batch[1].shape}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_loader, test_loader

def compare_activation_functions():
    """比较不同激活函数的性能"""
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    print("正在加载MNIST数据集...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # 定义要比较的激活函数
    activation_functions = ['relu', 'sigmoid', 'tanh']
    results = {}
    
    # 对每种激活函数进行训练和评估
    for activation in activation_functions:
        # 创建模型
        model = MultiClassNetwork(
            input_dim=28*28,
            hidden_dim=256,
            dropout_rate=0.2,
            num_classes=10,
            activation_fn=activation
        )
        
        # 创建训练器
        trainer = MultiClassTrainer(model, train_loader, test_loader, activation)
        
        # 训练模型
        result = trainer.conduct_training(epochs=50, lr=0.1, weight_decay=1e-3, patience=10)
        results[activation] = result
    
    # 比较结果
    compare_results(results)
    
    # 可视化比较结果
    visualize_comparison(results)
    
    return results

def compare_results(results):
    """比较并打印不同激活函数的结果"""
    print("\n" + "="*70)
    print("激活函数性能对比")
    print("="*70)
    
    comparison_data = []
    for activation, result in results.items():
        comparison_data.append({
            '激活函数': activation.upper(),
            '测试准确率': f"{result['final_test_accuracy']:.4f}",
            'F1分数': f"{result['final_f1_score']:.4f}",
            '训练时间(秒)': f"{result['training_time']:.2f}",
            '最佳轮次': len(result['train_losses'])
        })
    
    # 创建对比表格
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # 找出最佳性能
    best_accuracy = max(results.items(), key=lambda x: x[1]['final_test_accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['final_f1_score'])
    fastest = min(results.items(), key=lambda x: x[1]['training_time'])
    
    print(f"\n最佳准确率: {best_accuracy[0].upper()} ({best_accuracy[1]['final_test_accuracy']:.4f})")
    print(f"最佳F1分数: {best_f1[0].upper()} ({best_f1[1]['final_f1_score']:.4f})")
    print(f"最快训练: {fastest[0].upper()} ({fastest[1]['training_time']:.2f}秒)")

def visualize_comparison(results):
    """可视化比较结果"""
    # 设置图形样式
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练和测试损失对比
    for activation, result in results.items():
        axes[0, 0].plot(result['train_losses'], label=f'{activation.upper()} 训练损失', linewidth=2)
        axes[0, 0].plot(result['test_losses'], label=f'{activation.upper()} 测试损失', 
                       linestyle='--', linewidth=2)
    
    axes[0, 0].set_xlabel('训练轮次', fontsize=12)
    axes[0, 0].set_ylabel('损失值', fontsize=12)
    axes[0, 0].set_title('不同激活函数的训练和测试损失对比', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 训练和测试准确率对比
    for activation, result in results.items():
        axes[0, 1].plot(result['train_accuracies'], label=f'{activation.upper()} 训练准确率', linewidth=2)
        axes[0, 1].plot(result['test_accuracies'], label=f'{activation.upper()} 测试准确率', 
                       linestyle='--', linewidth=2)
    
    axes[0, 1].set_xlabel('训练轮次', fontsize=12)
    axes[0, 1].set_ylabel('准确率', fontsize=12)
    axes[0, 1].set_title('不同激活函数的训练和测试准确率对比', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1分数对比
    for activation, result in results.items():
        axes[1, 0].plot(result['f1_scores'], label=f'{activation.upper()}', linewidth=2)
    
    axes[1, 0].set_xlabel('训练轮次', fontsize=12)
    axes[1, 0].set_ylabel('F1分数', fontsize=12)
    axes[1, 0].set_title('不同激活函数的F1分数对比', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 最终性能条形图
    activations = list(results.keys())
    accuracies = [results[act]['final_test_accuracy'] for act in activations]
    f1_scores = [results[act]['final_f1_score'] for act in activations]
    
    x = np.arange(len(activations))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, accuracies, width, label='测试准确率', alpha=0.8)
    axes[1, 1].bar(x + width/2, f1_scores, width, label='F1分数', alpha=0.8)
    
    axes[1, 1].set_xlabel('激活函数', fontsize=12)
    axes[1, 1].set_ylabel('分数', fontsize=12)
    axes[1, 1].set_title('不同激活函数的最终性能对比', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([act.upper() for act in activations])
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 在条形图上添加数值标签
    for i, v in enumerate(accuracies):
        axes[1, 1].text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(f1_scores):
        axes[1, 1].text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('activation_function_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制最佳模型的混淆矩阵
    best_activation = max(results.items(), key=lambda x: x[1]['final_test_accuracy'])[0]
    plot_confusion_matrix(results[best_activation]['true_labels'], 
                         results[best_activation]['predicted_labels'],
                         best_activation)

def plot_confusion_matrix(true_labels, predicted_labels, activation_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title(f'{activation_name.upper()} 激活函数的混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{activation_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印分类报告
    print(f"\n{activation_name.upper()} 激活函数的详细分类报告:")
    print(classification_report(true_labels, predicted_labels, digits=4))

def analyze_activation_characteristics():
    """分析不同激活函数的特性"""
    print("\n" + "="*70)
    print("激活函数特性分析")
    print("="*70)
    
    characteristics = {
        'ReLU': {
            '公式': 'f(x) = max(0, x)',
            '优点': '计算简单，缓解梯度消失，稀疏激活',
            '缺点': '神经元死亡问题，输出不是零中心',
            '适用场景': '大多数前馈神经网络，特别是深层网络'
        },
        'Sigmoid': {
            '公式': 'f(x) = 1 / (1 + exp(-x))',
            '优点': '输出范围(0,1)，平滑梯度',
            '缺点': '梯度消失问题，输出不是零中心，计算较慢',
            '适用场景': '二分类输出层，需要概率输出的场景'
        },
        'Tanh': {
            '公式': 'f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))',
            '优点': '输出范围(-1,1)，零中心，比sigmoid梯度更强',
            '缺点': '仍然存在梯度消失问题',
            '适用场景': '需要零中心输出的隐藏层'
        }
    }
    
    for activation, props in characteristics.items():
        print(f"\n{activation}:")
        for key, value in props.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # 运行激活函数对比实验
    results = compare_activation_functions()
    
    # 分析激活函数特性
    analyze_activation_characteristics()
    
    # 保存结果
    torch.save(results, 'activation_comparison_results.pth')
    print("\n对比结果已保存到 'activation_comparison_results.pth'")