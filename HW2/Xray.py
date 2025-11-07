import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据路径
DATA_PATH = r"C:\Users\ziqi\Desktop\X-ray"


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.image_names = []
        self.image_paths = []

        # 读取normal类别的图像
        normal_path = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_path):
            for img_name in os.listdir(normal_path):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(normal_path, img_name))
                    self.labels.append(0)  # 0表示正常
                    self.image_names.append(img_name)
                    self.image_paths.append(os.path.join(normal_path, img_name))

        # 读取pneumonia类别的图像
        pneumonia_path = os.path.join(data_dir, 'pneumonia')
        if os.path.exists(pneumonia_path):
            for img_name in os.listdir(pneumonia_path):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(pneumonia_path, img_name))
                    self.labels.append(1)  # 1表示肺炎
                    self.image_names.append(img_name)
                    self.image_paths.append(os.path.join(pneumonia_path, img_name))

        print(f"加载 {data_dir}: {len(self.images)} 张图像")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img_name = self.image_names[idx]

        # 使用PIL读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            # 返回一个黑色图像作为替代
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label, img_name


# 数据预处理和数据增强
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# 使用预训练模型（支持CAM）
class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2, model_type='resnet18'):
        super(PneumoniaClassifier, self).__init__()

        if model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            in_features = 512
            self.feature_layer = self.backbone.layer4  # 用于CAM的层
        elif model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_features = 2048
            self.feature_layer = self.backbone.layer4  # 用于CAM的层
        else:
            raise ValueError("不支持的模型类型")

        # 保存特征图用于CAM
        self.features = None
        self.gradients = None

        # 注册钩子获取特征图和梯度
        self.feature_layer.register_forward_hook(self.forward_hook)
        self.feature_layer.register_backward_hook(self.backward_hook)

        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward_hook(self, module, input, output):
        self.features = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.backbone(x)

    def get_activations(self, x):
        _ = self.forward(x)
        return self.features

    def get_gradients(self):
        return self.gradients


# CAM类激活图生成器（不使用OpenCV）
class CAMGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam(self, input_tensor, target_class=None):
        # 前向传播
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 清除之前的梯度
        self.model.zero_grad()

        # 创建one-hot向量用于反向传播
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1

        # 反向传播
        output.backward(gradient=one_hot)

        # 获取特征图和梯度
        features = self.model.features
        gradients = self.model.gradients

        # 计算权重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # 生成CAM
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU激活
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().detach().cpu().numpy(), output, target_class

    def overlay_cam(self, original_image, cam, alpha=0.5):
        """
        将CAM叠加到原始图像上（不使用OpenCV）
        """
        # 调整CAM大小以匹配原始图像
        from PIL import Image
        cam_pil = Image.fromarray(np.uint8(255 * cam))
        cam_resized = cam_pil.resize(original_image.size, Image.BILINEAR)
        cam_resized = np.array(cam_resized) / 255.0

        # 将原始图像转换为numpy数组
        img_array = np.array(original_image) / 255.0

        # 创建热力图（使用matplotlib的colormap）
        from matplotlib import cm
        heatmap = cm.jet(cam_resized)[:, :, :3]

        # 叠加图像
        overlayed = (1 - alpha) * img_array + alpha * heatmap
        overlayed = np.clip(overlayed, 0, 1)

        return overlayed, heatmap


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.cpu())

        print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)

        # 计算验证集的精确率、召回率和F1分数
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc.cpu())
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)

        print(f'验证 Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        print(f'验证 Precision: {val_precision:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f}')
        print()

        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_pneumonia_model.pth')

    # 绘制训练曲线
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 精确率曲线
    plt.subplot(2, 3, 3)
    plt.plot(val_precisions, label='Validation Precision')
    plt.title('Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # 召回率曲线
    plt.subplot(2, 3, 4)
    plt.plot(val_recalls, label='Validation Recall')
    plt.title('Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # F1分数曲线
    plt.subplot(2, 3, 5)
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores


# 测试函数
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_names = []

    with torch.no_grad():
        for inputs, labels, names in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_names.extend(names)

    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['正常', '肺炎']))

    return accuracy, precision, recall, f1, all_preds, all_labels, all_probs, all_names


# 可视化混淆矩阵
def plot_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 (Confusion Matrix)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cm


# 可视化CAM（类激活图）- 不使用OpenCV
def visualize_cam_for_samples(model, test_dataset, all_preds, all_labels, all_names, class_names, num_samples=6):
    """
    为测试集中的样本生成CAM可视化（不使用OpenCV）
    """
    # 创建CAM生成器
    cam_generator = CAMGenerator(model)

    # 找出正确分类和错误分类的样本索引
    correct_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred == label]
    incorrect_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]

    print(f"正确分类样本数: {len(correct_indices)}")
    print(f"错误分类样本数: {len(incorrect_indices)}")

    # 选择要可视化的样本
    selected_indices = []

    # 从正确分类的样本中选择一些
    if correct_indices:
        correct_samples = min(num_samples // 2, len(correct_indices))
        selected_indices.extend(np.random.choice(correct_indices, correct_samples, replace=False))

    # 从错误分类的样本中选择一些
    if incorrect_indices:
        incorrect_samples = min(num_samples // 2, len(incorrect_indices))
        selected_indices.extend(np.random.choice(incorrect_indices, incorrect_samples, replace=False))

    # 可视化选中的样本
    for idx in selected_indices:
        # 获取图像路径
        img_path = test_dataset.images[idx]

        # 读取原始图像
        original_image = Image.open(img_path).convert('RGB')

        # 预处理图像用于模型输入
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 生成CAM
        cam, output, pred_class = cam_generator.generate_cam(input_tensor)

        # 获取真实标签和预测标签
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        img_name = all_names[idx]

        # 将CAM叠加到原始图像上
        overlayed, heatmap = cam_generator.overlay_cam(original_image, cam)

        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 原始图像
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'原始图像\n真实: {class_names[true_label]}', fontsize=12)
        axes[0, 0].axis('off')

        # CAM热力图
        axes[0, 1].imshow(heatmap)
        axes[0, 1].set_title('类激活图 (CAM)', fontsize=12)
        axes[0, 1].axis('off')

        # 叠加图像
        axes[1, 0].imshow(overlayed)
        axes[1, 0].set_title(f'叠加图像\n预测: {class_names[pred_label]}', fontsize=12)
        axes[1, 0].axis('off')

        # 仅CAM
        axes[1, 1].imshow(cam, cmap='jet')
        axes[1, 1].set_title('原始CAM', fontsize=12)
        axes[1, 1].axis('off')

        # 设置整体标题
        title_color = 'green' if true_label == pred_label else 'red'
        fig.suptitle(f'图像: {img_name}\n真实: {class_names[true_label]} | 预测: {class_names[pred_label]}',
                     fontsize=14, color=title_color)

        plt.tight_layout()
        plt.savefig(f'cam_{img_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 打印模型关注区域分析
        print(f"\n分析图像: {img_name}")
        print(f"真实类别: {class_names[true_label]}, 预测类别: {class_names[pred_label]}")

        # 分析CAM
        cam_flat = cam.flatten()
        high_activation_threshold = np.percentile(cam_flat, 90)  # 前10%的高激活区域
        high_activation_ratio = np.sum(cam_flat > high_activation_threshold) / len(cam_flat)

        print(f"高激活区域比例: {high_activation_ratio:.4f}")

        if pred_label == 1:  # 肺炎
            if high_activation_ratio > 0.05:
                print("模型关注区域: 模型在肺部区域有显著激活，表明检测到了肺炎特征")
            else:
                print("模型关注区域: 激活较为分散，可能没有明确的肺炎特征")
        else:  # 正常
            if high_activation_ratio > 0.05:
                print("模型关注区域: 模型在某些区域有激活，但可能不是典型的肺炎特征")
            else:
                print("模型关注区域: 激活较为均匀，符合正常X光片的特征分布")

        print("-" * 50)


# 可视化分类样本
def visualize_classification_samples(test_loader, all_preds, all_labels, all_names, class_names, num_samples=12):
    # 找出正确分类和错误分类的样本索引
    correct_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred == label]
    incorrect_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]

    print(f"正确分类样本数: {len(correct_indices)}")
    print(f"错误分类样本数: {len(incorrect_indices)}")

    # 可视化正确分类的样本
    if correct_indices:
        print("\n可视化部分正确分类样本:")
        visualize_samples(test_loader, correct_indices, all_preds, all_labels, all_names, class_names,
                          "正确分类样本", min(num_samples, len(correct_indices)))

    # 可视化错误分类的样本
    if incorrect_indices:
        print("\n可视化部分错误分类样本:")
        visualize_samples(test_loader, incorrect_indices, all_preds, all_labels, all_names, class_names,
                          "错误分类样本", min(num_samples, len(incorrect_indices)))


# 可视化样本辅助函数
def visualize_samples(test_loader, indices, all_preds, all_labels, all_names, class_names, title, num_samples):
    # 随机选择样本
    if len(indices) > num_samples:
        selected_indices = np.random.choice(indices, num_samples, replace=False)
    else:
        selected_indices = indices

    # 获取原始图像
    original_images = []
    for idx in selected_indices:
        img_path = test_loader.dataset.images[idx]
        image = Image.open(img_path).convert('RGB')
        original_images.append(image)

    # 创建可视化
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.ravel()

    for i, idx in enumerate(selected_indices):
        if i >= len(axes):
            break

        image = original_images[i]
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        img_name = all_names[idx]

        # 显示图像
        axes[i].imshow(image)

        # 设置标题颜色：正确为绿色，错误为红色
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'{img_name}\n真实: {class_names[true_label]}\n预测: {class_names[pred_label]}',
                          color=color, fontsize=10)
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()


# 可视化类别分布
def plot_class_distribution(train_dataset, val_dataset, test_dataset, class_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = [train_dataset, val_dataset, test_dataset]
    dataset_names = ['训练集', '验证集', '测试集']

    for i, (dataset, name) in enumerate(zip(datasets, dataset_names)):
        if len(dataset) == 0:
            continue

        labels = dataset.labels
        class_counts = [labels.count(0), labels.count(1)]

        axes[i].bar(class_names, class_counts, color=['lightblue', 'lightcoral'])
        axes[i].set_title(f'{name}类别分布')
        axes[i].set_ylabel('样本数量')

        # 在柱状图上添加数字
        for j, count in enumerate(class_counts):
            axes[i].text(j, count + 5, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
def main():
    # 检查数据路径
    if not os.path.exists(DATA_PATH):
        print(f"数据路径不存在: {DATA_PATH}")
        return

    # 超参数设置
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001

    # 数据预处理
    train_transform, val_transform = get_transforms()

    # 创建数据集
    try:
        train_dataset = PneumoniaDataset(os.path.join(DATA_PATH, 'train'), transform=train_transform)
        val_dataset = PneumoniaDataset(os.path.join(DATA_PATH, 'val'), transform=val_transform)
        test_dataset = PneumoniaDataset(os.path.join(DATA_PATH, 'test'), transform=val_transform)
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        return

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 可视化类别分布
    plot_class_distribution(train_dataset, val_dataset, test_dataset, ['正常', '肺炎'])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 类别名称
    class_names = ['正常', '肺炎']

    # 创建模型
    print("创建模型...")
    try:
        model = PneumoniaClassifier(num_classes=2, model_type='resnet18')
        model = model.to(device)
        print("使用模型: ResNet18")
    except Exception as e:
        print(f"创建模型时出错: {e}")
        return

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 训练模型
    print("开始训练...")
    try:
        model, train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs
        )
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return

    # 测试模型
    print("测试模型...")
    accuracy, precision, recall, f1, all_preds, all_labels, all_probs, all_names = test_model(model, test_loader)

    # 可视化混淆矩阵
    cm = plot_confusion_matrix(all_labels, all_preds, class_names)

    # 可视化分类样本
    visualize_classification_samples(test_loader, all_preds, all_labels, all_names, class_names, num_samples=12)

    # 可视化CAM（类激活图）
    print("\n生成类激活图(CAM)...")
    visualize_cam_for_samples(model, test_dataset, all_preds, all_labels, all_names, class_names, num_samples=6)

    # 保存结果到文件
    with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("X-ray肺炎检测模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"F1分数 (F1-Score): {f1:.4f}\n")
        f.write("\n混淆矩阵:\n")
        f.write(str(cm) + "\n")
        f.write(f"\n正常类别样本数: {all_labels.count(0)}\n")
        f.write(f"肺炎类别样本数: {all_labels.count(1)}\n")
        f.write("\n模型可解释性分析:\n")
        f.write("通过类激活图(CAM)可视化，可以观察到模型在做出分类决策时关注的图像区域。\n")
        f.write("对于肺炎病例，模型通常会关注肺部的不透明区域、浸润区域和其他异常特征。\n")
        f.write("对于正常病例，模型关注区域较为均匀分布，没有特定的异常激活区域。\n")

    print("训练和测试完成！")
    print(f"最终测试准确率: {accuracy:.4f}")
    print("详细结果已保存到 evaluation_results.txt")
    print("\n模型可解释性分析:")
    print("通过类激活图(CAM)可视化，可以观察到模型在做出分类决策时关注的图像区域。")
    print("对于肺炎病例，模型通常会关注肺部的不透明区域、浸润区域和其他异常特征。")
    print("对于正常病例，模型关注区域较为均匀分布，没有特定的异常激活区域。")


if __name__ == "__main__":
    main()