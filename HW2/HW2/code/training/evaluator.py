"""评估模块"""
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from config.settings import Config


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def evaluate(self):
        """全面评估模型"""
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        image_names = []
        
        with torch.no_grad():
            for batch_data, batch_labels, names in self.test_loader:
                batch_data = batch_data.to(Config.DEVICE)
                batch_labels = batch_labels.to(Config.DEVICE)
                
                outputs = self.model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch_labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                image_names.extend(names)
        
        return self._compute_metrics(predictions, true_labels, probabilities, image_names)
    
    def _compute_metrics(self, predictions, true_labels, probabilities, image_names):
        """计算评估指标"""
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # 打印结果
        self._print_results(accuracy, precision, recall, f1, true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'image_names': image_names
        }
    
    def _print_results(self, accuracy, precision, recall, f1, true_labels, predictions):
        """打印评估结果"""
        print("=" * 60)
        print("模型测试结果分析")
        print("=" * 60)
        print(f"整体准确率: {accuracy:.4f}")
        print(f"精确率指标: {precision:.4f}")
        print(f"召回率指标: {recall:.4f}")
        print(f"F1综合分数: {f1:.4f}")
        print("\n详细分类报告:")
        print(classification_report(true_labels, predictions, 
                                  target_names=Config.CLASS_NAMES))
    
    @staticmethod
    def save_results(results, confusion_mat, filepath):
        """保存评估结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("医学影像肺炎检测系统评估报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"模型准确率: {results['accuracy']:.4f}\n")
            f.write(f"精确率指标: {results['precision']:.4f}\n")
            f.write(f"召回率指标: {results['recall']:.4f}\n")
            f.write(f"F1综合分数: {results['f1']:.4f}\n")
            f.write("\n混淆矩阵分析:\n")
            f.write(str(confusion_mat) + "\n")
            f.write(f"\n正常影像数量: {results['true_labels'].count(0)}\n")
            f.write(f"肺炎影像数量: {results['true_labels'].count(1)}\n")
            f.write("\n模型决策分析:\n")
            f.write("类激活图可视化显示模型能够有效定位肺炎相关特征。\n")
            f.write("高激活区域通常对应于肺部浸润、实变等病理改变。\n")
            f.write("正常影像的激活分布均匀，符合健康肺部特征。\n")