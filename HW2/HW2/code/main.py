"""主程序"""
import os
from config.settings import Config
from data.dataset import DataLoaderManager
from models.classifier import ModelFactory
from training import ModelTrainer, ModelEvaluator
from visualization.plots import PlotManager
from visualization.cam import CAMVisualizer

from utils import Timer 

Config.OUTPUT_DIR = "./output/Res18"

def main():
    """主函数"""
    # 初始化配置
    Config.setup_environment()
    timer = Timer()
    timer.__enter__()

    # 检查数据路径
    if not os.path.exists(Config.DATA_ROOT):
        print(f"数据路径不存在: {Config.DATA_ROOT}")
        return
    
    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            DataLoaderManager.create_data_loaders(Config.DATA_ROOT)
        
        print(f"训练集规模: {len(train_dataset)}")
        print(f"验证集规模: {len(val_dataset)}")
        print(f"测试集规模: {len(test_dataset)}")
        
        # 可视化数据分布
        PlotManager.plot_class_distribution(train_dataset, val_dataset, test_dataset)
        
        # 创建模型
        model = ModelFactory.create_model()
        
        # 训练模型
        trainer = ModelTrainer(model, train_loader, val_loader)
        trained_model, training_metrics = trainer.train()
        
        # 评估模型
        evaluator = ModelEvaluator(trained_model, test_loader)
        results = evaluator.evaluate()
        
        # 可视化结果
        confusion_mat = PlotManager.plot_confusion_matrix(
            results['true_labels'], 
            results['predictions']
        )
        
        # 可视化样本预测
        PlotManager.plot_sample_predictions(
            test_loader,
            results['predictions'],
            results['true_labels'], 
            results['image_names']
        )
        
        # 可视化CAM
        print("\n生成类激活图...")
        cam_visualizer = CAMVisualizer(trained_model)
        cam_visualizer.visualize_samples(
            test_dataset,
            results['predictions'],
            results['true_labels'],
            results['image_names']
        )
        
        # 保存结果
        evaluator.save_results(results, confusion_mat, f'{Config.OUTPUT_DIR}/evaluation_report.txt')
        
        print("\n医学影像分析流程完成！")
        print(f"最终测试准确率: {results['accuracy']:.4f}")
        print(f"详细结果保存至: {Config.OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"流程执行失败: {e}")
        raise

    timer.__exit__()


if __name__ == "__main__":
    main()