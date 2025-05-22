import torch
from src.config import Config
from src.data_loader import IMDBDatasetLoader
from src.model_factory import ModelFactory
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer


def main():
    # 加载配置
    config = Config()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 数据加载
    print("加载数据...")
    data_loader = IMDBDatasetLoader(config)

    # 调用修正后的加载方法（无参数版本，默认使用config中的data_path和train_split）
    train_loader, test_loader = data_loader.load_and_preprocess_data() 

    # 模型创建
    print(f"创建 {config.model_type} 模型...")
    model = ModelFactory.create_model(config)
    if config.model_type == "bert":
        print(f"使用设备: {device}")
        model = model.to(device)  # 仅BERT模型需要迁移设备
    else:
        print(f"使用设备: CPU (sk-learn)")

    # 训练模型
    print("开始训练...")
    # 假设Trainer的初始化参数为(model, train_loader, test_loader, config, device)
    trainer = Trainer(config, model, train_loader, test_loader, device)
    trainer.train()

    # 评估模型（使用test_loader或val_loader，根据实际数据划分）
    print("评估模型...")
    evaluator = Evaluator(model, test_loader, device) 
    results = evaluator.evaluate()

    # 打印评估结果
    print("\n--------------------- 评估结果 ---------------------")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")

    # 可视化训练过程（假设trainer包含损失数据）
    visualizer = Visualizer(config)
    visualizer.plot_training_metrics(
        trainer.iterations,
        trainer.train_losses,
        trainer.test_losses 
    )
    visualizer.plot_model_result(results['accuracy'],results['precision'],results['recall'],results['f1'])
    print("训练和评估完成!")


if __name__ == '__main__':
    main()
