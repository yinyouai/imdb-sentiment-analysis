import os
from dataclasses import dataclass


@dataclass
class Config:
    # 数据配置
    data_path: str = "data/IMDB Dataset.csv"
    bert_path: str = "src/bert-base-uncased"
    max_length: int = 256
    train_split: float = 0.8
    batch_size: int = 64

    # 模型配置
    model_type: str = "bert"  # 可选: bert, tfidf_nb, tfidf_svm, tfidf_rf, tfidf_lr
    num_labels: int = 2

    # 训练配置
    learning_rate: float = 2e-5
    num_epochs: int = 3
    use_amp: bool = True  # 混合精度训练

    # 输出配置
    model_save_path: str = f"models/best_{model_type}.pth"
    plot_save_path: str = f"plots/{model_type}_training_metrics.png"
    plot_save2_path: str = f"plots/{model_type}_result.png"
    def __post_init__(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.plot_save_path), exist_ok=True)