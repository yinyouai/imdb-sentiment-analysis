# IMDB影评情感分析项目

## 项目概述
本项目实现了基于深度学习和传统机器学习的IMDB影评情感分析系统，支持使用BERT模型或TF-IDF+分类器的方式进行影评情感（积极/消极）分类。项目采用模块化设计，包含数据加载、模型训练、评估和可视化等完整流程。

项目已融合机器学习方法四种，朴素贝叶斯、支持向量机、随机森林以及逻辑回归，深度学习模型一种Bert，由HuaggingFace-Goog版，已收入项目src中，参数1.1亿，预训练建议采用cuda上传至GPU

模型可在config内选择

数据集可以更换为其他文本数据

编者注：
一个课设，仅供纪念娱乐。

## 项目结构
```
imdb-sentiment-analysis/
├── src/                     # 源代码
│   ├── config.py            # 配置管理
│   ├── data_loader.py       # 数据加载与预处理
│   ├── model_factory.py     # 模型构建
│   ├── trainer.py           # 训练逻辑
│   ├── evaluator.py         # 评估逻辑
│   ├── visualizer.py        # 可视化
│   └── utils.py             # 工具函数
├── models/                  # 训练好的模型
├── data/                    # 数据集
├── plots/                   # 可视化结果
├── main.py                  # 主入口
├── requirements.txt         # 依赖文件
└── README.md                # 项目说明
```

## 环境准备
1. 克隆项目：
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载数据集：
- 从Kaggle下载IMDB影评数据集：https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- 将`IMDB Dataset.csv`放入`data/`目录

4. 下载BERT预训练模型：
- 从Hugging Face下载`bert-base-uncased`模型：https://huggingface.co/bert-base-uncased
- 将模型文件放入`src/bert-base-uncased/`目录

## 使用方法src

### 配置参数
在`src/config.py`中可以修改以下参数：
- `data_path`: 数据集路径
- `bert_path`: BERT模型路径
- `model_type`: 模型类型（可选：bert, tfidf_nb, tfidf_svm, tfidf_rf, tfidf_lr）
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `use_amp`: 是否使用混合精度训练

### 运行训练
```bash
python main.py
```

### 主要功能

1. **数据处理**：
   - 加载IMDB影评数据集
   - 文本分词和编码
   - 数据集划分

2. **模型支持**：
   - BERT微调模型
   - TF-IDF+朴素贝叶斯
   - TF-IDF+支持向量机
   - TF-IDF+随机森林
   - TF-IDF+逻辑回归

3. **训练特性**：
   - 混合精度训练加速
   - 训练过程可视化
   - 模型自动保存

4. **评估指标**：
   - 准确率(Accuracy)
   - 精确率(Precision)
   - 召回率(Recall)
   - F1分数

## 实验结果
训练完成后，模型评估结果将显示在控制台，同时生成训练过程曲线图保存到`plots/`目录。

## 模型部署
训练好的模型保存在`models/`目录，可以通过以下方式加载使用：
```python
from src.config import Config
from src.model_factory import ModelFactory
import torch

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
model = ModelFactory.create_model(config)

# 加载BERT模型权重
if config.model_type == "bert":
    model.load_state_dict(torch.load(config.model_save_path))

model.to(device)
model.eval()

# 进行预测
def predict_sentiment(text):
    # 文本预处理和预测逻辑
    pass
```

## 贡献
欢迎提交问题和PR来改进这个项目。

## 许可证
本项目采用MIT许可证。

## 依赖版本说明
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- scikit-learn 1.2+
- pandas 1.5+
- numpy 1.23+
- matplotlib 3.6+
- tqdm 4.64+
