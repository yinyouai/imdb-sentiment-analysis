# src/data_loader.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import pickle


class IMDBDatasetLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path) if config.model_type == "bert" else None
        self.cache_dir = config.cache_dir if hasattr(config, 'cache_dir') else './.cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_raw_data(self, data_path):
        df = pd.read_csv(data_path)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0}).astype(int)
        return df

    def load_and_preprocess_data(self):
        df = self._load_raw_data(self.config.data_path)

        # 划分数据集
        train_size = int(self.config.train_split * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]

        if self.config.model_type == "bert":
            train_dataset = self._process_bert_data(train_df)
            test_dataset = self._process_bert_data(test_df)
        else:
            train_dataset = MLIMDBDataset(train_df)
            test_dataset = MLIMDBDataset(test_df)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn if self.config.model_type == "bert" else None
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn if self.config.model_type == "bert" else None
        )

        return train_loader, test_loader

    def _process_bert_data(self, df):

        processed_data = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            encoding = self.tokenizer(
                row['review'],
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            processed_data.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(row['sentiment'], dtype=torch.long)
            })
        return processed_data

    def _collate_fn(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }


# 机器学习数据集类（返回文本和标签）
class MLIMDBDataset(Dataset):
    def __init__(self, df):
        self.texts = df['review'].tolist()
        self.labels = df['sentiment'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]