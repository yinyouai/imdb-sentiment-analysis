import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss
import joblib
class Trainer:
    def __init__(self, config, model, train_loader, test_loader, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = self.model.to(device) if config.model_type == "bert" else self.model  # 仅BERT需要移至设备

        # 初始化优化器/损失函数（机器学习模型无需优化器）
        self.loss_fn = torch.nn.CrossEntropyLoss() if config.model_type == "bert" else None
        self.optimizer = None
        if config.model_type == "bert":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            self.scaler = GradScaler(enabled=config.use_amp)

        # 训练指标
        self.train_losses = []
        self.test_losses = []
        self.iterations = []
        self.iteration_count = 0

    def train(self):
        for epoch in range(self.config.num_epochs):
            if self.config.model_type == "bert":
                self._train_bert(epoch)
            else:
                self._train_ml(epoch)
            self.save_model()
    def _train_bert(self, epoch):
        self.model.train()
        running_train_loss = 0.0
        with tqdm(self.train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{self.config.num_epochs}") as tepoch:
            for batch in tepoch:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                with autocast(enabled=self.config.use_amp, device_type="cuda"):
                    outputs = self.model(input_ids, attention_mask=attention_mask).logits
                    loss = self.loss_fn(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_train_loss += loss.item()
                current_loss = running_train_loss / (tepoch.n + 1)
                tepoch.set_postfix(loss=f"{current_loss:.4f}", refresh=False)
                self.train_losses.append(current_loss)
                self.iterations.append(self.iteration_count)
                self.iteration_count += 1

    def _train_ml(self, epoch):
        self.model.fit(
            [text for text, _ in self.train_loader.dataset],
            [label for _, label in self.train_loader.dataset]
        )
        # 计算训练损失（仅适用于支持概率输出的模型）
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba([text for text, _ in self.train_loader.dataset])
            loss = log_loss([label for _, label in self.train_loader.dataset], y_proba)
        else:
            loss = 0.0  # 无法计算时设为0
        self.train_losses.append(loss)
        self.iterations.append(self.iteration_count)
        self.iteration_count += 1
        print(f"Epoch {epoch + 1} 训练损失: {loss:.4f}")

    def evaluate(self):
        if self.config.model_type == "bert":
            test_loss = self._evaluate_bert()
            self.test_losses.append(test_loss)  # 记录BERT测试损失
            return {"loss": test_loss}  # 统一返回字典格式
        else:
            results = self._evaluate_ml()
            self.test_losses.append(results.get("loss", 0.0))  # 虚拟损失
            return results

    def _evaluate_bert(self):
        self.model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask).logits
                loss = self.loss_fn(outputs, labels)
                running_test_loss += loss.item()
        return running_test_loss / len(self.test_loader)

    def _evaluate_ml(self):
        # 机器学习模型评估逻辑
        self.model.eval()  # 兼容Scikit-learn接口
        y_true = [label for _, label in self.test_loader.dataset]
        y_pred = self.model.predict([text for text, _ in self.test_loader.dataset])

        y_proba = self.model.predict_proba([text for text, _ in self.test_loader.dataset])
        loss = log_loss([label for _, label in self.test_loader.dataset], y_proba)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "loss": loss  # 添加损失值
        }

    def save_model(self):
        """保存模型（支持BERT和机器学习模型）"""
        if self.config.model_type == "bert":
            # 保存PyTorch模型
            torch.save(self.model.state_dict(), self.config.model_save_path)
        else:
            # 保存Scikit-learn模型（使用joblib）
            joblib.dump(self.model, self.config.model_save_path)
        print(f"模型已保存至: {self.config.model_save_path}")