import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluator:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def evaluate(self):
        if isinstance(self.model, torch.nn.Module):
            return self._evaluate_bert()
        else:
            return self._evaluate_ml()

    def _evaluate_bert(self):
        self.model.eval()
        running_loss = 0.0
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch in self.data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask).logits
                loss = self._calculate_loss(outputs, labels)
                running_loss += loss.item()

                # 提取真实标签和预测标签（转换为 CPU  numpy 数组）
                _, preds = torch.max(outputs, dim=1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

        # 计算完整评估指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average="binary")
        recall = recall_score(true_labels, pred_labels, average="binary")
        f1 = f1_score(true_labels, pred_labels, average="binary")

        return {
            "loss": running_loss / len(self.data_loader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def _evaluate_ml(self):
        y_true = [label for _, label in self.data_loader.dataset]
        y_pred = self.model.predict([text for text, _ in self.data_loader.dataset])
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary"),
            "recall": recall_score(y_true, y_pred, average="binary"),
            "f1": f1_score(y_true, y_pred, average="binary")
        }

    def _calculate_loss(self, outputs, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(outputs, labels)