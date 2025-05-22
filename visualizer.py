import matplotlib.pyplot as plt
import numpy as np
import os


class Visualizer:
    def __init__(self, config):
        self.config = config
        plt.rcParams['font.family'] = 'SimSun'

    def plot_training_metrics(self, iterations, train_losses, test_losses):
        plt.figure(figsize=(10, 5))

        # 训练损失曲线
        plt.plot(iterations, train_losses, label='训练损失', marker='o', color='blue', alpha=0.7)

        # 测试损失曲线
        plt.plot(iterations[:len(test_losses)], test_losses, label='测试损失', marker='s', color='red', linestyle='--',
                 alpha=0.7)

        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('训练与测试损失趋势')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.config.plot_save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线图已保存至: {self.config.plot_save_path}")
        plt.close()

    def plot_model_result(self, accuracy, precision, recall, f1):
        plt.figure(figsize=(10, 5))

        # 指标名称和对应数值
        metrics = {
            '准确率': accuracy,
            '精确率': precision,
            '召回率': recall,
            'F1分数': f1
        }

        # 提取指标名称和数值
        names = list(metrics.keys())
        values = list(metrics.values())

        # 创建柱状图
        bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=0.6)

        # 添加数值标签（居中显示在柱状图上方）
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')

        # 设置图表标题和坐标轴标签
        plt.title('模型评估指标对比', fontsize=14)
        plt.xlabel('评估指标', fontsize=12)
        plt.ylabel('数值', fontsize=12)

        # 设置纵轴刻度范围（根据数值自动调整，此处示例为0-1）
        plt.ylim(0, max(values) + 0.1 if max(values) < 1 else 1.1)

        # 美化
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()  # 自动调整布局

        # 保存图像（
        result_save_path = os.path.join(self.config.plot_save_path, 'model_metrics.png')
        plt.savefig(self.config.plot_save2_path, dpi=300, bbox_inches='tight')
        print(f"评估指标图已保存至: {result_save_path}")

        plt.close()