from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from transformers import BertForSequenceClassification


class ModelFactory:
    @staticmethod
    def create_model(config):
        if config.model_type == "bert":
            return BertForSequenceClassification.from_pretrained(
                config.bert_path,
                num_labels=config.num_labels
            )
        elif config.model_type.startswith("tfidf_"):

            model_name = config.model_type.split("_")[1]
            tfidf_pipeline = make_pipeline(
                TfidfVectorizer(max_features=5000, stop_words='english'),
                ModelFactory._get_ml_model(model_name)
            )
            return tfidf_pipeline
        else:
            raise ValueError(f"不支持的模型类型: {config.model_type}")

    @staticmethod
    def _get_ml_model(model_name):
        models = {
            "nb": MultinomialNB(),
            "svm": SVC(probability=True),
            "rf": RandomForestClassifier(n_estimators=100),
            "lr": LogisticRegression(max_iter=1000)
        }
        return models.get(model_name, MultinomialNB())  # 默认朴素贝叶斯