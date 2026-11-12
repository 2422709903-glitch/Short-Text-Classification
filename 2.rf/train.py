"""在固定训练集上训练 Random Forest，并仅使用 dev 选择与评估。"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from config import Config


config = Config()


def train():
    train_data = pd.read_csv(config.processed_train_data_path)
    dev_data = pd.read_csv(config.processed_dev_data_path)
    train_texts, train_labels = train_data['words'], train_data['label']
    dev_texts, dev_labels = dev_data['words'], dev_data['label']

    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    train_features = vectorizer.fit_transform(train_texts)
    dev_features = vectorizer.transform(dev_texts)

    # 训练使用整个（可选过采样的）train；固定 dev/test 不参与重新切分。
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
    )
    model.fit(train_features, train_labels)

    predictions = model.predict(dev_features)
    print(f'验证集 Accuracy: {accuracy_score(dev_labels, predictions):.4f}')
    print(f'验证集 Macro Precision: {precision_score(dev_labels, predictions, average="macro", zero_division=0):.4f}')
    print(f'验证集 Macro Recall: {recall_score(dev_labels, predictions, average="macro", zero_division=0):.4f}')
    print(f'验证集 Macro-F1: {f1_score(dev_labels, predictions, average="macro", zero_division=0):.4f}')

    with open(config.class_data_path, encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(classification_report(
        dev_labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))

    os.makedirs(os.path.dirname(config.balanced_rf_model_save_path), exist_ok=True)
    with open(config.balanced_tfidf_model_save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(config.balanced_rf_model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print('新模型已保存；原有 Random Forest 模型保持不变。')


if __name__ == '__main__':
    train()

