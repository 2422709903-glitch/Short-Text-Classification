"""BERT 实验配置。"""

from pathlib import Path

import torch
from transformers import BertConfig, BertTokenizer


class Config:
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent
        self.data_dir = self.project_dir.parent / '1.data' / 'data'
        original_train = self.data_dir / 'train.txt'
        balanced_train = self.data_dir / 'train_balanced.txt'
        self.train_data_path = str(balanced_train if balanced_train.is_file() else original_train)
        self.test_data_path = str(self.data_dir / 'test.txt')
        self.dev_data_path = str(self.data_dir / 'dev.txt')
        self.class_data_path = str(self.data_dir / 'class.txt')

        with open(self.class_data_path, encoding='utf-8') as f:
            self.class_dict = {index: line.strip() for index, line in enumerate(f) if line.strip()}

        self.bert_model_path = str(self.project_dir / 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_model_path)

        # 保留已验证的 BERT 训练参数。
        self.epochs = 3
        self.batch_size = 64
        self.learning_rate = 5e-5
        self.num_classes = len(self.class_dict)
        self.pad_size = 32
        self.device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

        self.model_save_path = str(self.project_dir / 'save_model' / 'bert_model.pt')
        self.balanced_model_save_path = str(self.project_dir / 'save_model' / 'bert_model_balanced.pt')
        self.active_model_path = self.balanced_model_save_path if Path(self.balanced_model_save_path).is_file() else self.model_save_path
        self.result_path = str(self.project_dir / 'result' / 'bert_predict_result.csv')


if __name__ == '__main__':
    config = Config()
    print('训练数据路径:', config.train_data_path)
    print('验证数据路径:', config.dev_data_path)
    print('测试数据路径:', config.test_data_path)
    print('分类数量:', config.num_classes)
    print('训练设备:', config.device)

