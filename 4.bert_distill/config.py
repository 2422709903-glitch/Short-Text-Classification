"""
  该脚本作用:
     --项目的配置文件(防止硬编码)
     --配置项目中的数据路径
"""
from transformers import BertTokenizer, BertConfig
import torch
from pathlib import Path

class Config():
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent
        self.data_dir = self.project_dir.parent / '1.data' / 'data'
        original_train_path = self.data_dir / 'train.txt'
        balanced_train_path = self.data_dir / 'train_balanced.txt'
        # 1.原始数据文件路径
        self.train_data_path = str(balanced_train_path if balanced_train_path.is_file() else original_train_path)
        self.test_data_path = str(self.data_dir / 'test.txt')
        self.dev_data_path = str(self.data_dir / 'dev.txt')
        self.class_data_path = str(self.data_dir / 'class.txt')
        self.stopwords_data_path = str(self.data_dir / 'stopwords.txt')

        # 2.类别字典 --> {0:game, 1:'education',....}
        self.class_dict = {index: class_name.strip() for index, class_name in enumerate(open(self.class_data_path, 'r', encoding='utf-8'))}

        # 3.bert模型
        self.bert_model_path = str(self.project_dir / 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)  # Tokenizer
        self.bert_config = BertConfig.from_pretrained(self.bert_model_path)  # Bert配置

        # 4.学生模型(bilstm)模型定义配置
        self.embedding_dim = 128  # 嵌入层维度
        self.hidden_dim = 256  # 隐藏层维度
        self.dropout = 0.3  # 随机失活比例
        self.num_layers = 3  # 层数

        # 4.学生模型(bilstm)模型训练配置
        self.epochs = 3  # 训练轮数
        self.batch_size = 64  # 批次
        self.learning_rate = 5e-4  # 学习率
        self.num_classes = len(self.class_dict)  # 类别数量
        self.pad_size = 32  # 填充到最大长度
        self.device = torch.device('xpu' if torch.xpu.is_available() else 'cpu') # 设备

        # 5.模型保存路径
        balanced_teacher_path = self.project_dir.parent / '3.bert' / 'save_model' / 'bert_model_balanced.pt'
        old_teacher_path = self.project_dir.parent / '3.bert' / 'save_model' / 'bert_model.pt'
        self.teacher_model_save_path = str(balanced_teacher_path if balanced_teacher_path.is_file() else old_teacher_path)

        #学生模型保存路径
        self.distilled_model_save_path = str(self.project_dir / 'save_model' / 'distilled_tmf_model_balanced.pt')
        self.active_distilled_model_path = str(
            self.project_dir / 'save_model' / 'distilled_tmf_model_balanced.pt'
            if (self.project_dir / 'save_model' / 'distilled_tmf_model_balanced.pt').is_file()
            else self.project_dir / 'save_model' / 'distilled_tmf_model.pt'
        )

if __name__ == '__main__':
    # 1. 创建配置对象
    config = Config()

    # 2.获取原始数据路劲
    print(f'训练数据路径 --> {config.train_data_path}')
    print(f'测试数据路径 --> {config.test_data_path}')
    print(f'验证数据路径 --> {config.dev_data_path}')

    # 3.获取模型
    print(f'Bert模型路径-->{config.bert_model_path}')

    # 4.训练参数
    print(f"设备-->{config.device}")


