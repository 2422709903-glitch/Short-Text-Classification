"""
  该脚本作用:
     --项目的配置文件(防止硬编码)
     --配置项目中的数据路径
"""
from transformers import BertModel, BertTokenizer, BertConfig
import torch

class Config():
    def __init__(self):
        # 1.原始数据文件路径
        self.train_data_path = '../1.data/data/train.txt'         # 训练数据集文件位置
        self.test_data_path = '../1.data/data/test.txt'           # 测试数据集文件位置
        self.dev_data_path = '../1.data/data/dev.txt'             # 验证数据集文件位置
        self.class_data_path = '../1.data/data/class.txt'         # 类别数据文件位置
        self.stopwords_data_path = '../1.data/data/stopwords.txt' # 停用词数据文件位置

        # 2.类别字典 --> {0:game, 1:'education',....}
        self.class_dict = {index: class_name.strip() for index, class_name in enumerate(open(self.class_data_path, 'r', encoding='utf-8'))}

        # 3.bert模型
        self.bert_model_path = '../3.bert_fasttext/bert-base-chinese'  # 复用现有本地 BERT
        self.bert_model = BertModel.from_pretrained(self.bert_model_path)  # Bert模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)  # Tokenizer
        self.bert_config = BertConfig.from_pretrained(self.bert_model_path)  # Bert配置

        # 4.学生模型(bilstm)模型定义配置
        self.embedding_dim = 128  # 嵌入层维度
        self.hidden_dim = 256  # 隐藏层维度
        self.dropout = 0.3  # 随机失活比例
        self.num_layers = 3  # 层数

        # 4.学生模型(bilstm)模型训练配置
        self.epochs = 2  # 训练轮数
        self.batch_size = 64  # 批次
        self.learning_rate = 5e-4  # 学习率
        self.num_classes = len(self.class_dict)  # 类别数量
        self.pad_size = 32  # 填充到最大长度
        self.device = torch.device('xpu' if torch.xpu.is_available() else 'cpu') # 设备

        # 5.模型保存路径
        self.teacher_model_save_path = '../3.bert_fasttext/save_model/bert_model.pt'

        #学生模型保存路径
        self.distilled_model_save_path='./save_model/distilled_tmf_model.pt'

if __name__ == '__main__':
    # 1. 创建配置对象
    config = Config()

    # 2.获取原始数据路劲
    print(f'训练数据路径 --> {config.train_data_path}')
    print(f'测试数据路径 --> {config.test_data_path}')
    print(f'验证数据路径 --> {config.dev_data_path}')

    # 3.获取模型
    print(f'Bert模型-->{config.bert_model}')

    # 4.训练参数
    print(f"设备-->{config.device}")


