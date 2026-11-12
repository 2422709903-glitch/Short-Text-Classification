
"""
该脚本作用:
    1. 项目的配置文件
    2. 配置项目中的数据路径
    3. 配置 BERT 模型
    4. 配置模型训练参数
"""

from transformers import BertTokenizer, BertConfig
import torch


class Config():
    def __init__(self):

        # 1. 原始数据文件路径
        self.train_data_path = '../1.data/data/train.txt'
        self.test_data_path = '../1.data/data/test.txt'
        self.dev_data_path = '../1.data/data/dev.txt'
        self.class_data_path = '../1.data/data/class.txt'

        # 2. 类别字典
        self.class_dict = {
            index: class_name.strip()
            for index, class_name in enumerate(
                open(self.class_data_path, 'r', encoding='utf-8')
            )
        }

        # 3. BERT 模型
        self.bert_model_path = './bert-base-chinese'

        self.tokenizer = BertTokenizer.from_pretrained(
            self.bert_model_path
        )

        self.bert_config = BertConfig.from_pretrained(
            self.bert_model_path
        )

        # 4. BERT 模型训练配置
        self.epochs = 30# 训练轮数
        self.batch_size = 64# 批处理大小
        self.learning_rate = 5e-5# 学习率

        self.num_classes = len(self.class_dict)
        self.pad_size = 32# 文本最大长度

        # 5. 设备配置
        self.device = torch.device(
            'xpu' if torch.xpu.is_available() else 'cpu'
        )

        # 6. 模型保存路径
        self.model_save_path = './save_model/bert_model.pt'

        # 7. 预测结果保存路径
        self.result_path = './result/bert_predict_result.csv'


if __name__ == '__main__':

    # 1. 创建配置对象
    config = Config()

    # 2. 获取原始数据路径
    print('训练数据路径:', config.train_data_path)
    print('测试数据路径:', config.test_data_path)
    print('验证数据路径:', config.dev_data_path)

    # 3. 获取 BERT 模型配置
    print('BERT 模型路径:', config.bert_model_path)
    print('隐藏层维度:', config.bert_config.hidden_size)

    # 4. 获取训练参数
    print('分类数量:', config.num_classes)
    print('Batch Size:', config.batch_size)
    print('最大文本长度:', config.pad_size)

    # 5. 获取训练设备
    print('当前训练设备:', config.device)

