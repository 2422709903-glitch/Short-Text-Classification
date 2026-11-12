"""
该脚本的作用：
    ①定义新闻文本分类模型
        1.bert模型
        2.liner线性层
    ②测试模型
"""
import torch
import torch.nn as nn
from transformers import BertModel

from config import Config

config = Config()


class TMFModel(nn.Module):

    def __init__(self):
        super(TMFModel, self).__init__()

        # 加载 BERT，并明确指定 eager 注意力实现
        self.bert = BertModel.from_pretrained(
            config.bert_model_path,
            attn_implementation="eager"
        )

        # BERT 输出维度是 768
        # 分类层将 768 维特征映射到新闻类别数
        self.linear = nn.Linear(
            config.bert_config.hidden_size,
            config.num_classes
        )

    def forward(self, input_ids, attention_mask):

        # BERT 提取文本特征
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取 [CLS] 对应的池化向量
        pooler_output = bert_output.pooler_output

        # 分类
        logits = self.linear(pooler_output)

        return logits
