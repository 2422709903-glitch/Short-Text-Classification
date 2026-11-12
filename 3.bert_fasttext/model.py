"""
该脚本的作用：
    1. 定义 BERT 文本分类模型
    2. 使用 Linear 层将 BERT 输出映射到 26 个类别
"""
import torch
import torch.nn as nn
from transformers import BertModel
from config import Config

config=Config()
class TMFModel(nn.Module):
    def __init__(self):
        super(TMFModel, self).__init__()
        #加载预训练模型BERT
        self.bert=BertModel.from_pretrained(config.bert_model_path,
                                            attn_implementation='eager')
        # 分类层：768 维映射到 26 个类别
        self.linear=nn.Linear(
            config.bert_config.hidden_size,
            config.num_classes
        )

    def forward(self,input_ids,attention_mask):
        #BERT文本特征提取
        bert_output=self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #获取[CLS]对应的池化结果
        pooler_out=bert_output.pooler_output

        #分类预测
        logits=self.linear(pooler_out)

        return logits

