"""
    注意:
       1. 学生模型的选择 --> 因为我们要蒸馏的模型是Bert,学生模型首要要比Bert简答还能处理自然语言,所以选择了RNN --> BiLSTM
       2. 学生模型配置(BiLSTM)：    nn.Embedding --> BiLSTM -->nn.Dropout --> nn.Linear
            1. 双向LSTM
            2. 3 层LSTM
            3. 隐藏层大小为 256
            4. 输出层大小为 256 * 2
            5. BiLSTM不提供嵌入层  自己搭建嵌入层
               --嵌入层维度为 128
       3. 因为学生模型要向教师模型学习(蒸馏) 包含数据输入和损失计算，所以要求学生模型输入和输出的维度必须与教师模型一致
"""
from config import Config
import torch.nn as nn
import torch

config = Config()


class BiLstmModel(nn.Module):
    def __init__(self):
        super(BiLstmModel, self).__init__()
        # 嵌入层
        self.embeding = nn.Embedding(config.bert_config.vocab_size, config.embedding_dim)
        # 2.BiLstm(参数1: 输入维度  参数2: 隐藏层维度  参数3：层数  参数4:是否双向  参数5:Batch Size 维度放到前面 )
        self.bilstm = nn.LSTM(config.embedding_dim, config.hidden_dim, config.num_layers, bidirectional=True,
                              batch_first=True)
        # 3.随机失活
        self.dropout = nn.Dropout(config.dropout)
        # 4.全连接层
        self.linear = nn.Linear(config.hidden_dim * 2, config.num_classes)

    def forward(self, input_ids, attention_mask):
        # 0.准备工作 --> 有效token的掩码矩阵 (因为我们处理数据采用的还是Bert模型分分词器,会添加特殊符号和padding 这些对于BiLstm其实无效的Token,要过滤掉)
        cls_token_id, sep_token_id = 101, 102
        # 0.1 排除特殊符号(cls sep token位置变为False 其他事变为True)
        exclude_spicial_token_mask = (input_ids != cls_token_id) & (input_ids != sep_token_id)
        # 0.2 排除padding token位置变为False 其他位置变为True
        exclude_padding_token_mask = attention_mask & exclude_spicial_token_mask
        #0.3升维 为什么? 因为BiLSTM的输出维度是 hidden_dim * 2
        final_valid_token_mask=exclude_padding_token_mask.unsqueeze(-1)# final_valid_token_mask.shape == (batch_size, seq_len, 1)
        # 1.嵌入层
        embeded = self.embeding(input_ids)
        embeded=embeded*final_valid_token_mask
        # 2.数据进入到BiLstm --> lstm_out(每个时间步的输出)  hn(最后一个时间步的输出) cell(最后一个时间步记忆状态)
        lstm_out,(hn,c)=self.bilstm(embeded)# lstm_out.shape = (batch_size, seq_len, hidden_dim * 2)
        lstm_out=lstm_out*final_valid_token_mask
        # 3.将BiLstm的输出进行平均池化--模拟Bert模型 pooler_output (因为使用Bert模型使用句向量cls作为句子的语义表示, 所以我们将BiLstm每一个时间步输出累加求平均进行模拟)
        sum_hidden = torch.sum(lstm_out, dim=1)  # sum_hidden.shape = (batch_size, hidden_dim * 2) = (2, 512)
        valid_token_count = final_valid_token_mask.sum(dim=1) + 1e-8  # valid_token_count.shape = (batch_size,token_count) = [[4], [6]]
        pooler_output = sum_hidden / valid_token_count
        # 4.数据进入随机失活层
        dropout = self.dropout(pooler_output)
        # 5.数据进入全连接
        logits = self.linear(dropout)
        # 6.返回结果
        return logits
