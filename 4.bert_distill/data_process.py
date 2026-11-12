"""
该脚本的作用: 数据处理
1.读取原始数据
2.将原始数据封装成Dataset
3.实现批次数据的整理函数 collate_fn
4.定义Dataloader
"""
from torch.utils.data import Dataset, DataLoader

from config import Config
from tqdm import tqdm
import torch
config=Config()

#1.读取原始数据
def load_raw_data(data_path):
    """
    读取原始数据
    :param data_path: 原始数据路径 --> 词汇阅读是关键 08年考研暑期英语复习全指南	3
    :return: 列表，每个元素是一个字典，字典包含文本和标签 [(text,label),(text,label),...]
    """
    # 1.创建空列表用户保存数据
    result_list=[]
    with open(data_path,mode='r',encoding='utf-8') as f:
        #2.1读取所有行
        for line in f.readlines():
            # 2.2去除前后空格
            line=line.strip()
            #2.3如果数据为空则跳过
            if not line:
                continue
            #2.4分割数据
            label, text = line.split('\t', 1)
            # 当前数据格式为 label\ttext，封装为 (text, label)
            result_list.append((text, int(label)))
        return result_list
#2.将处理后的数据封装成Dataset
class TMFDataset(Dataset):
    def __init__(self,data_list):
        self.data_list=data_list

    def __len__(self):
        return len(self.data_list)

    # 3.根据数据索引获取指定数据
    def __getitem__(self, index):
        # 根据索引返回标题和标签数据
        text,label=self.data_list[index]
        return text,label

# 3.实现批次数据的整理函数 collate_fn
def collate_fn(batch):
    """
       批次数据整理: 词汇阅读是关键 08年考研暑期英语复习全指南
           ① 分词：将新闻标题按照字进行分词 --> 词 汇 阅 读 是 关 键 0 8 年 考 研 暑 期 英 语 复 习 全 指 南
           ② 添加特殊符号：[CLS] 词 汇 阅 读 是 关 键 0 8 年 考 研 暑 期 英 语 复 习 全 指 南 [SEP]
           ③ 编码：将分词后的词汇转换为对应的ID -->  101 23 45 567 890 46 89 35 102
           ④ 填充和截断：将所有数据填充为相同长度(input_ids) --> 101 23 45 567 890 46 89 35 102 0 0 0 0 0 0
           ⑤ 创建 attention_mask: 创建一个和input_ids形状相同的张量，用于标识输入序列中的有效令牌 --> 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
           ⑥ 将编码后的结果转换为张量
       :param batch: 一批数据 [(text,label),(text,label),...]
       :return: input_ids.shape = attention_mask = (batch_size, seq_len) labels.shape=(batch_size)
    """
    # 1.使用zip函数将一批数据的文本和标签分开 --> [(text,text,text),(label,label,label)]
    texts, labels = zip(*batch)
    encoded=config.tokenizer(
        texts,#要编码的文本
        add_special_tokens=True,#添加特殊符号
        max_length=config.pad_size,#最大长度
        padding='max_length',# 填充策略-->按照最大长度填充
        truncation=True,# 超过最大长度要进行截断
        return_attention_mask=True   # 返回 attention_mask
    )
    #3.获取变码后的结果
    input_ids=encoded['input_ids']
    attention_mask=encoded['attention_mask']
    # 4.将编码后的结果转换为张量
    input_ids = torch.tensor(input_ids)
    attention_mask=torch.tensor(attention_mask)
    labels=torch.tensor(labels)
    #返回结果
    return input_ids, attention_mask, labels
#4.定义Dataloader
def build_dataloader():
    #1.调用load_raw_data函数加载数据
    train_data_list=load_raw_data(config.train_data_path)
    test_data_list=load_raw_data(config.test_data_path)
    dev_data_list = load_raw_data(config.dev_data_path)
    # 2.将原始数据封装为Dataset
    train_dataset = TMFDataset(train_data_list)
    test_dataset = TMFDataset(test_data_list)
    dev_dataset = TMFDataset(dev_data_list)
    # 3.构建数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset,  # 指定数据集
        batch_size=config.batch_size,  # 批次大小
        shuffle=True,  # 是否打乱数据(训练集需要打乱 验证集和测试集不需要)
        collate_fn=collate_fn  # 数据整理函数
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,  # 指定数据集
        batch_size=config.batch_size,  # 批次大小
        shuffle=False,  # 是否打乱数据(训练集需要打乱 验证集和测试集不需要)
        collate_fn=collate_fn  # 数据整理函数
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,  # 指定数据集
        batch_size=config.batch_size,  # 批次大小
        shuffle=False,  # 是否打乱数据(训练集需要打乱 验证集和测试集不需要)
        collate_fn=collate_fn  # 数据整理函数
    )
    # 4.返回数据加载器
    return train_dataloader, test_dataloader, dev_dataloader
if __name__ == '__main__':
    # 测试加载并处理原始数据
    result_list = load_raw_data(config.train_data_path)
    print(f'加载并处理原始数据:{result_list[:2]}')
    print(f'数据集类型: {type(result_list)}')

    # 测试Dataset
    dataset = TMFDataset(result_list)
    print(f'Dataset数据集: {dataset[:2]}')
    print(f'数据集类型: {type(dataset)}')


    # 测试Dataloader
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        print(f'batch_idx: {batch_idx}')
        print(f'input_ids: {input_ids} , {input_ids.shape}')
        print(f'attention_mask: {attention_mask} , {attention_mask.shape}')
        print(f'labels: {labels} , {labels.shape}')
        break


