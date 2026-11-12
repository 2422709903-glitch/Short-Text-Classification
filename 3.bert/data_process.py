"""
该脚本的作用: 数据处理
    1. 读取原始数据
    2. 将原始数据封装成 Dataset
    3. 实现批次数据的整理函数 collate_fn
    4. 定义 DataLoader
"""
from torch.utils.data import Dataset, DataLoader
from config import Config
from tqdm import tqdm
import torch

config=Config()

def load_raw_data(data_path):
    """
       读取原始数据

       原始数据格式:
           label \t text

       例如:
           2    夫妻俩边说情话边做那种事

       :param data_path: 原始数据路径
       :return: [(text, label), (text, label), ...]
       """
    # 1.1 创建空列表保存数据
    result_list=[]
    with open(data_path,mode='r',encoding='utf-8')as f:
        for line in tqdm(f,desc='读取数据'):
            line=line.rstrip('\r\n')# 去掉行末的换行符
            # 跳过空行
            if not line.strip():
                continue
            label,text=line.split('\t', 1)

            result_list.append((text,int(label)))
    return result_list
#将处理后的数据封装成Dataset
class TMFDataset(Dataset):
    def __init__(self,data_list):
        self.data_list=data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        text,label=self.data_list[index]
        return text,label

#实现批次数据的整理函数
def collate_fn(batch):
    """
        批次数据整理:

        ① 获取一批文本和标签
        ② 使用 BERT Tokenizer 编码
        ③ 添加特殊符号
        ④ 填充和截断
        ⑤ 创建 attention_mask
        ⑥ 转换为 PyTorch 张量

        :param batch: [(text, label), (text, label), ...]
        :return:
            input_ids.shape = (batch_size, seq_len)
            attention_mask.shape = (batch_size, seq_len)
            labels.shape = (batch_size,)
    """
    #使用zip将文本和标签分离
    texts,labels=zip(*batch)
    #使用BERT Tokenizer编码
    encoded=config.tokenizer(
        texts,
        add_special_tokens=True,
        max_length=config.pad_size,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
    )
    #获取编码结果
    input_ids=encoded['input_ids']
    attention_mask=encoded['attention_mask']

    #转换张量
    input_ids=torch.tensor(input_ids,dtype=torch.long)
    attention_mask=torch.tensor(attention_mask,dtype=torch.long)
    labels=torch.tensor(labels,dtype=torch.long)

    return input_ids, attention_mask, labels

#定义DataLoader
def build_dataloader():
    # 4.1 加载原始数据
    train_data_list = load_raw_data(config.train_data_path)
    test_data_list = load_raw_data(config.test_data_path)
    dev_data_list = load_raw_data(config.dev_data_path)

    # 4.2 将原始数据封装成 Dataset
    train_dataset = TMFDataset(train_data_list)
    test_dataset = TMFDataset(test_data_list)
    dev_dataset = TMFDataset(dev_data_list)

    # 4.3 构建训练集 DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 4.4 构建测试集 DataLoader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 4.5 构建验证集 DataLoader
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 4.6 返回 DataLoader
    return train_dataloader, test_dataloader, dev_dataloader
if __name__ == '__main__':

    # 1. 测试原始数据读取
    result_list = load_raw_data(config.train_data_path)

    print('加载的前两条数据:', result_list[:2])
    print('数据集类型:', type(result_list))
    print('训练集样本数量:', len(result_list))

    # 2. 测试 Dataset
    dataset = TMFDataset(result_list)

    print('Dataset 数据:', dataset[:2])
    print('Dataset 数据类型:', type(dataset))

    # 3. 测试 DataLoader
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    print('训练集批次数量:', len(train_dataloader))
    print('验证集批次数量:', len(dev_dataloader))
    print('测试集批次数量:', len(test_dataloader))

    # 4. 获取一个批次进行测试
    for batch_idx, batch in enumerate(train_dataloader):

        input_ids, attention_mask, labels = batch

        print('批次编号:', batch_idx)

        print('input_ids:', input_ids)
        print('input_ids.shape:', input_ids.shape)

        print('attention_mask:', attention_mask)
        print('attention_mask.shape:', attention_mask.shape)

        print('labels:', labels)
        print('labels.shape:', labels.shape)

        break

