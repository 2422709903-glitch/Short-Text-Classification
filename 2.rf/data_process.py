"""
数据预处理：
    对数据集进行再加工，将文本使用jieba进行分词，
    将分词结果添加到数据集中的words列
"""
import os
import jieba
import pandas as pd
from config import Config
config =Config()

def process_data(data_path):
    data=pd.read_csv(data_path,sep='\t',names=['label','text'],encoding='utf-8')
    data['words']=data['text'].apply(lambda x:' '.join(jieba.lcut(x)))
    #创建处理数据后的目录
    os.makedirs('processed_data',exist_ok=True)
    #保存数据
    if data_path==config.train_data_path:
        data.to_csv(config.processed_train_data_path,index=False)
        print("训练数据处理完成")
    if data_path==config.test_data_path:
        data.to_csv(config.processed_test_data_path,index=False)
        print("测试数据处理完成")
    if data_path==config.dev_data_path:
        data.to_csv(config.processed_dev_data_path,index=False)
        print("验证数据处理完成")
if __name__ == '__main__':
    process_data(config.train_data_path)
    process_data(config.test_data_path)
    process_data(config.dev_data_path)

