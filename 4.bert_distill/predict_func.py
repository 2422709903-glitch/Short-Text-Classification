"""
该脚本作用: 实现预测函数，接受一条数据并返回预测结果
"""
import xpu_mask_patch
import pickle
import pandas as pd
import torch
from config import Config
from student_model import BiLstmModel

config = Config()
model = BiLstmModel().to(config.device)

state_dict = torch.load(
    config.active_distilled_model_path,
    map_location=config.device
)

model.load_state_dict(state_dict)
model.eval()

# 4.预测函数
def predict(input_data):
    """
        :param  input_data: --> {'text':'体验2D巅峰 倚天屠龙记十大创新概览'}
        :return: --> {'text':'体验2D巅峰 倚天屠龙记十大创新概览', 'pred_class':'game'}
    """
    # 4.1获取用户输入
    text = input_data['text']
    # 4.2使用分词器处理用户数据
    encoded = config.tokenizer(
        [text],
        add_special_tokens=True,
        max_length=config.pad_size,

        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    # 4.3获取inputs_ids和attention_mask
    input_ids = encoded['input_ids'].to(config.device)
    attention_mask = encoded['attention_mask'].to(config.device)
    # 模型前向传播获取预测结果
    with torch.no_grad():
        # 4.4.1 获取概率化前的预测得分
        logits = model(input_ids, attention_mask)  # logits = (batch_size, num_classes)
        # 4.4.2 获取预测结果索引
        pred_index = torch.argmax(logits, dim=1)
        # 4.4.3 根据索引获取分类名称
        pred_class = config.class_dict[pred_index.item()]
        # 4.4.4 输入数据新增预测结果属性
        input_data['pred_class'] = pred_class
        # 4.5 返回结果
    return input_data


if __name__ == '__main__':
    input_data = {'text': '体验2D巅峰 倚天屠龙记十大创新概览'}
    print(predict(input_data))

