
"""
该脚本作用：实现预测函数，接受一条数据并返回预测结果
"""

import xpu_mask_patch
import torch
from config import Config
from model import TMFModel

config = Config()

# 1. 加载模型
model = TMFModel().to(config.device)

state_dict = torch.load(
    config.model_save_path,
    map_location=config.device
)

model.load_state_dict(state_dict)
model.eval()


# 2. 预测函数
def predict(input_data):
    """
    :param input_data: {'text': '体验2D巅峰 倚天屠龙记十大创新概览'}
    :return: {'text': '体验2D巅峰 倚天屠龙记十大创新概览', 'pred_class': '游戏'}
    """

    # 2.1 获取用户输入
    text = input_data['text']

    # 2.2 使用分词器处理用户数据
    encoded = config.tokenizer(
        [text],
        add_special_tokens=True,
        max_length=config.pad_size,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 2.3 获取 input_ids 和 attention_mask
    input_ids = encoded['input_ids'].to(config.device)
    attention_mask = encoded['attention_mask'].to(config.device)

    # 2.4 模型前向传播
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

        # 2.5 获取预测结果索引
        pred_index = torch.argmax(logits, dim=1)

        # 2.6 根据索引获取分类名称
        pred_class = config.class_dict[pred_index.item()]

        # 2.7 将预测结果添加到输入数据
        input_data['pred_class'] = pred_class

    # 2.8 返回结果
    return input_data


