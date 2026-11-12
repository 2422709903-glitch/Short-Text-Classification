"""
该脚本用于评估模型
"""
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

def eval_model(model, dataloader, device):
    # 1.开启模型的评估模式
    model.eval()
    # 2.初始化列表，存储预测结果和真实值
    all_preds, all_labels = [], []
    # 3.不计算梯度
    with torch.no_grad():
        # 4.获取数据开始评估
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='模型评估')):
            # 4.1 获取input_ids attention_mask labels
            input_ids, attention_mask, labels = batch
            # 4.2 将数据放到device上
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # 4.3 前向传播
            logits = model(input_ids, attention_mask)
            # 4.4 添加预测值和真实值到列表中
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    # 4.计算评估指标
    # 4.1 计算分类报告
    report = classification_report(all_labels, all_preds)
    # 4.2 计算F1
    f1 = f1_score(all_labels, all_preds, average='macro')
    # 4.3 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    # 4.4 计算精确率
    precision = precision_score(all_labels, all_preds, average='macro')
    # 4.5 计算召回率
    recall = recall_score(all_labels, all_preds,average='macro')

    # 5. 返回评估结果
    return report, f1, acc, precision, recall
