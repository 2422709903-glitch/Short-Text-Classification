
"""
该脚本的作用：
    1. 对模型进行评估
    2. 计算准确率、精确率、召回率和 F1 值
    3. 输出分类报告
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def eval_model(model,dev_data_loader,config):
    model.eval()
    predict_all=[]
    labels_all=[]
    #不计算梯度
    with torch.no_grad():
        #获取数据集开始评估
        for batch in dev_data_loader:
            # 获取input_ids attention_mask labels
            input_ids,attention_mask,labels=batch
            # 将数据放到device上
            input_ids=input_ids.to(config.device)
            attention_mask=attention_mask.to(config.device)
            labels=labels.to(config.device)
            #模型预测
            logits=model(input_ids,attention_mask)
            #获取预测类别
            predict=torch.argmax(logits,dim=1)
            # 获取真实类别
            predict_all.extend(predict.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    #计算评价指标
    accuracy = accuracy_score(labels_all, predict_all)

    precision = precision_score(
        labels_all, predict_all,
        average='macro',
        zero_division=0
    )

    recall = recall_score(
        labels_all, predict_all,
        average='macro',
        zero_division=0
    )

    f1 = f1_score(
        labels_all, predict_all,
        average='macro',
        zero_division=0
    )

    print('准确率：', accuracy)
    print('精确率：', precision)
    print('召回率：', recall)
    print('F1 Score：', f1)

    print(classification_report(
        labels_all,
        predict_all,
        labels=list(range(config.num_classes)),
        target_names=[config.class_dict[i] for i in range(config.num_classes)],
        zero_division=0
    ))
    return accuracy, precision, recall, f1

