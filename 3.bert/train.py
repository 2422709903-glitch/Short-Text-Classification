"""
该脚本的作用：
    1. 加载训练集和验证集
    2. 定义模型、损失函数和优化器
    3. 训练 BERT 模型
    4. 使用验证集评估模型
    5. 保存表现最好的模型
"""
import xpu_mask_patch

import os
import torch
import torch.nn as nn
from config import Config
from data_process import build_dataloader
from model import TMFModel
from eval import eval_model

"""
该脚本的作用：
    1. 加载训练集和验证集
    2. 定义模型、损失函数和优化器
    3. 训练 BERT 模型
    4. 使用验证集评估模型
    5. 保存表现最好的模型
"""

# 必须放在最前面，启用 XPU Mask 修复
import xpu_mask_patch

import os
import torch
import torch.nn as nn
from config import Config
from data_process import build_dataloader
from model import TMFModel
from eval import eval_model
from tqdm import tqdm

config = Config()


if __name__ == '__main__':

    # 创建保存模型的目录
    os.makedirs('./save_model', exist_ok=True)

    # 加载数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 初始化模型
    model = TMFModel().to(config.device)

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        foreach=False
    )

    # 记录最佳 F1 值
    best_f1 = 0

    print('当前训练设备：', config.device)

    # 开始训练
    for epoch in range(config.epochs):

        model.train()

        total_loss = 0

        for batch_idx, batch in enumerate(
                tqdm(train_dataloader, desc=f'第 {epoch + 1} 轮模型训练')
        ):

            input_ids, attention_mask, labels = batch

            # 将数据移动到训练设备
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # 前向传播
            logits = model(input_ids, attention_mask)

            # 计算损失
            loss = loss_func(logits, labels)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            total_loss += loss.item()

            # 每 100 个批次打印一次训练情况
            if (batch_idx + 1) % 100 == 0:

                avg_loss = total_loss / 100

                print(
                    f'轮数: {epoch + 1}, '
                    f'批次数: {batch_idx + 1}, '
                    f'平均损失: {avg_loss:.4f}'
                )

                total_loss = 0

        # 每轮训练结束后，使用验证集评估
        print(f'\n第 {epoch + 1} 轮训练结束，开始验证模型...')

        accuracy, precision, recall, f1 = eval_model(
            model,
            dev_dataloader,
            config
        )

        # 保存表现最好的模型
        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                config.balanced_model_save_path
            )

            print('当前模型已保存！')

        print(
            f'第 {epoch + 1} 轮结果：'
            f'Accuracy={accuracy:.4f}, '
            f'Macro-F1={f1:.4f}'
        )

    print('\n训练完成！')
    print('最佳 Macro-F1：', best_f1)

