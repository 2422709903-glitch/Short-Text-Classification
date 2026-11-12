"""
该脚本的作用：
    1.使用新闻文本分类数据实现TMFModel的蒸馏
    2.一边蒸馏一边评估
    3.保存蒸馏模型
"""
#import xpu_mask_patch
from sklearn.metrics import accuracy_score, f1_score
from config import Config
from data_process import build_dataloader
from teacher_model import TMFModel    # 教师模型
from student_model import BiLstmModel # 学生模型
import torch
import torch.nn as nn
from tqdm import tqdm
from eval import eval_model
import torch.nn.functional as F

# 1.创建配置对象
config = Config()

# 2.定义模型训练方式
def distill_model():
    # 1 创建数据加载器
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 2 创建模型并放到指定设备上
    teacher_model = TMFModel().to(config.device)
    teacher_model.load_state_dict(torch.load(config.teacher_model_save_path,map_location='cpu'))
    student_model = BiLstmModel().to(config.device)
    # 3 创建交叉熵损失函数
    cross_loss_fn = nn.CrossEntropyLoss()
    # 4 创建优化器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate,foreach=False)
    # 5 外层循环控制轮次
    best_f1, T, alpha = 0.0, 2, 0.7  # best_f1 --> 最佳的f1分数  T--> 蒸馏温度(平滑系数) alpha --> 权重比例
    for epoch in range(config.epochs):
        # 5.1 模型开启模型
        teacher_model.eval()
        student_model.train()
        # 5.2 记录总损失、预测值、真实值
        total_loss, pred_labels, true_labels = 0.0, [], []
        # 5.3 开启批次训练
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc='模型蒸馏')):
            # 5.3.1 获取数据并将数据放到和模型相同的设备上
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # 5.3.2 前向传播
            # ① 学生模型前向传播
            student_logits = student_model(input_ids, attention_mask)
            student_labels = torch.argmax(student_logits, dim=-1)
            # ② 教师模型的前向传播
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)
                teacher_labels = torch.argmax(teacher_logits, dim=-1)   # 硬标签
            # 5.3.3 计算损失
            # ① 计算教师模型的和学生模型概率分布
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            # ② 计算硬损失
            hard_loss = cross_loss_fn(student_log_probs, teacher_labels)
            # ③ 计算软损失(参数1: 学生预测的概率  参数2: 教师预测的概率  参数3: 使用批次平均方式计算损失  参数4: 对softmax结果取对数)
            soft_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True)
            # ④ 计算总损失
            loss = (1 - alpha) * hard_loss + alpha * (T * T) * soft_loss

            # 5.3.4 梯度清零
            optimizer.zero_grad()
            # 5.3.5 反向传播
            loss.backward()
            # 5.3.6 梯度更新
            optimizer.step()

            # 5.4 打印日志
            # 5.4.1 累加损失
            total_loss += loss.item()
            # 5.4.2 添加预测值和真实值
            pred_labels.extend(student_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # 5.4.3 每隔10个批次或者最后一个批次打印日志(实时查看模型训练情况)
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_dataloader) - 1:
                # ①计算平均损失
                batch_count = batch_idx % 10 + 1
                avg_loss = total_loss / batch_count
                # ②计算准确率
                acc = accuracy_score(true_labels, pred_labels)
                # ③计算F1 Score
                f1 = f1_score(true_labels, pred_labels, average='macro')
                # ④打印日志
                print(f"\n轮数: {epoch + 1}, 批次数: {batch_idx + 1}, 平均损失: {avg_loss:.4f}, 准确率:{acc:.4f}, F1 Score:{f1:.4f}")
                # ⑤重置总损失、预测值、真实值
                total_loss, pred_labels, true_labels = 0.0, [], []

            # 5.5 评估和保存模型(每100个批次或者最后一个批次)
            if (batch_idx + 1) % 100 == 0 or batch_idx == len(train_dataloader) - 1:
                # 5.5.1 评估模型
                report, f1, acc, precision, recall = eval_model(student_model, dev_dataloader, config.device)
                # 5.5.2 打印评估结果
                print("验证集评估报告：\n", report)
                print(f"验证集f1，accuracy, precision, recall: {f1:.4f}, {acc:.4f}, {precision:.4f}, {recall:.4f}")
                # 5.5.3 保存模型
                if f1 > best_f1:
                    # 更新f1值
                    best_f1 = f1
                    # 保存模型
                    torch.save(student_model.state_dict(), config.distilled_model_save_path)
                    print('❤'*5,'模型保存成功','❤'*5)
                # 5.5.4 将模型切换回训练模式
                student_model.train()

    print('模型训练完成......')


if __name__ == '__main__':
    distill_model()

