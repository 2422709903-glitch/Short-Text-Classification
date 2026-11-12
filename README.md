# 中文短文本分类实验

包含 Random Forest、BERT、FastText 和 BERT 蒸馏 BiLSTM。根目录的 Flask 服务提供网页客户端，可以对照这四个模型的类别概率。

## 数据与类别

原始数据位于 `1.data/data/`，格式是 `label<TAB>text`，26 个类别的编号按 `class.txt` 行顺序从 0 开始。原始 `train.txt`、`dev.txt`、`test.txt` 都会保留。

检查发现原训练集类别数量差异明显：最少类别 77 条，最多类别 14,383 条。运行下面脚本会从训练集少数类中可复现地随机重复抽样，将少数类补到至少 1,000 条，生成 `train_balanced.txt`。脚本不修改原始文件，也不改动验证集和测试集；模型验证仍使用原始 `dev.txt`。

```powershell
python 1.data/balance_train.py
```

## FastText

旧项目拷贝来的 FastText 配置使用了其他项目路径和数据列顺序。现在代码读取当前项目的 `label<TAB>text` 并使用 26 个类别；旧的 10 类模型和旧预处理文件已清理。训练脚本生成的平衡版权重与验证集选出的当前部署权重分开保存。

`processed_data/` 中 `*_char` 是逐字切分，`*_word` 是 jieba 分词；`train/dev/test` 分别对应三个固定数据集。当前版本文件名带 `_26class`，训练集处理结果来自 `train_balanced.txt`，dev/test 仍来自原始划分。

```powershell
cd 5.fasttext
python data_process.py
python train.py
```

FastText 训练 3 轮，脚本只在验证集评估，不用测试集挑选模型。原始测试集保留给最终模型对比。

## 统一网页客户端与服务端

先进入项目根目录，使用同时包含 Flask、PyTorch、Transformers、scikit-learn、jieba 和 fasttext 的 Python 环境。当前 `bert_intel` 环境缺少 fasttext 包；在该环境里 FastText 会显示为不可用。可以改用装齐这些依赖的环境启动：

```powershell
conda activate python312
python server.py
```

另开一个终端可运行 `python run_web_client.py` 自动打开网页，也可以手动访问 `http://127.0.0.1:5000`。页面只比较 RF、BERT、FastText 和蒸馏 BiLSTM；可并排查看 26 类概率、展开全部类别、复制结果、导出 CSV，并查看最近预测记录。缺少权重或依赖的模型会显示为不可用。

浏览器打开 `http://127.0.0.1:5000`。服务默认在 CPU 上运行 BERT 与 BiLSTM 推理，绕开当前 Intel XPU 环境中 oneDNN/Level Zero 的运行时故障。启动环境缺少某模型依赖或权重时，该模型会显示为不可用，其余模型仍可使用。

## 各模型训练入口

```powershell
# Random Forest：预处理后在补齐训练集训练，只在 dev 评估
cd 2.rf
python data_process.py
python train.py

# BERT：复用本地 bert-base-chinese，使用补齐训练集训练 3 轮，验证集选择最佳权重
cd ..\3.bert
python train.py

# BERT Teacher 蒸馏 BiLSTM Student，训练 3 轮
cd ..\4.bert_distill
python bilstm_distill_bert.py

```

RF、BERT 和蒸馏 BiLSTM 的补齐版权重都使用独立文件名保存，原有模型不会被覆盖。蒸馏 BiLSTM 固定在 CPU 上运行，避开 Intel XPU 的 LSTM allocator 错误。

## 文件结构

```text
1.data/                 数据、类别映射、训练集补齐脚本
2.rf/                   TF-IDF + Random Forest
3.bert/                 BERT 分类
4.bert_distill/         BERT Teacher 与 BiLSTM 蒸馏
5.fasttext/             FastText 分类与独立预测接口
web_app.py              根目录统一 Flask 服务
server.py               服务端启动入口
run_web_client.py       网页客户端启动入口
web/                    网页客户端
```

