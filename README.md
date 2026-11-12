# Chinese Short Text Classification

中文短文本主题分类项目，包含 TF-IDF + Random Forest、FastText、BERT 分类和 BERT → BiLSTM 知识蒸馏。建议 GitHub 仓库名：`Chinese-Short-Text-Classification`。

## 目录

```text
1.data/                 数据集
2.rf/                   TF-IDF + Random Forest
3.bert/                 BERT 分类及 Web 服务/客户端
4.bert_distill/         BERT Teacher 与 BiLSTM Student 蒸馏
5.fasttext/             FastText
```

## 数据

数据格式为 `label<TAB>text`，类别从 0 开始，编号顺序对应 `class.txt`。固定划分为 train 105,000 条、dev 22,500 条、test 22,500 条。训练和评估沿用原划分，不重新切分。

数据文件位于 `1.data/data/`。

## 环境

项目使用 Python、PyTorch、Transformers、scikit-learn、Flask 和 Streamlit。Intel Arc XPU 用户使用已配置好的 `bert_intel` Conda 环境：

```powershell
conda activate bert_intel
```

项目在 XPU 可用时优先使用 XPU，否则使用 CPU。若遇到 `ZE_RESULT_ERROR_UNINITIALIZED` 或 `could not make an engine with allocator`，请检查 Intel 显卡驱动和 Level Zero 运行时。

## 运行

脚本使用相对路径，请从相应目录启动。

```powershell
cd 2.rf
python data_process.py
python train.py
```

```powershell
cd 3.bert
python train.py
```

```powershell
cd 5.fasttext
python data_process.py
python train.py
```

蒸馏复用 `3.bert/save_model/bert_model.pt` Teacher 权重：

```powershell
cd 4.bert_distill
python bilstm_distill_bert.py
python predict_func.py
```

BERT 服务端和 Streamlit 客户端分别在 `3.bert` 目录运行 `python server.py`、`python run_web_client.py`。

## 模型文件

Git 不跟踪 `.pt`、`.pkl`、`.bin` 或 `.safetensors` 权重文件，避免将大型模型写进仓库历史。训练权重及本地 BERT checkpoint 需单独准备；现有 Teacher 权重文件名为 `bert_model.pt`。

## 一键提交

在 Windows 文件资源管理器中双击项目根目录的 `一键提交.bat`，按提示输入提交说明。脚本会提交代码变更并推送到 `codex-project-upload` 分支；模型权重、缓存和生成数据不会被重新提交。


