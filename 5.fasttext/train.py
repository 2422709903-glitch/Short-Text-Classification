import fasttext
from config import Config
import numpy

config = Config()


def print_model_info(model):
    print(f'类别数：{len(model.labels)}')
    print(f'类别：{model.labels}')
    print(f'词向量维度：{model.get_dimension()}')
    print(f'词表大小：{len(model.words)}')


# 1. 训练FastText模型(1.默认参数 2.使用字级分词数据)
def train_fasttext_dafault_char():
    # 1.训练
    model = fasttext.train_supervised(input=config.processed_char_train_data_path,
                                      dim=10,
                                      minn=1,
                                      maxn=4)
    # 2.打印模型关键信息
    print_model_info(model)
    # 3.测试
    print('验证集指标：', model.test(config.processed_char_dev_data_path))
    # 4.预测
    print(model.predict("词 汇 阅 读 是 关 键 0 8 年 考 研 暑 期 英 语 复 习 全 指 南"))
    # 5.模型保存
    model.save_model(config.ft_model_char_balanced_path)


# 1. 训练FastText模型(1.默认参数 2.使用词级分词数据)
def train_fasttext_default_word():
    # 1. 训练
    model = fasttext.train_supervised(
        input=config.processed_word_train_data_path,  # 训练数据路径
        epoch=3,  # 使用补齐训练集训练 3 轮
        dim=10,  # 词向量维度
        minn=1,  # 最小子词长度
        maxn=4  # 最大子词长度
    )
    # 验证集用于开发期评估；测试集留给最终统一评估脚本。
    print('验证集指标：', model.test(config.processed_word_dev_data_path))
    # 3.预测
    print(model.predict("词汇 阅读 是 关键 08年 考研 暑期 英语 复习 全指南"))
    # 4.模型保存
    model.save_model(config.ft_model_word_balanced_path)
if __name__ == '__main__':
    train_fasttext_default_word()

