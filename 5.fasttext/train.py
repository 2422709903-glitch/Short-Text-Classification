import fasttext
from config import Config
import numpy

config = Config()


def print_model_info(model):
    # 查看某个词最相近的词
    print(model.get_nearest_neighbors("马"))
    # 查看某个字的词向量
    print(model.get_word_vector('我'))
    # 查看词向量的维度
    print(model.get_dimension())
    # 查看模型的标签
    print(model.labels)
    print(len(model.labels))
    # 产看词表
    print(model.words)
    print(len(model.words))
    # 查看词频
    print(model.get_words(include_freq=True))
    # print(list(zip((*model.get_words(include_freq=True)))))
    print(model.get_subwords('我 喜 欢 玩 游 戏'))


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
    print(model.test(config.processed_char_test_data_path))
    # 4.预测
    print(model.predict("词 汇 阅 读 是 关 键 0 8 年 考 研 暑 期 英 语 复 习 全 指 南"))
    # 5.模型保存
    model.save_model(config.ft_model_char_default_path)


# 1. 训练FastText模型(1.默认参数 2.使用词级分词数据)
def train_fasttext_default_word():
    # 1. 训练
    model = fasttext.train_supervised(
        input=config.processed_word_train_data_path,  # 训练数据路径
        dim=10,  # 词向量维度
        minn=1,  # 最小子词长度
        maxn=4  # 最大子词长度
    )
    # 2.测试 -->(10000, 0.9171, 0.9171)
    print(model.test(config.processed_word_test_data_path))
    # 3.预测
    print(model.predict("词汇 阅读 是 关键 08年 考研 暑期 英语 复习 全指南"))
    # 4.模型保存
    model.save_model(config.ft_model_word_default_path)
if __name__ == '__main__':
    train_fasttext_default_word()
