"""
该脚本的作用
    --> 数据处理:将原始数据处理成FastText模型需要的格式
    --> FastText数据格式上的要求 : __label__education 词1 词2 词3 ...词n
    --> 我们原始数据格式: 词汇阅读是关键 08年考研暑期英语复习全指南	3


注意:
   --> FastText对于英文来说 一个单词就是一个词
   --> FastText对于中文句子有两种方式
       ①一个汉字一个词:  __label__education 词 汇 阅 读 是 关 键 0 8 年 考 研 暑 期 英 语 复 习 全 指 南
       ②使用jieba进行分词: __label__education 词汇 阅读 是 关键 08年 考研 暑期 英语 复习 全指南

数据处理
    训练集2种处理产生2个文件
    验证集2种处理产生2个文件
    测试集2种处理产生2个文件
"""
from config import Config
import jieba
# 1.创建配置对象
config = Config()
# 2.数据处理函数
def process_data(raw_data_path,processed_data_path,is_char=True):
    """
       :param raw_data_path: 原始数据文件路径
       :param process_data_path: 处理后数据文件路径
       :param is_char: 是否按照字符分词
     """
    #2.1打开原始文件和处理后的文件
    with open(raw_data_path,mode='r',encoding='utf-8') as fr,open(processed_data_path,mode='w',encoding='utf-8') as fw:
        #2.2读取原数据所有行所有列
        for line in fr.readlines():
            #2.3如果行数据为空则跳过
            if not line :
                continue
            # 2.4 切分数据 -> 词汇阅读是关键 08年考研暑期英语复习全指南	->3
            text_str,label_str=line.split('\t')
            #2.5处理标签
            ft_label='__label__'+config.class_dict[int(label_str)]
            #2.6处理特征列
            if is_char:
                #按照char处理
                ft_feature=' '.join(list(text_str))
            else:
                # 使用jieba分词
                ft_feature = ' '.join(jieba.lcut(text_str))
            new_line=ft_label+' '+ft_feature+'\n'
            fw.write(new_line)

# 3.处理数据
if __name__ == '__main__':
    # 训练集
    process_data(config.train_data_path, config.processed_char_train_data_path) # 按照char处理
    process_data(config.train_data_path, config.processed_word_train_data_path, is_char=False) # 按照jieba处理

    # 测试集
    process_data(config.test_data_path, config.processed_char_test_data_path) # 按照char处理
    process_data(config.test_data_path, config.processed_word_test_data_path,is_char=False) # 按照jieba处理

    # 验证集
    process_data(config.dev_data_path, config.processed_char_dev_data_path)  # 按照char处理
    process_data(config.dev_data_path, config.processed_word_dev_data_path, is_char=False) # 按照jieba处理

