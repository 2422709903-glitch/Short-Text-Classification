"""
  该脚本作用:
     --项目的配置文件(防止硬编码)
     --配置项目中的数据路径
"""
class Config():
    def __init__(self):
        # 1.原始数据文件路径
        self.train_data_path = '../1.data/data/train.txt'  # 训练数据集文件位置
        self.test_data_path = '../1.data/data/test.txt'  # 测试数据集文件位置
        self.dev_data_path = '../1.data/data/dev.txt'  # 验证数据集文件位置
        self.class_data_path = '../1.data/data/class.txt'  # 类别数据文件位置
        self.stopwords_data_path = '../1.data/data/stopwords.txt'  # 停用词数据文件位置
        # 2.类别字典 --> {0:game, 1:'education',....}
        self.class_dict={index:class_name.strip() for index,class_name in enumerate(open(self.class_data_path,mode='r',encoding='utf-8'))}
        # 3.处理后的数据路径
        self.processed_char_train_data_path = 'processed_data/processed_train_char.txt'
        self.processed_word_train_data_path = 'processed_data/processed_train_word.txt'
        self.processed_char_test_data_path = 'processed_data/processed_test_char.txt'
        self.processed_word_test_data_path = 'processed_data/processed_test_word.txt'
        self.processed_char_dev_data_path = 'processed_data/processed_dev_char.txt'
        self.processed_word_dev_data_path = 'processed_data/processed_dev_word.txt'
        # 4.模型保存路径
        self.ft_model_char_default_path = 'save_model/ft_model_char_default.bin'
        self.ft_model_word_default_path = 'save_model/ft_model_word_default.bin'

if __name__ == '__main__':
    config=Config()
    print(f'训练数据路径 --> {config.train_data_path}')
    print(f'测试数据路径 --> {config.test_data_path}')
    print(f'验证数据路径 --> {config.dev_data_path}')

