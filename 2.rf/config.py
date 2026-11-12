
"""
该脚本作用:
    --项目的配置文件(防止硬编码)
    --配置项目中的数据路径
"""

class Config():
    def __init__(self):
        # 1. 原始数据文件路径
        self.train_data_path = '../1.data/data/train.txt'
        self.test_data_path = '../1.data/data/test.txt'
        self.dev_data_path = '../1.data/data/dev.txt'
        self.class_data_path = '../1.data/data/class.txt'

        # 2. 处理后的数据文件路径
        self.processed_train_data_path = 'processed_data/processed_train.txt'
        self.processed_test_data_path = 'processed_data/processed_test.txt'
        self.processed_dev_data_path = 'processed_data/processed_dev.txt'

        # 3. 模型保存路径
        self.tfidf_model_save_path = 'save_model/tfidf_model.pkl'
        self.rf_model_save_path = 'save_model/rf_model.pkl'

        # 4. 结果保存路径
        self.result_path = 'result/predict_result.csv'


if __name__ == '__main__':
    config = Config()
    print(config.train_data_path)
    print(config.test_data_path)
    print(config.dev_data_path)
    print(config.class_data_path)

