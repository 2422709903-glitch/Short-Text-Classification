"""FastText 配置：路径均以当前文件位置为基准，避免依赖启动目录。"""

from pathlib import Path


class Config:
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent
        self.data_dir = self.project_dir.parent / '1.data' / 'data'

        # 原始训练文件保留；若已生成过采样集，训练时使用补齐后的副本。
        self.original_train_data_path = str(self.data_dir / 'train.txt')
        balanced_train_path = self.data_dir / 'train_balanced.txt'
        self.train_data_path = str(balanced_train_path if balanced_train_path.is_file() else self.data_dir / 'train.txt')
        # 当前项目原始格式为：label<TAB>text
        self.test_data_path = str(self.data_dir / 'test.txt')
        self.dev_data_path = str(self.data_dir / 'dev.txt')
        self.class_data_path = str(self.data_dir / 'class.txt')

        with open(self.class_data_path, mode='r', encoding='utf-8') as f:
            self.class_list = [line.strip() for line in f if line.strip()]
        self.class_dict = dict(enumerate(self.class_list))
        self.class_to_id = {name: index for index, name in self.class_dict.items()}

        # 使用带 26class 标记的新文件，保留从旧项目拷贝的处理结果。
        processed_dir = self.project_dir / 'processed_data'
        self.processed_char_train_data_path = str(processed_dir / 'processed_train_char_26class.txt')
        self.processed_word_train_data_path = str(processed_dir / 'processed_train_word_26class.txt')
        self.processed_char_test_data_path = str(processed_dir / 'processed_test_char_26class.txt')
        self.processed_word_test_data_path = str(processed_dir / 'processed_test_word_26class.txt')
        self.processed_char_dev_data_path = str(processed_dir / 'processed_dev_char_26class.txt')
        self.processed_word_dev_data_path = str(processed_dir / 'processed_dev_word_26class.txt')

        model_dir = self.project_dir / 'save_model'
        self.ft_model_char_balanced_path = str(model_dir / 'ft_model_char_balanced_26class.bin')
        self.ft_model_word_balanced_path = str(model_dir / 'ft_model_word_balanced_26class.bin')
        # 依据验证集 Macro-F1 选择的当前部署模型；训练脚本仍保存独立的 balanced 版本。
        self.ft_model_word_selected_path = str(model_dir / 'ft_model_word_26class.bin')


if __name__ == '__main__':
    config = Config()
    print(f'训练数据路径 --> {config.train_data_path}')
    print(f'验证数据路径 --> {config.dev_data_path}')
    print(f'测试数据路径 --> {config.test_data_path}')
    print(f'类别数量 --> {len(config.class_list)}')

