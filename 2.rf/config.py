"""Random Forest 路径配置。"""

from pathlib import Path


class Config:
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent
        self.data_dir = self.project_dir.parent / '1.data' / 'data'
        original_train = self.data_dir / 'train.txt'
        balanced_train = self.data_dir / 'train_balanced.txt'
        self.train_data_path = str(balanced_train if balanced_train.is_file() else original_train)
        self.test_data_path = str(self.data_dir / 'test.txt')
        self.dev_data_path = str(self.data_dir / 'dev.txt')
        self.class_data_path = str(self.data_dir / 'class.txt')

        processed_dir = self.project_dir / 'processed_data'
        self.processed_train_data_path = str(processed_dir / 'processed_train.csv')
        self.processed_test_data_path = str(processed_dir / 'processed_test.csv')
        self.processed_dev_data_path = str(processed_dir / 'processed_dev.csv')

        model_dir = self.project_dir / 'save_model'
        # 旧模型路径保留，新的训练结果单独保存，避免覆盖旧实验。
        self.tfidf_model_save_path = str(model_dir / 'tfidf_model.pkl')
        self.rf_model_save_path = str(model_dir / 'rf_model.pkl')
        self.balanced_tfidf_model_save_path = str(model_dir / 'tfidf_model_balanced.pkl')
        self.balanced_rf_model_save_path = str(model_dir / 'rf_model_balanced.pkl')

        self.result_path = str(self.project_dir / 'result' / 'predict_result.csv')


if __name__ == '__main__':
    config = Config()
    print(config.train_data_path)
    print(config.test_data_path)
    print(config.dev_data_path)
    print(config.class_data_path)

