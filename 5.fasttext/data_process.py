"""把当前项目的 label<TAB>text 数据转换成 FastText 训练格式。"""

import os

import jieba

from config import Config


config = Config()


def process_data(raw_data_path, processed_data_path, is_char=True):
    """生成每行 ``__label__类别 分词文本`` 的 FastText 文件。"""
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    line_number = 0
    with open(raw_data_path, mode='r', encoding='utf-8') as source, \
            open(processed_data_path, mode='w', encoding='utf-8', newline='\n') as target:
        for line_number, raw_line in enumerate(source, start=1):
            raw_line = raw_line.rstrip('\r\n')
            if not raw_line:
                continue
            parts = raw_line.split('\t', 1)
            if len(parts) != 2:
                raise ValueError(f'{raw_data_path}:{line_number}: 数据行必须为 label<TAB>text')

            label_text, text = parts
            try:
                label_id = int(label_text)
            except ValueError as exc:
                raise ValueError(f'{raw_data_path}:{line_number}: 标签不是整数：{label_text!r}') from exc
            if label_id not in config.class_dict:
                raise ValueError(f'{raw_data_path}:{line_number}: 标签 {label_id} 超出类别范围')

            label = config.class_dict[label_id]
            # FastText 的监督标签不应包含空格；当前 class.txt 类别名均为单词。
            if any(char.isspace() for char in label):
                raise ValueError(f'class.txt 中类别名不能包含空格：{label!r}')
            tokens = list(text) if is_char else jieba.lcut(text)
            target.write(f"__label__{label} {' '.join(tokens)}\n")


if __name__ == '__main__':
    process_data(config.train_data_path, config.processed_char_train_data_path)
    process_data(config.train_data_path, config.processed_word_train_data_path, is_char=False)
    process_data(config.test_data_path, config.processed_char_test_data_path)
    process_data(config.test_data_path, config.processed_word_test_data_path, is_char=False)
    process_data(config.dev_data_path, config.processed_char_dev_data_path)
    process_data(config.dev_data_path, config.processed_word_dev_data_path, is_char=False)
    print('FastText 数据处理完成。')

