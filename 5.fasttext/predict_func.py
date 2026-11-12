"""FastText 单条文本预测。"""

from pathlib import Path

import fasttext
import jieba

from config import Config


config = Config()
model_path = Path(config.ft_model_word_selected_path)
if not model_path.is_file():
    raise FileNotFoundError(
        f'当前项目的 FastText 模型尚未训练：{model_path}\n'
        '请先运行 data_process.py，再运行 train.py。旧的 10 类模型不会自动加载。'
    )
model = fasttext.load_model(str(model_path))


def predict(input_data):
    """输入 ``{'text': '...'} ``，返回文本及预测类别。"""
    if not isinstance(input_data, dict) or not isinstance(input_data.get('text'), str):
        raise ValueError("请求内容必须包含字符串字段 'text'")

    text = input_data['text'].strip()
    if not text:
        raise ValueError('待分类文本不能为空')

    words = ' '.join(jieba.lcut(text))
    labels, probabilities = model.predict(words, k=1)
    predicted_class = labels[0].removeprefix('__label__')
    result = dict(input_data)
    result['pred_class'] = predicted_class
    result['probability'] = float(probabilities[0])
    return result


if __name__ == '__main__':
    print('模型类别数：', len(model.labels))
    print(predict({'text': '吴亦凡被抓了'}))

