"""根目录统一预测服务：RF、BERT、FastText 与蒸馏 BiLSTM。"""

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

import jieba
import numpy as np
from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / 'web'
DATA_DIR = ROOT / '1.data' / 'data'
CLASS_NAMES = [line.strip() for line in (DATA_DIR / 'class.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

PATHS = {
    'rf_model': ROOT / '2.rf' / 'save_model' / 'rf_model.pkl',
    'rf_tfidf': ROOT / '2.rf' / 'save_model' / 'tfidf_model.pkl',
    'bert_base': ROOT / '3.bert' / 'bert-base-chinese',
    'bert_weights': ROOT / '3.bert' / 'save_model' / 'bert_model_balanced.pt',
    'fasttext': ROOT / '5.fasttext' / 'save_model' / 'ft_model_word_26class.bin',
    'distill_base': ROOT / '4.bert_distill' / 'bert-base-chinese',
    'distill_weights': ROOT / '4.bert_distill' / 'save_model' / 'distilled_tmf_model_balanced.pt',
}

BALANCED_RF_MODEL = ROOT / '2.rf' / 'save_model' / 'rf_model_balanced.pkl'
BALANCED_RF_TFIDF = ROOT / '2.rf' / 'save_model' / 'tfidf_model_balanced.pkl'
BALANCED_FASTTEXT = ROOT / '5.fasttext' / 'save_model' / 'ft_model_word_balanced_26class.bin'
LEGACY_BERT = ROOT / '3.bert' / 'save_model' / 'bert_model.pt'
LEGACY_DISTILL = ROOT / '4.bert_distill' / 'save_model' / 'distilled_tmf_model.pt'

MODEL_NAMES = {
    'rf': 'Random Forest',
    'bert': 'BERT',
    'fasttext': 'FastText',
    'distill_bilstm': '蒸馏 BiLSTM',
}

app = Flask(__name__)


def model_status():
    """报告本项目四组模型的权重和运行依赖是否齐备。"""
    statuses = {
        'rf': PATHS['rf_model'].is_file() and PATHS['rf_tfidf'].is_file() and find_spec('sklearn') is not None,
        'bert': active_bert_path().is_file() and (PATHS['bert_base'] / 'config.json').is_file() and find_spec('torch') is not None and find_spec('transformers') is not None,
        'fasttext': active_fasttext_path().is_file() and find_spec('fasttext') is not None,
        'distill_bilstm': active_distill_path().is_file() and (PATHS['distill_base'] / 'config.json').is_file() and find_spec('torch') is not None and find_spec('transformers') is not None,
    }
    return statuses


def active_rf_paths():
    if BALANCED_RF_MODEL.is_file() and BALANCED_RF_TFIDF.is_file():
        return BALANCED_RF_TFIDF, BALANCED_RF_MODEL
    return PATHS['rf_tfidf'], PATHS['rf_model']


def active_fasttext_path():
    if PATHS['fasttext'].is_file():
        return PATHS['fasttext']
    return BALANCED_FASTTEXT


def active_bert_path():
    return PATHS['bert_weights'] if PATHS['bert_weights'].is_file() else LEGACY_BERT


def active_distill_path():
    return PATHS['distill_weights'] if PATHS['distill_weights'].is_file() else LEGACY_DISTILL


@lru_cache(maxsize=1)
def load_rf():
    import pickle

    tfidf_path, model_path = active_rf_paths()
    with open(tfidf_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    return vectorizer, classifier


@lru_cache(maxsize=1)
def load_fasttext():
    import fasttext

    return fasttext.load_model(str(active_fasttext_path()))


def load_torch_classifier(kind):
    import torch
    import torch.nn as nn
    from transformers import BertConfig, BertModel, BertTokenizer

    torch.set_num_threads(min(torch.get_num_threads(), 4))

    if kind == 'bert':
        base_dir, weights_path = PATHS['bert_base'], active_bert_path()
        bert_config = BertConfig.from_pretrained(str(base_dir))

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained(str(base_dir), attn_implementation='eager')
                self.linear = nn.Linear(bert_config.hidden_size, len(CLASS_NAMES))

            def forward(self, input_ids, attention_mask):
                output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                return self.linear(output.pooler_output)

        tokenizer = BertTokenizer.from_pretrained(str(base_dir))
    else:
        base_dir, weights_path = PATHS['distill_base'], active_distill_path()
        bert_config = BertConfig.from_pretrained(str(base_dir))

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeding = nn.Embedding(bert_config.vocab_size, 128)
                self.bilstm = nn.LSTM(128, 256, 3, bidirectional=True, batch_first=True)
                self.dropout = nn.Dropout(0.3)
                self.linear = nn.Linear(512, len(CLASS_NAMES))

            def forward(self, input_ids, attention_mask):
                valid = attention_mask.bool() & (input_ids != 101) & (input_ids != 102)
                mask = valid.unsqueeze(-1)
                embedded = self.embeding(input_ids) * mask
                output, _ = self.bilstm(embedded)
                output = output * mask
                pooled = output.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                return self.linear(self.dropout(pooled))

        tokenizer = BertTokenizer.from_pretrained(str(base_dir))

    model = Classifier()
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=2)
def load_torch_model(kind):
    return load_torch_classifier(kind)


def _normalize(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probabilities.size != len(CLASS_NAMES):
        raise ValueError(f'模型输出了 {probabilities.size} 个类别概率，预期为 {len(CLASS_NAMES)}')
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if total > 0:
        probabilities /= total
    return probabilities


def predict_rf(text):
    vectorizer, classifier = load_rf()
    features = vectorizer.transform([' '.join(jieba.lcut(text))])
    raw = classifier.predict_proba(features)[0]
    scores = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for class_id, probability in zip(classifier.classes_, raw):
        scores[int(class_id)] = probability
    return _normalize(scores)


def predict_fasttext(text):
    model = load_fasttext()
    tokens = ' '.join(jieba.lcut(text))
    labels, values = model.predict(tokens, k=len(CLASS_NAMES))
    scores = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for label, probability in zip(labels, values):
        name = label.removeprefix('__label__')
        if name in CLASS_TO_ID:
            scores[CLASS_TO_ID[name]] = float(probability)
    return _normalize(scores)


def predict_torch(kind, text):
    import torch

    model, tokenizer = load_torch_model(kind)
    encoded = tokenizer(
        [text], add_special_tokens=True, max_length=32,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    with torch.inference_mode():
        logits = model(encoded['input_ids'], encoded['attention_mask'])
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


def predict_one(kind, text):
    started = perf_counter()
    if kind == 'rf':
        scores = predict_rf(text)
    elif kind == 'fasttext':
        scores = predict_fasttext(text)
    elif kind in ('bert', 'distill_bilstm'):
        scores = predict_torch(kind, text)
    else:
        raise ValueError(f'未知模型：{kind}')

    scores = _normalize(scores)
    order = np.argsort(scores)[::-1]
    return {
        'key': kind,
        'name': MODEL_NAMES[kind],
        'predicted_class': CLASS_NAMES[int(order[0])],
        'confidence': float(scores[order[0]]),
        'latency_ms': (perf_counter() - started) * 1000,
        'probabilities': [
            {'label': CLASS_NAMES[int(index)], 'class_id': int(index), 'probability': float(scores[index])}
            for index in order
        ],
    }


@app.get('/')
def index():
    return send_from_directory(WEB_DIR, 'index.html')


@app.get('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(WEB_DIR, filename)


@app.get('/api/models')
def api_models():
    statuses = model_status()
    return jsonify({
        'models': [
            {'key': key, 'name': name, 'available': statuses[key]}
            for key, name in MODEL_NAMES.items()
        ],
        'class_count': len(CLASS_NAMES),
        'classes': CLASS_NAMES,
    })


@app.post('/api/predict')
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get('text')
    selected = payload.get('models') or []
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': '请输入要分类的中文短文本。'}), 400
    if not isinstance(selected, list) or not selected:
        return jsonify({'error': '至少选择一个模型。'}), 400
    try:
        statuses = model_status()
        results = []
        for key in dict.fromkeys(selected):
            if key not in MODEL_NAMES:
                raise ValueError(f'未知模型：{key}')
            if not statuses[key]:
                raise ValueError(f'{MODEL_NAMES[key]} 的模型文件尚未准备好。')
            results.append(predict_one(key, text.strip()))
        return jsonify({'text': text.strip(), 'results': results})
    except Exception as exc:
        app.logger.exception('Prediction failed')
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

