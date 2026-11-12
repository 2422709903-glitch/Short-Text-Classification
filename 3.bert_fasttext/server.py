"""Flask API for BERT text classification."""
from flask import Flask, jsonify, request

from predict_func import predict

app = Flask(__name__)


@app.get('/health')
def health():
    return jsonify({'status': 'ok'})


@app.post('/predict')
def predict_text():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get('text'), str) or not data['text'].strip():
        return jsonify({'error': '请输入需要分类的文本'}), 400
    try:
        return jsonify(predict({'text': data['text'].strip()}))
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify({'error': '模型预测失败'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)

