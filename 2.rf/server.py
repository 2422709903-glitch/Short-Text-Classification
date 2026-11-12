
"""
该脚本作用:
    1. 创建 Flask 应用
    2. 定义视图函数接受用户的请求(json)
    3. 调用预测函数进行预测
    4. 将预测结果返回
"""

from flask import Flask, request, jsonify
from predict_func import predict

# 1. 创建 Flask 对象
app = Flask(__name__)


# 2. 创建视图函数
@app.route('/predict', methods=['POST'])
def predict_text():
    # 2.1 接受用户请求
    data = request.get_json()

    # 2.2 判断输入是否为空
    if not data or not data.get('text', '').strip():
        return jsonify({'error': '请输入需要分类的文本'}), 400

    # 2.3 调用预测函数
    result = predict(data)

    # 2.4 返回预测结果
    return jsonify(result)


# 3. 启动应用
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

