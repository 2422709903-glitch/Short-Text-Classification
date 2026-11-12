"""
1. 创建Flask应用
2. 定义视图函数接受用户的请求(json)
3. 调用预测方式进行预测
4. 将预测结果返回
"""
from flask import Flask, request, jsonify
from predict_func import predict

# 1.创建Flask对象
app = Flask(__name__)

# 2.创建视图函数
@app.route('/predict', methods=['POST'])  # http://127.0.0.1:5000/predict
def predict_news():
    # 接受用户的请求(json)
    data = request.get_json()
    # 调用预测函数进行结果预测
    result = predict(data)
    # 创建json对象并返回
    return jsonify(result)

# 3. 启动应用
if __name__ == '__main__':
    app.run(host='127.0.0.1')
