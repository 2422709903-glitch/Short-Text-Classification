"""
 该脚本的作用：
  1.使用requests库发送post请求   pip install requests
  2.请求flask后端接口进行预测
  3.接受结果
"""

import requests

# 1.定义要访问后端的接口地址
url = 'http://127.0.0.1:5000/predict'

# 2.发送post请求
try:
    # 2.1 接受用户控制台输入
    user_input = input('请输入要预测的新闻标题:')
    # 2.2 发送post请求
    response = requests.post(url, json={'text': user_input})
    print(response.json())
except Exception as e:
    print(f'出错了,请联系管理员')
