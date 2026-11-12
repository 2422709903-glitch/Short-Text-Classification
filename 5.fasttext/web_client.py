"""
该脚本的作用:
    1.使用streamlit创建一个web页面进行用户输入
    2.使用requests库发送post请求进行预测
    3.接受结果并显示在web页面上
"""
import time

import requests
import streamlit as st

# 0 url地址
url = 'http://127.0.0.1:5000/predict'

# 1.创建web页面

# 1.1 页面标题
st.title('投满分新闻分类系统')

# 1.2 提示语
st.write('♥'*5,'嘎嘎好用', '♥'*5)

# 1.3 用户输入框
text = st.text_input('请输入要预测的新闻标题:')

# 1.4 预测按钮
if st.button('点击获取分类'):
    # 记录开始时间
    start_time = time.time()
    try:
        # 发送post请求
        response = requests.post(url, json={'text': text})
        # 计算推理时间
        cost_time = (time.time() - start_time) * 1000
        # 显示耗时
        st.write(f'推理耗时:{cost_time:.2f}ms')
        # 显示预测结果
        st.write(f"预测结果:{response.json()['pred_class']}")
    except Exception as e:
        print(f'出错了,请联系管理员')



"""
推理耗时:8.66ms
模型大小:75M
"""




