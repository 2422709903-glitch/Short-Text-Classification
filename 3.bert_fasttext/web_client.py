"""Streamlit client for the BERT classification service."""
import os

import requests
import streamlit as st

API_URL = os.getenv('BERT_FASTTEXT_API_URL', 'http://127.0.0.1:5001/predict')

st.set_page_config(page_title='BERT 文本分类', page_icon='🧠', layout='centered')
st.title('🧠 BERT 中文文本分类')
st.caption('输入一段中文文本，服务端将预测其所属主题。')

text = st.text_area('待分类文本', placeholder='例如：周杰伦最新歌曲演唱会现场', height=140)

if st.button('开始分类', type='primary', use_container_width=True):
    if not text.strip():
        st.warning('请先输入需要分类的文本。')
    else:
        try:
            with st.spinner('正在分析…'):
                response = requests.post(API_URL, json={'text': text.strip()}, timeout=60)
            if response.ok:
                result = response.json()
                st.success(f"预测类别：{result['pred_class']}")
            else:
                st.error(response.json().get('error', '预测请求失败'))
        except requests.exceptions.ConnectionError:
            st.error('无法连接 Flask 服务，请先启动 server.py。')
        except requests.exceptions.Timeout:
            st.error('请求超时，请稍后重试。')
        except (ValueError, KeyError):
            st.error('服务端返回的数据格式不正确。')

