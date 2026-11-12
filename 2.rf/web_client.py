
"""
该脚本的作用:
    1. 使用 Streamlit 创建 Web 页面
    2. 使用 requests 发送 POST 请求进行预测
    3. 接收预测结果并显示
"""

import time
import requests
import streamlit as st

# 0. 后端接口地址
url = 'http://127.0.0.1:5000/predict'


# 1. 页面基本配置
st.set_page_config(
    page_title='VSQ 智能视频搜索分类系统',
    page_icon='🎬',
    layout='wide'
)


# 2. 自定义 CSS 样式
st.markdown("""
<style>

/* 页面背景 */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, #172554 0%, transparent 30%),
        radial-gradient(circle at 90% 20%, #312e81 0%, transparent 25%),
        linear-gradient(135deg, #080d1b 0%, #111827 55%, #0b1120 100%);
    color: #f8fafc;
}

/* 主页面宽度 */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* 标题 */
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: 2px;
    margin-bottom: 5px;
}

.sub-title {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

/* 输入区域 */
.input-card {
    background: rgba(20, 30, 55, 0.85);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
}

/* 结果卡片 */
.result-card {
    background: linear-gradient(
        135deg,
        rgba(30, 64, 175, 0.35),
        rgba(76, 29, 149, 0.35)
    );
    border: 1px solid rgba(96, 165, 250, 0.5);
    border-radius: 18px;
    padding: 28px;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.12);
}

.result-label {
    color: #93c5fd;
    font-size: 14px;
    letter-spacing: 3px;
}

.result-value {
    color: #ffffff;
    font-size: 42px;
    font-weight: 800;
    margin-top: 10px;
}

/* 信息卡片 */
.info-card {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.info-number {
    color: #60a5fa;
    font-size: 23px;
    font-weight: 800;
}

.info-text {
    color: #94a3b8;
    font-size: 13px;
}

/* 按钮 */
.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 16px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border: none;
    color: white;
    box-shadow: 0 0 20px rgba(96, 165, 250, 0.45);
    transform: translateY(-2px);
}

/* 文本输入框 */
.stTextArea textarea {
    background: rgba(8, 15, 32, 0.85);
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 10px;
}

.stTextArea textarea:focus {
    border-color: #60a5fa;
    box-shadow: 0 0 0 1px #60a5fa;
}

/* 底部 */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# 3. 页面顶部
st.markdown("""
<div style="display:flex; align-items:center; gap:12px;">
    <div style="
        background:linear-gradient(135deg,#2563eb,#7c3aed);
        width:48px;height:48px;border-radius:13px;
        display:flex;align-items:center;justify-content:center;
        font-size:25px;">
        🎬
    </div>
    <div>
        <div style="font-size:13px;color:#60a5fa;letter-spacing:3px;">
            VSQ INTELLIGENCE
        </div>
        <div style="font-size:12px;color:#94a3b8;">
            VIDEO QUERY CLASSIFICATION SYSTEM
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">智能视频搜索分类系统</div>
<div class="sub-title">
    基于 TF-IDF 与 Random Forest 的中文视频搜索 Query 主题分类
</div>
""", unsafe_allow_html=True)


# 4. 系统信息
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">26</div>
        <div class="info-text">主题分类类别</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">150K</div>
        <div class="info-text">数据集样本数量</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">RF</div>
        <div class="info-text">Random Forest 模型</div>
    </div>
    """, unsafe_allow_html=True)


st.write('')
st.write('')


# 5. 输入区域
st.markdown("""
<div style="color:#cbd5e1;font-size:19px;font-weight:700;">
    ✦ 输入视频搜索内容
</div>
<div style="color:#64748b;font-size:13px;margin-top:5px;">
    输入一段中文搜索 Query，系统将自动识别其所属主题
</div>
""", unsafe_allow_html=True)


# 5.1 用户输入框
text = st.text_area(
    '请输入文本',
    placeholder='例如：周杰伦最新歌曲演唱会现场',
    height=120,
    label_visibility='collapsed'
)


# 5.2 示例文本
st.markdown(
    '<div style="color:#94a3b8;font-size:13px;">'
    '示例：周杰伦最新歌曲演唱会现场 · 王者荣耀精彩操作 · '
    '汽车保养知识分享'
    '</div>',
    unsafe_allow_html=True
)

st.write('')


# 6. 预测按钮
if st.button('✦  开始智能分类'):

    if not text.strip():
        st.warning('请先输入需要分类的文本！')

    else:
        try:
            # 记录开始时间
            start_time = time.time()

            # 发送 POST 请求
            with st.spinner('AI 正在分析文本，请稍候...'):
                response = requests.post(
                    url,
                    json={'text': text},
                    timeout=30
                )

            # 计算推理耗时
            cost_time = (time.time() - start_time) * 1000

            # 判断请求是否成功
            if response.status_code == 200:

                result = response.json()

                # 获取预测结果
                pred_class = result['pred_class']

                # 显示预测结果卡片
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">
                        PREDICTION RESULT
                    </div>
                    <div class="result-value">
                        {pred_class}
                    </div>
                    <div style="color:#94a3b8;margin-top:12px;">
                        文本主题识别完成
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 显示推理时间
                st.write('')
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label='模型推理耗时',
                        value=f'{cost_time:.2f} ms'
                    )

                with col2:
                    st.metric(
                        label='预测类别数量',
                        value='26 类'
                    )

                # 显示原始文本
                with st.expander('查看本次预测文本'):
                    st.write(text)

            else:
                st.error(response.json().get('error', '预测请求失败'))

        except requests.exceptions.ConnectionError:
            st.error('无法连接 Flask 服务，请先启动 server.py！')

        except requests.exceptions.Timeout:
            st.error('请求超时，请稍后重试！')

        except Exception as e:
            st.error(f'预测出现异常：{e}')


# 7. 底部信息
st.markdown("""
<div class="footer">
    VSQ TEXT CLASSIFICATION SYSTEM
    <br>
    POWERED BY TF-IDF & RANDOM FOREST
</div>
""", unsafe_allow_html=True)

