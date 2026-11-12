"""
该脚本作用:
    实现预测函数，接受一条数据并返回预测结果
"""

from config import Config
import pickle
import jieba

config=Config()
#读取模型
with open(config.tfidf_model_save_path,"rb")as f:
    tfidf=pickle.load(f)

with open(config.rf_model_save_path,'rb') as f:
    rf=pickle.load(f)

class_dict={index:class_name.strip() for index,class_name in enumerate(open(config.class_data_path,'r',encoding='utf-8'))}
def predict(input_data):
     """
    :param input_data: {'text': '某条中文视频搜索文本'}
    :return: {'text': '某条中文视频搜索文本', 'pred_class': '电视剧'}
    """
     words=' '.join(jieba.lcut(input_data['text']))

     features=tfidf.transform([words])

     pred_class=rf.predict(features)

     class_name=class_dict[int(pred_class[0])]

     input_data['pred_class']=class_name

     return input_data

if __name__ == '__main__':
    input_data = {'text': '周杰伦最新歌曲演唱会现场'}
    print(predict(input_data))



