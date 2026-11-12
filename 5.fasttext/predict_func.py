"""
该脚本作用: 实现预测函数，接受一条数据并返回预测结果
"""
import fasttext

from config import  Config
import pickle
import  jieba
#1.创建配置对象获取验证集数据路径
config=Config()
# 2.读取模型
model=fasttext.load_model(config.ft_model_word_default_path)
#3.预测函数
def predict(input_data):
    """
    :param  input_data: --> {'text':'体验2D巅峰 倚天屠龙记十大创新概览'}
    :return: --> {'text':'体验2D巅峰 倚天屠龙记十大创新概览', 'pred_class':'game'}
    """
    #3.1获取用户输入进行分词
    words=' '.join(jieba.lcut(input_data['text']))
    # 3.2 获取训练结果
    result=model.predict(words)
    ## 3.3 输入数据新增预测结果属性
    input_data['pred_class'] = result[0][0].replace('__label__', '')
    # 3.4 返回结果
    return input_data
if __name__ == '__main__':
    # input_data = {'text':'体验2D巅峰 倚天屠龙记十大创新概览'}
    print(model.labels)
    input_data = {'text':'吴亦凡被抓了'}
    print(predict(input_data))

