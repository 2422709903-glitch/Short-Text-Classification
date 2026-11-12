
"""
该脚本作用:
    使用训练好的 TF-IDF 模型和随机森林模型进行预测
    使用 dev 数据集进行预测和评估
"""
import pickle
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import Config
# 设置 pandas 显示选项
pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行

pd.set_option('display.max_columns', None)# 确保所有列可见
config=Config()
dev_data_path=config.processed_dev_data_path

#读取tfidf模型
with open(config.tfidf_model_save_path,'rb')as f:
    tfidf=pickle.load(f)
#读取随机森林模型
with open(config.rf_model_save_path,'rb')as f:
    rf=pickle.load(f)

#读取验证集数据4
dev_data=pd.read_csv(dev_data_path)
# print(dev_data.head(10))
#获取分词列
words=dev_data['words']
#获取真实标签
y_true=dev_data['label']

print('⭐⭐⭐⭐⭐ 验证集数据读取完成 ⭐⭐⭐⭐⭐')
print('验证集样本数量',len(dev_data))
#tfidf特征提取
features=tfidf.transform(words)
print('特征提取完成')
#使用随机森林进行预测
y_pred=rf.predict(features)
print('预测完成')
#模型评估
print('准确率:',accuracy_score(y_true,y_pred))
print('精确率:',precision_score(y_true,y_pred,average='micro'))
print('召回率:',recall_score(y_true,y_pred,average='micro'))
print('F1分数:',f1_score(y_true,y_pred,average='micro'))
print('混淆矩阵:',confusion_matrix(y_true,y_pred))
print('分类报告:',classification_report(y_true,y_pred))
print('⭐⭐⭐⭐⭐ 模型评估完成 ⭐⭐⭐⭐⭐')
#7.保存评估结果
dev_data['pred_class'] = y_pred
dev_data.to_csv(config.result_path, index=False)

