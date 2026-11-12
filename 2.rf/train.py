from config import Config
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

config=Config()
train_data_path=config.processed_train_data_path
dev_data_path=config.processed_dev_data_path

data=pd.read_csv(train_data_path)
# print(data.head(10))

# 获取分词列作为特征
words=data['words']
label=data['label']

print("⭐"*5,"数据加载完成","⭐"*5)
#特征工程
# 创建 TF-IDF 模型
tfidf = TfidfVectorizer(
    token_pattern=r'(?u)\b\w+\b'
)
#  使用训练数据训练 TF-IDF 模型并转换文本
features=tfidf.fit_transform(words)
print("⭐"*5,"特征工程完成","⭐"*5)
# print('特征矩阵形状',features.shape)
#创建随机森林模型
rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
for _ in range(10):
    rf.fit(x_train, y_train)
print("⭐"*5,"模型训练完成","⭐"*5)
#模型评估
y_pred = rf.predict(x_test)
#计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')
conf_matrix = confusion_matrix(y_test, y_pred)
print("准确率:", accuracy)
print("精确率:", precision)
print("召回率:", recall)
print("F1分数:", f1)
print("混淆矩阵:", conf_matrix)
print("分类报告:", classification_report(y_test, y_pred))

print("⭐"*5,"模型评估完成","⭐"*5)
#保存模型
with open(config.tfidf_model_save_path,'wb')as f:
    pickle.dump(tfidf,f)
with open(config.rf_model_save_path,'wb')as f:
    pickle.dump(rf,f)
print("⭐"*5,"模型保存完成","⭐"*5)

