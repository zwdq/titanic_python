###
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import data_utils


#设置
model_m_name = "./train_model/tensor_model.h5" #产生的模型名及路径
submission_name = "./submission/tensor_submisson.csv" #输出的预测文件名及路径

def modeltrain(xdata,ydata):
   
    #切分训练集为训练集和测试集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata,test_size=0.3,random_state=27)
    #再次切分训练集为训练集和验证集用于建模；这样，总共有三组样本：训练集，验证集和测试集
    #主要是因为现在还不会tensorflow的交叉验证
    training_features,validation_features,training_target,validation_target = train_test_split(training_features,training_target)
    #tensorflow2.0的神经网络
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8,input_shape=(7, ), activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
        #tf.keras.layers.Dropout(0.5),
    ])
    #keras的compile方法，定义损失函数、优化器和指标等
    model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=[tf.keras.metrics.AUC()],
             ) #metrics输出正确率，它是一个列表
    #fit
    model.fit(training_features,training_target,validation_data=(validation_features,validation_target),epochs=100,verbose=2)
    predict_target = (model.predict(testing_features) > 0.5).astype("int32")
    #算准确率
    acc = metrics.accuracy_score(testing_target,predict_target)
    #打印准确率
    print("*************************")
    print(r"the acc of test_data is :")
    print(acc)
    #保存模型
    model.save(model_m_name)
    return model

def modelout(model):
    data_load = pd.read_csv("./data_download/test.csv")
    PassengerId = pd.DataFrame(data_load["PassengerId"])
    xdata,ydata = api.datachange(data_load)
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame((model.predict(xdata) > 0.5).astype("int32"))
    Survived.columns = ["Survived"]
    #df横向连接，输出为csv，不要标签
    pd.concat([PassengerId,Survived],axis = 1).to_csv(submission_name,index = 0)
    return
    
    
def main():
    #读取数据
    data_load = pd.read_csv("./data_download/train.csv")
    global api
    api = data_utils.data_utils_method()
    xdata,ydata = api.datachange(data_load)
    model = modeltrain(xdata,ydata)
    modelout(model)
    print(r"Model Finished")
    print("*************************")
    return


if __name__ == "__main__":
    main()