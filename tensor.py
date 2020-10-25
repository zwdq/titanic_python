###
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import api

#设置
model_m_name = "./train_model/my_model.h5" #产生的模型名及路径
submission_name = "./submission/tensor__submisson.csv" #输出的预测文件名及路径

def modeltrain(xdata,ydata):
    #切分训练集为训练集和测试集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata,test_size=0.3,random_state=27)
    #再次切分训练集为训练集和验证集用于建模；这样，总共有三组样本：训练集，验证集和测试集
    #主要是因为现在还不会tensorflow的交叉验证
    training_features,validation_features,training_target,validation_target = train_test_split(training_features,training_target)
    #tensorflow2.0的神经网络
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8,input_shape=(8, ), activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    #keras的compile方法，定义损失函数、优化器和指标等
    model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['acc'],
             ) #metrics输出正确率，它是一个列表
    #fit
    history = model.fit(training_features,training_target,validation_data=(validation_features,validation_target),epochs=1000,verbose=2)
    #预测
    #result = model.evaluate(testing_features,testing_target)
    predict_target = (model.predict(testing_features) > 0.5).astype("int32")
    #算准确率
    acc = metrics.accuracy_score(testing_target,predict_target)
    #打印准确率
    print("测试集准确率是:")
    print(acc)
    #print("准确率训练集是:")
    #print(model.evaluate(training_features,training_target))
    #保存模型
    model.save(model_m_name)
    return model

def modelout(model):
    data_load = pd.read_csv("./data_download/test.csv")
    PassengerId = pd.DataFrame(data_load["PassengerId"])
    data_load = api.datachange(data_load).values
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame((model.predict(data_load) > 0.5).astype("int32"))
    Survived.columns = ["Survived"]
    #df横向连接，输出为csv，不要标签
    pd.concat([PassengerId,Survived],axis = 1).to_csv(submission_name,index = 0)
    return
    
    
def main():
    #读取数据
    data_load = pd.read_csv("./data_download/train.csv")
    data_load = api.datachange(data_load)
    xdata,ydata = api.datachange2(data_load)
    model = modeltrain(xdata,ydata)
    modelout(model)
    print("模型已处理完毕")
    return


if __name__ == "__main__":
    main()