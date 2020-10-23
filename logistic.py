import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
#导入自己的api.py,里面共有两个方法api.datachange和api.datachange2，用于特征工程
import api

#设置
model_m_name = "./train_model/logistc_train_model.m" #产生的模型名及路径
submission_name = "./submission/logistic_submisson.csv"


def modeltrain(xdata,ydata):
    #调用sklearn逻辑回归api
    model = LogisticRegression(max_iter=500000)
    #切分训练集
    features_train,features_test,predict_train,predict_test = train_test_split(xdata,ydata,test_size=0.3)
    #fit
    model = model.fit(features_train,predict_train)
    #预测
    y_predict = model.predict(features_test)
    #算准确率
    acc = metrics.accuracy_score(predict_test,y_predict)
    #打印准确率
    print("准确率是:")
    print(acc)
    #保存模型
    joblib.dump(model, model_m_name)
    return model

def modelout(model):
    data_load = pd.read_csv("./data_download/test.csv")
    PassengerId = pd.DataFrame(data_load["PassengerId"])
    data_load = api.datachange(data_load).values
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame(model.predict(data_load))
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