import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
import api

#设置
model_m_name = "./train_model/xgb_train_model.m" #产生的模型名及路径
submission_name = "./submission/xgb_submisson.csv" #输出的预测文件名及路径


def modeltrain(xdata,ydata):
    #调用sklearn逻辑回归api
    model = XGBClassifier(
        learning_rate=0.01,      # 学习速率
        reg_alpha=1,            # l1正则权重
        n_estimators=1000,      # 树的个数 --n棵树建立xgboost
        max_depth=5,            # 树的深度
        subsample=0.9,          # 随机选择x%样本建立决策树，小了欠拟合，大了过拟合
        colsample_bytree = 0.9, # x%特征建立决策树
        scale_pos_weight=1,     # 解决样本个数不平衡的问题
        gamma=0.1, 
        random_state=20,        # 随机数
        objective='binary:logistic',  # 损失函数 objective='multi:softmax' 'binary:logistic' reg:linear
        )
    #切分训练集
    X_train,X_test,y_train,y_test = train_test_split(xdata,ydata,test_size=0.5)
    #fit
    model = model.fit(X_train,y_train)
    #预测
    y_predict = model.predict(X_test)
    #算准确率
    acc = metrics.accuracy_score(y_test,y_predict)
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