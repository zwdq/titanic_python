def datachange(data_load):
    #把男女标签变成0或1
    data_load.loc[data_load['Sex'] == "male", 'Sex'] = 1
    data_load.loc[data_load['Sex'] == "female", 'Sex'] = 0
    #缺失值用平均值替代
    data_load['Age'].fillna(int(data_load['Age'].mean()),inplace=True)
    data_load['Fare'].fillna(int(data_load['Fare'].mean()),inplace=True)
    #用最常见的填充缺失值
    data_load['Embarked'].fillna('S',inplace=True)
    #使用未知填充
    data_load['Cabin'].fillna('U',inplace=True)
    #把登船港口标签变成0，1，2
    data_load.loc[data_load['Embarked'] == "S", 'Embarked'] = 0
    data_load.loc[data_load['Embarked'] == "C", 'Embarked'] = 1
    data_load.loc[data_load['Embarked'] == "Q", 'Embarked'] = 2

    #用前面的数据替换剩下数据的缺失值
    data_load = data_load.fillna(method='ffill')
    data_load = data_load.dropna()
    #把姓名标签换成姓名长度
    data_load["Name"] = data_load["Name"].map(lambda x:len(x))
    
    #特征工程
    del data_load["PassengerId"]
    #del data_load["Name"]
    del data_load["Ticket"]
    del data_load["Cabin"]
    #观看一下特征工程处理后的csv
    data_load.to_csv("./api_data.csv")
    return data_load

#获取y标签，同时把data_load从pd换成数组
def datachange2(x):
    y = x["Survived"].values
    del x["Survived"]
    return x.values,y