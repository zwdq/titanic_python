def datachange(data_load):
    #把男女标签变成0或1
    data_load.loc[data_load['Sex'] == "male", 'Sex'] = 0
    data_load.loc[data_load['Sex'] == "female", 'Sex'] = 1
    #把登船港口标签变成0，1，2
    data_load.loc[data_load['Embarked'] == "S", 'Embarked'] = 0
    data_load.loc[data_load['Embarked'] == "C", 'Embarked'] = 1
    data_load.loc[data_load['Embarked'] == "Q", 'Embarked'] = 2
    #把姓名标签换成姓名长度
    data_load["Name"] = data_load["Name"].map(lambda x:len(x))
    data_load["Ticket"] = data_load["Ticket"].map(lambda x:len(x))
    #特征工程
    del data_load["PassengerId"]
    #del data_load["Name"]
    #del data_load["Ticket"]
    del data_load["Cabin"]
    #del data_load["Embarked"]
    #删掉不要的行
    data_load = data_load.fillna(0)
    return data_load

#获取y标签，同时把data_load从pd换成数组
def datachange2(x):
    y = x["Survived"].values
    del x["Survived"]
    return x.values,y