
from sklearn.preprocessing import StandardScaler

class data_utils_method():

    def __init__(self):
        self.Name = "特征工程"
        pass

    def datachange(self,data_load):
        #把男女标签变成0或1
        data_load.loc[data_load['Sex'] == "male", 'Sex'] = 1
        data_load.loc[data_load['Sex'] == "female", 'Sex'] = 0
        #用前面的数据替换剩下数据的缺失值
        data_load = data_load.fillna(data_load["Sex"].mean())
        #特征工程
        del data_load["PassengerId"]
        del data_load["Embarked"]
        del data_load["Name"]
        del data_load["Ticket"]
        del data_load["Cabin"]
        del data_load["Fare"]
        del data_load["Age"]
        del data_load["Pclass"]

        #sc = StandardScaler()
        #data_load = sc.fit_transform(data_load)
        #print(data_load)
        #观看一下特征工程处理后的csv
        #data_load.to_csv("./api_data.csv")
        return data_load

    #获取y标签，同时把data_load从pd换成数组
    def datachange2(self,x):
        y = x["Survived"].values
        del x["Survived"]
        return x.values,y