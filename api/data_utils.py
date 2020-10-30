from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

class data_utils_method():

    def __init__(self):
        self.Name = "特征工程"
        pass

    def datachange(self,data_load):

        #如果包含Survived则输入的是训练集，取y切掉标签；不包含，则输入的是测试集，不用管y,直接特征工程
        if "Survived" in data_load.columns:
            ydata = data_load["Survived"]
            del data_load["Survived"]
        else:
            ydata = []

        #把男女标签变成0或1
        data_load.loc[data_load['Sex'] == "male", 'Sex'] = 1
        data_load.loc[data_load['Sex'] == "female", 'Sex'] = 0
        
        #把登船港口标签变成0，1，2
        data_load.loc[data_load['Embarked'] == "S", 'Embarked'] = 0
        data_load.loc[data_load['Embarked'] == "C", 'Embarked'] = 1
        data_load.loc[data_load['Embarked'] == "Q", 'Embarked'] = 2

        #把姓名标签换成姓名长度
        #data_load["Name"] = data_load["Name"].map(lambda x:len(x))
    
        #特征工程
        del data_load["PassengerId"]
        del data_load["Name"]
        del data_load["Ticket"]
        del data_load["Cabin"]
        
        #均值填充
        si = SimpleImputer(missing_values=np.nan,strategy="mean")
        xdata = si.fit_transform(data_load)
        #标准归一化
        sc = StandardScaler()
        xdata = sc.fit_transform(xdata)
        
        
        

        return xdata,ydata

