

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


data_load = pd.read_csv("./data_download/train.csv")
 

#把男女标签变成0或1
#print(pd.value_counts(data_load["Sex"]))
#print(pd.value_counts(data_load["Embarked"]))
le = LabelEncoder() 
data_load.loc[:,'Sex'] = le.fit_transform(data_load['Sex'])
data_load.loc[:,'Embarked'] = data_load.loc[:,'Embarked'].apply(lambda x: str(x))
data_load.loc[:,'Embarked'] = le.fit_transform(data_load['Embarked'])


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

        
        


