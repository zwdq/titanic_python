def datachange(data_load):
    data_load = data_load.dropna(axis=0)
    return data_load

#获取y标签，同时把data_load从pd换成数组
def datachange2(x):
    y = x["Survived"].values
    del x["Survived"]
    return x.values,y