from sklearn.impute import SimpleImputer
def datachange(data_load):

    #data_load = data_load.fillna(0)
    data_load = data_load.fillna(0)
    data_load = data_load.dropna()
    return data_load
    '''
    columns = data_load.columns
    imp = SimpleImputer(missing_values = np.nan , strategy = 'constant')
    data_load = imp.fit_transform(data_load)
    data_load = pd.DataFrame(data_load,columns=columns)
    '''

#获取y标签，同时把data_load从pd换成数组
def datachange2(x):
    y = x["Survived"].values
    del x["Survived"]
    return x.values,y

