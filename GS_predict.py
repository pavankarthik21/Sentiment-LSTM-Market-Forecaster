import pandas as pd
import joblib
import json
import numpy as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def predict1():
    df = pd.read_csv("output" + "gs"+".csv")
    df = df.iloc[::-1]
    df.reset_index(inplace = True,drop = True)
    df.reset_index(inplace = True,drop = True)
    df = df.drop(['news'], axis = 1)
    final_test = df.tail(1)
    df = df.drop([len(df)-1])


    new_df = df.filter(['Close'])
    last_60days = new_df[-60:].values
    scaler = MinMaxScaler(feature_range = (0,1))

    last_60days_scaled = scaler.fit_transform(last_60days)
    print(new_df)
    X_test = []
    X_test.append(last_60days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    model = joblib.load('Mlgs.pkl')

    predict_price = model.predict(X_test)
    x=scaler.inverse_transform(predict_price)
    y=np.array(final_test['Close'])
    print("Predicted Price of GS in Nov 3 2023 = ", x)
    print("Original Price of GS in Nov 3 2023 = ", y)

    c = dict(enumerate(x.flatten(), 1))
    d = dict(enumerate(y.flatten(), 2))
    # c= list(x)
    # d=list(y)
    # def Merge(dict1, dict2): 
    #     res = dict1 | dict2
    #     return res 
    
    # def Convert(lst):
    #     res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    #     return res_dct
    c.update(d)
    print(c)
    # print(d)
    # print(type(c),type(d))
    # return(c,d)
    # v= {"1":x,"2":y}
    return json.dumps(str(c))
