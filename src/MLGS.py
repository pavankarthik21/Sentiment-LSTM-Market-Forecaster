import pandas as pd

df = pd.read_csv("output" + "gs"+".csv")
df = df.iloc[::-1]
df.reset_index(inplace = True,drop = True)
df = df.drop(['news'], axis = 1)
df.set_index('Date')
final_test = df.tail(1)
df = df.drop([len(df)-1])
print(df)

#libraries

import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import joblib
plt.style.use('fivethirtyeight')

print(df.shape)

data = df.filter(['Close'])
dataset = data.values  #numpy array
train_len = math.ceil(len(dataset)*0.8)

print(train_len)

print(dataset) 
print("----------------------------------------")
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)     



train_data = scaled_data[0:train_len, :]
#split 
x_train = []
y_train = []

for i in range(60,len(train_data)) :
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    
x_train,y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))




print(x_train.shape)

#Build LSTM

model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam',loss = 'mean_squared_error')

#train the model
model.fit(x_train,y_train,batch_size=1,epochs= 1)


test_data = scaled_data[train_len-60:,:]

# saving our ML model
joblib.dump(model, 'Mlgs.pkl')

x_test = []
y_test = dataset[train_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#plot the data

train = data[:train_len]
valid = data[train_len:]
valid['Predictions'] = predictions

#Plotting graph
plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date',fontsize = 16)
plt.ylabel('close price USD ($)', fontsize =16)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc = 'lower right')
plt.savefig("image1.png")
plt.show()

# #Plotting graph 2
# from datetime import datetime
# plt.figure(figsize = (16,8))
# plt.title('Model')
# plt.xlabel('Date',fontsize = 16)
# plt.ylabel('close price USD ($)', fontsize =16)

# D1 = pd.to_datetime(df['Date'])
# X = ['11-03-2023']
# df_x = pd.DataFrame(X)
# print(df['Date'])


# last_60days = df[-60:].values
# scaler = MinMaxScaler(feature_range = (0,1))

# last_60days_scaled = scaler.fit_transform(last_60days)
# print(df)
# X_test = []
# X_test.append(last_60days_scaled)
# X_test = np.array(X_test)
# X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
# predict_price = model.predict(X_test)
# x=scaler.inverse_transform(predict_price)
# lst = []
# lst.append(x)
# y = pd.DataFrame(lst)

# ax = df_x.plot()
# y.plot(ax=ax)



# to estimate stock price on 3rd Nov,2023
# We use previous sixty days data from our dataset and perform test



# def predict():

#     new_df = df.filter(['Close'])
#     last_60days = new_df[-60:].values
#     last_60days_scaled = scaler.transform(last_60days)
#     print(new_df)
#     X_test = []
#     X_test.append(last_60days_scaled)
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#     model = joblib.load('Mlgs.pkl')

#     predict_price = model.predict(X_test)
    # print("Predicted Price of Nov 3 2023 = ", scaler.inverse_transform(predict_price))
    # print("Original Price of Nov 3 2023 = ", np.array(final_test['Close']))

#     return {1:scaler.inverse_transform(predict_price),2:np.array(final_test['Close'])}







