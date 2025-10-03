import pandas as pd

df = pd.read_csv("outputjp.csv")
df = df.iloc[::-1]
df.reset_index(inplace = True,drop = True)
df.reset_index(inplace = True,drop = True)
df = df.drop(['news'], axis = 1)
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
joblib.dump(model, 'Mljp.pkl')

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
plt.savefig("image2.png")
plt.show()
