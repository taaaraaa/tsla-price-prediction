# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 01:02:45 2023

@author: farha
"""
import sys 
import numpy as np 
from scipy.stats import randint
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

# for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

from datetime import datetime
import math


df = pd.read_csv('tesla_all_data_original_dates.csv', index_col='Date')

data_set = df.loc[:, 'Adj Close'].values

#Split test and train
split=round(len(data_set)*.95)
training_set=data_set[0:split]
test_set=data_set[split:]

#Scale train data
training_set = training_set.reshape(-1, 1)
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# X_train y_train
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Define the LSTM model architecture
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the model on the training set and validate on the validation set
history = regressor.fit(X_train, y_train, epochs = 95, batch_size = 32)

#calculating train_predict
train_predict = regressor.predict(X_train)

#Plot Train Data vs predicted data
plt.rc("figure", figsize=(14,8))
plt.rcParams.update({'font.size': 16})
plt.plot(y_train, label = 'Actual')
plt.plot(train_predict, label = 'Predicted')
plt.xlabel('Time in days')
plt.ylabel('Adjusted Close price')
plt.title('Tesla price prediction using LSTM - Train data')
plt.legend()
plt.show()

#Model Evaluation on Test Data
inputs = data_set[len(data_set) - len(test_set) - 60:]

inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)


X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plot on test data
plt.plot(test_set, color = 'black', label = 'TESLA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TESLA Stock Price')
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()



# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(test_set[:], predicted_stock_price[:,0]))
print('Root Mean Square Error: %.2f RMSE' % (testScore))


# calculate R squared
r2 = r2_score(test_y[:], test_predict[:,0])

print('R-squared:', r2)


