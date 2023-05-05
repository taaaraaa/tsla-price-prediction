import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math

# ML libraries
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping  # To stop training early if val loss stops decreasing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.stattools import pacf

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import warnings
warnings.filterwarnings('ignore')

# import the table
tesla=pd.read_csv('interpolated_tesla_df.csv',index_col='Date')
#this table only has one column and has date as index

data=tesla.values

# Separate train and test data
train_length = int(len(data) * 0.9)

train_data, test_data = data[:train_length], data[train_length:]
print('Shape of Train and Test data: ', train_data.shape, test_data.shape)

# Reshape the data
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)
print('Shape of Train and Test data: ', train_data.shape, test_data.shape)


# Calculating pacf value

pacf_value = pacf(data, nlags=10)
lag = 0
# collect lag values greater than 10% correlation 
for x in pacf_value:
    if x > 0.1:
        lag += 1
    else:
        break
print('Selected look_back (or lag = ): ', lag)


# 1. Multi-Layer Perceptron
print('-----------Model 1------------')

# define a function to split a univariate sequence into supervised learning [Input and Output]
def create_dataset(dataset, lookback):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback -1):
        a = dataset[i: (i+lookback), 0]
        dataX.append(a)
        b = dataset[i+lookback, 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train_data, lag)
test_X, test_y = create_dataset(test_data, lag)

print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)


# Fix random seed for reproducibility
np.random.seed(7)

#Model 1 Arcitecture
model1 = Sequential()
model1.add(Dense(64, input_dim = lag, activation='relu', name= "1st_hidden"))

model1.add(Dense(1, name = 'Output_layer', activation='linear'))
model1.compile(loss="mean_squared_error", optimizer="adam")
#model1.summary()

# Fitting the model/callbacks prevents the code from running once not improving results after 3 epoches 
history1 = model1.fit(train_X, train_y, epochs = 200, batch_size = 64, callbacks=[EarlyStopping(patience=3)], verbose = 1, shuffle=False, 
                    validation_split=0.1)

# plot history
plt.clf
plt.figure(figsize=(10,8))
plt.plot(history1.history['loss'][2:], label='Train')
plt.plot(history1.history['val_loss'][2:], label='Validation')
plt.xlabel('Number of Epochs')
plt.ylabel('Train and Test Loss')
plt.title('Train and Validation loss per epoch (MLP)',fontsize=22)
plt.savefig('./data/Train and Validation loss MLP.png')

plt.legend()
plt.show()

# Make prediction
testPredict = model1.predict(test_X)

# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(test_y[:], testPredict[:,0]))
print(' Root Mean Squared Error: %.2f RMSE' % (testScore))

# R-squared
r2 = r2_score(test_y[:], testPredict[:,0])
print('R-squared:', r2)


#Model 2 (RNN)
print('--------Model 2----------')

# split a univariate sequence into supervised learning [Input and Output]

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


lag = 2  # Empirically we have choosen
n_features = 1


train_X2, train_y2 = split_sequence(train_data, lag)
test_X2, test_y2 = split_sequence(test_data, lag)

#6.5 Reshape train_X and test_X to 3-Dimension

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))

# define model
model2 = Sequential()
model2.add(SimpleRNN(64, activation='relu', return_sequences=False, input_shape=(lag, n_features)))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mse')

# fit model
history2 = model2.fit(train_X, train_y, epochs = 200, callbacks=[EarlyStopping(patience=3)], batch_size=64, verbose=1, validation_split= 0.1)

# plot history2
plt.plot(history2.history['loss'][2:])
plt.plot(history2.history['val_loss'][1:])
plt.title('Train and Validation loss per epochs (RNN)',fontsize=24)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'])

plt.savefig('./data/Train and Validation loss RNN.png')
plt.show()


train_predict = model2.predict(train_X)
test_predict = model2.predict(test_X)

print('Shape of train and test predict: ', train_predict.shape, test_predict.shape)

# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(test_y[:], test_predict[:,0]))
print(' Root Mean Squared Error: %.2f RMSE' % (testScore))


# calculate R squared
r2 = r2_score(test_y[:], test_predict[:,0])
print('R-squared:', r2)

# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(test_y[:], test_predict[:,0]))
print(' Root Mean Squared Error: %.2f RMSE' % (testScore))