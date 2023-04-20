
data = tesla['Adj Close'].values
data.shape
(2719,)
# Separate train and test data
train_length = int(len(data) * 0.8)
print('Train length: ', train_length)
​
train_data, test_data = data[:train_length], data[train_length:]
print('Shape of Train and Test data: ', train_data.shape, test_data.shape)
Train
length: 2175
Shape
of
Train and Test
data: (2175,)(544, )
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)
print('Shape of Train and Test data: ', train_data.shape, test_data.shape)
Shape
of
Train and Test
data: (2175, 1)(544, 1)


# split a univariate sequence into supervised learning [Input and Output]
def create_dataset(dataset, lookback):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback - 1):
        a = dataset[i: (i + lookback), 0]
        dataX.append(a)
        b = dataset[i + lookback, 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


plot_pacf(data, lags=10)
plt.show()

from statsmodels.tsa.stattools import pacf
​
pacf_value = pacf(data, nlags=10)
​
lag = 0
# collect lag values greater than 10% correlation
for x in pacf_value:
    if x > 0.1:
        lag += 1
    else:
        break
print('Selected look_back (or lag = ): ', lag)
Selected
look_back( or lag = ):  2
train_X, train_y = create_dataset(train_data, lag)
test_X, test_y = create_dataset(test_data, lag)
​
print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)
Shape
of
train_X and train_y: (2172, 2)(2172, )
Shape
of
test_X and test_y: (541, 2)(541, )
test
print(test_data[:5])  # original data
for x in range(len(train_X[:5])):
    print(test_X[x], test_y[x], )  # trainX and trainY after lookback
[[20.17066765]
 [19.41533279]
 [19.64733315]
 [19.91799927]
 [19.85733223]]
[20.17066765 19.41533279]
19.6473331451416
[19.41533279 19.64733315]
19.917999267578125
[19.64733315 19.91799927]
19.857332229614258
[19.91799927 19.85733223]
20.982667922973633
[19.85733223 20.98266792]
21.325332641601562
# Fix random seed for reproducibility
# Thes seed value helps in initilizing random weights and biases to the neural network.
np.random.seed(7)
# ML libraries
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import keras

model = Sequential()
model.add(Dense(64, input_dim=lag, activation='relu', name="1st_hidden"))
# model.add(Dense(64, activation='relu', name = '2nd_hidden'))
model.add(Dense(1, name='Output_layer', activation='linear'))
# model.add(Activation("linear", name = 'Linear_activation'))
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()
Model: "sequential"
_________________________________________________________________
Layer(type)
Output
Shape
Param  #
== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
1
st_hidden(Dense)(None, 64)
192

Output_layer(Dense)(None, 1)
65

== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
Total
params: 257
Trainable
params: 257
Non - trainable
params: 0
_________________________________________________________________
100
epoch_number = 100
batches = 64
​
history = model.fit(train_X, train_y, epochs=epoch_number, batch_size=batches, verbose=1, shuffle=False,
                    validation_split=0.1)

ms / step - loss: 0.1335 - val_loss: 0.6266
# plot history
plt.clf
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Number of Epochs')
plt.ylabel('Train and Test Loss')
plt.title('Train and Test loss per epochs [Univariate]')
plt.legend()
plt.show()

# Make prediction
testPredict = model.predict(test_X)
17 / 17[ == == == == == == == == == == == == == == ==] - 0
s
833u
s / step
5
testPredict[:5]
​
array([[19.702312],
       [19.574635],
       [19.830656],
       [19.890656],
       [20.582138]], dtype=float32)
# calculate root mean squared error
# RMSE between actual and predicted cpu values
import math
from sklearn.metrics import mean_squared_error
​
testScore = math.sqrt(mean_squared_error(test_y[:], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
Test
Score: 5.55
RMSE
# Here we're plotting Test and Predicted data
​
plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 12})
plt.plot(test_y[:], '#0077be', label='Actual')
plt.plot(testPredict[:, 0], '#ff8841', label='Predicted')
plt.title('MLP Model for Tesla Stock Forecasting')
plt.ylabel('Tesla Stock Price [in Dollar]')
plt.xlabel('Time Steps [in Days] ')
plt.legend()
plt.show()

RNN - Recurrent
Neural
Network
tesla
data = tesla['Adj Close'].values
print('Shape of data: ', data.shape)
Shape
of
data: (2719,)
# Separate train and test data
train_length = int(len(data) * 0.8)
print('Train length: ', train_length)
​
train_data, test_data = data[:train_length], data[train_length:]
print('Shape of Train and Test data: ', len(train_data), len(test_data))
Train
length: 2175
Shape
of
Train and Test
data: 2175
544
# split a univariate sequence into supervised learning [Input and Output]
from numpy import array


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


lag = 2  # Empirically we have choosen
n_features = 1
train_X, train_y = split_sequence(train_data, lag)
test_X, test_y = split_sequence(test_data, lag)
print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)
Shape
of
train_X and train_y: (2173, 2)(2173, )
Shape
of
test_X and test_y: (542, 2)(542, )
# 6.5 Reshape train_X and test_X to 3-Dimension¶
​
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
# New shape of train_X and test_X are :-
print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)
Shape
of
train_X and train_y: (2173, 2, 1)(2173, )
Shape
of
test_X and test_y: (542, 2, 1)(542, )
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN
from keras.layers import Dense

# define model
model = Sequential()
model.add(SimpleRNN(64, activation='relu', return_sequences=False, input_shape=(lag, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
Model: "sequential_1"
_________________________________________________________________
Layer(type)
Output
Shape
Param  #
== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
simple_rnn(SimpleRNN)(None, 64)
4224

dense(Dense)(None, 1)
65

== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
Total
params: 4, 289
Trainable
params: 4, 289
Non - trainable
params: 0
_________________________________________________________________
# As you are trying to use function decorator in TF 2.0,
# please enable run function eagerly by using below line after importing TensorFlow:
import tensorflow as tf

tf.config.run_functions_eagerly(True)
# fit model
​
history = model.fit(train_X, train_y, epochs=50, batch_size=64, verbose=1, validation_split=0.1)
Epoch
1 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 1
s
15
ms / step - loss: 31.8533 - val_loss: 5.8629
Epoch
2 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.9032 - val_loss: 1.1280
Epoch
3 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.2228 - val_loss: 0.8398
Epoch
4 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1932 - val_loss: 0.8395
Epoch
5 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1945 - val_loss: 0.8431
Epoch
6 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1920 - val_loss: 0.8391
Epoch
7 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1920 - val_loss: 0.8395
Epoch
8 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1905 - val_loss: 0.8385
Epoch
9 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1903 - val_loss: 0.8364
Epoch
10 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1894 - val_loss: 0.8365
Epoch
11 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1889 - val_loss: 0.8353
Epoch
12 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1887 - val_loss: 0.8360
Epoch
13 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1888 - val_loss: 0.8375
Epoch
14 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1909 - val_loss: 0.8340
Epoch
15 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
15
ms / step - loss: 0.1861 - val_loss: 0.8201
Epoch
16 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1831 - val_loss: 0.8187
Epoch
17 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1823 - val_loss: 0.8164
Epoch
18 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1827 - val_loss: 0.8167
Epoch
19 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1828 - val_loss: 0.8140
Epoch
20 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1820 - val_loss: 0.8129
Epoch
21 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1816 - val_loss: 0.8107
Epoch
22 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
15
ms / step - loss: 0.1810 - val_loss: 0.8102
Epoch
23 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
16
ms / step - loss: 0.1804 - val_loss: 0.8123
Epoch
24 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1818 - val_loss: 0.8071
Epoch
25 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1804 - val_loss: 0.8057
Epoch
26 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1798 - val_loss: 0.8043
Epoch
27 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1811 - val_loss: 0.8040
Epoch
28 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1800 - val_loss: 0.8017
Epoch
29 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1794 - val_loss: 0.8015
Epoch
30 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1800 - val_loss: 0.8172
Epoch
31 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1790 - val_loss: 0.7986
Epoch
32 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1780 - val_loss: 0.8047
Epoch
33 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1774 - val_loss: 0.7952
Epoch
34 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
15
ms / step - loss: 0.1771 - val_loss: 0.7936
Epoch
35 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1803 - val_loss: 0.7946
Epoch
36 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1797 - val_loss: 0.7902
Epoch
37 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1788 - val_loss: 0.8469
Epoch
38 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1828 - val_loss: 0.7954
Epoch
39 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
15
ms / step - loss: 0.1757 - val_loss: 0.7849
Epoch
40 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1757 - val_loss: 0.7833
Epoch
41 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1752 - val_loss: 0.7879
Epoch
42 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1748 - val_loss: 0.7821
Epoch
43 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1734 - val_loss: 0.7783
Epoch
44 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1738 - val_loss: 0.7765
Epoch
45 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1741 - val_loss: 0.7750
Epoch
46 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1726 - val_loss: 0.7731
Epoch
47 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
14
ms / step - loss: 0.1744 - val_loss: 0.7988
Epoch
48 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1791 - val_loss: 0.7723
Epoch
49 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
12
ms / step - loss: 0.1705 - val_loss: 0.7678
Epoch
50 / 50
31 / 31[ == == == == == == == == == == == == == == ==] - 0
s
13
ms / step - loss: 0.1703 - val_loss: 0.7681
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

train_predict = model.predict(train_X)
test_predict = model.predict(test_X)
​
print('Shape of train and test predict: ', train_predict.shape, test_predict.shape)
68 / 68[ == == == == == == == == == == == == == == ==] - 0
s
4
ms / step
17 / 17[ == == == == == == == == == == == == == == ==] - 0
s
5
ms / step
Shape
of
train and test
predict: (2173, 1)(542, 1)
# root mean squared error or rmse
import math
from sklearn.metrics import mean_squared_error
​

def measure_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

​
train_score = measure_rmse(train_y, train_predict)
test_score = measure_rmse(test_y, test_predict)
​
print('Train and Test RMSE: ', train_score, test_score)
Train and Test
RMSE: 0.480246527292266
6.232530536785542
plt.rc("figure", figsize=(14, 8))
plt.rcParams.update({'font.size': 16})
plt.plot(test_y, label='Actual')
plt.plot(test_predict, label='Predicted')
plt.xlabel('Time in days')
plt.ylabel('Adjusted Close price')
plt.title('Tesla price prediction using Simple RNN - Test data')
plt.legend()
plt.show()

