from operator import index
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.python.client import device_lib
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf

# for replicability purposes
tf.random.set_seed(91195003) 
np.random.seed(91190530) 
# for an easy reset backend session state 
tf.keras.backend.clear_session()

# -- Carregar dados processados
def loadData():
    # dtypes for csv fields
    dtypes = {
        'holiday':int,
        'temp':float,
        'clouds_all':int,
        'weather_main':str,
        'date_time':str,
        'traffic_volume':int,
        'weekend':int
        }
    # dates to be parsed from the csv
    parse_dates = ['date_time']

    # read csv
    data = pd.read_csv("../data/metro_processed.csv", dtype=dtypes, parse_dates=parse_dates, index_col=False)
    data['date_time'] = pd.to_datetime(data.date_time, format='%Y-%m-%d %H:%M:%S', errors='raise')

    # drop unwanted columns
    unwanted_cols = ['holiday', 'temp', 'clouds_all', 'weather_main', 'weekend'] 
    data = data.drop(unwanted_cols, axis=1)

    # sort by date
    data = data.sort_values(by=['date_time'])
    data = data.drop('date_time', axis=1)

    return data

# -- Normalizar dados carregados
def dataNorm():
    scaler = MinMaxScaler()
    data[['traffic_volume']] = scaler.fit_transform(data[['traffic_volume']])
    return scaler

# -- Separar dados em Treino, Validação e Teste
def splitData(data, perc):
    train_df = data[0 : int(len(data)*perc)]
    val_df = data[int(len(data)*perc+1) : len(data)]
    return train_df.values, val_df.values


def to_supervised(data, timesteps, out=1):
    df = pd.DataFrame(data)
    cols = list()

    for i in range(timesteps, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, out):
        cols.append(df.shift(-i))

    agg = pd.concat(cols, axis=1)
    agg.dropna(inplace=True)
    df = agg.values
    X, y = df[:, :-1], df[:,-1]
    X = X.reshape((-1,timesteps,1))
    return X, y


# -- Criar modelo
# LSTM
def createModel(h_neurons, timesteps):
    features=1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    #tf.keras.utils.plot_model(model, 'Traffic_lstm.png', show_shapes=True)
    return model

# GRU
def createModel2(h_neurons, timesteps):
    features=1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    #tf.keras.utils.plot_model(model, 'Traffic_lstm.png', show_shapes=True)
    return model

# CNN
def createModel3(config=0, filters=16, kernel_size=5, pool_size=2):
    #timesteps = config[0]
    timesteps = 7
    features=1
    # using the Functional API
    inputs = tf.keras.layers.Input(shape=(timesteps, features)) 
    # microarchitecture
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', data_format='channels_last')(inputs)
    x = tf.keras.layers.AveragePooling1D(pool_size=pool_size, data_format='channels_first')(x)
    # last layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(filters)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    # the model
    cnnModel = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_model') 
    # show model summary (and save it as PNG)
    #tf.keras.utils.plot_model(cnnModel, 'Traffic_snn.png', show_shapes=True) 
    return cnnModel

# root mean squared error or rmse
def rmse(actual, predicted):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(actual - predicted)))

def compile_and_fit(model, epochs, batch_size):
    model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(), metrics = ['mae', rmse])

    hist_list = list()
    loss_list = list()  

    # timeseries cross validator -> ver depois
    # tscv = TimeSeriesSplit(n_splits=cv_splits)



# ------------- MAIN ------------- #
# -- Variavéis --
h_neurons = [64, 128]
timesteps = [3, 5]
learn_rate = [0.1, 0.01]

cv_splits=3

# ---- Execução ----

# - Carrega dados -
data = loadData()
# - Normalização dos dados -
scaler = dataNorm()
# - Divide os dados em subsets, treino, validação e teste-
train_df, val_df = splitData(data, 0.7)
# - To supervised
X, y  = to_supervised(train_df, 7)
valX, valy = to_supervised(val_df, 7)



# - Criar modelo
#model = createModel(128, 5)
model = createModel3()
model.compile(loss='mse', optimizer='adam')
# fit model
model.fit(X, y, epochs=5, batch_size=24)

predictions = model.evaluate(valX)
print(predictions)
'''
val = scaler.inverse_transform(valy.reshape(-1,1))
predictions = scaler.inverse_transform(predictions)

for x in range(10):
    print("---- {} ----".format(x))
    print(predictions[x])
    print(val[x])

'''
