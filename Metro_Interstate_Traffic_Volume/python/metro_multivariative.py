import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import csv
from keras.preprocessing.sequence import TimeseriesGenerator

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
    unwanted_cols = ['weather_main'] 
    data = data.drop(unwanted_cols, axis=1)

    # sort by date
    data = data.sort_values(by=['date_time'])
    data = data.drop('date_time', axis=1)

    return data

# -- Normalizar dados carregados
def dataNorm(data):
    scalerTemp = MinMaxScaler()
    data[['temp']] = scalerTemp.fit_transform(data[['temp']])

    scalerClouds = MinMaxScaler()
    data[['clouds_all']] = scalerClouds.fit_transform(data[['clouds_all']])

    scalerTraffic = MinMaxScaler()
    data[['traffic_volume']] = scalerTraffic.fit_transform(data[['traffic_volume']])    

    return [scalerTemp,scalerClouds,scalerTraffic]

# -- Desnormalizar dados carregados
def dataDesnorm(data):
    return data

# -- Separar dados em Treino e Teste
def splitData(data, train, val):
    train_df = data[0 : int(len(data)*train)]
    val_df = data[int(len(data)*train)+1 : int(len(data)*train)+int(len(data)*val)]
    test_df = data[int(len(data)*train)+int(len(data)*val)+1 : len(data)]
    return train_df.values, val_df.values, test_df.values

# LSTM
def createModel(h_neurons, timesteps, features=5):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    #tf.keras.utils.plot_model(model, 'Traffic_lstm.png', show_shapes=True)
    return model     
# GRU
def createModel2(h_neurons, timesteps,features=5):    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model
# CNN
def createModel3(timesteps=7, filters=16, kernel_size=5, pool_size=2, features=5):
    #timesteps = config[0]
    timesteps = 7
    
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
# -- root mean squared error or rmse
def rmse(actual, predicted):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(actual - predicted)))

# -- return grid to search
def grid_generate(neurons, timesteps, batch_size, l_rate):
    configs = []
    for a in neurons:
        for b in timesteps:
            for c in batch_size:
                for d in l_rate:
                    configs.append([a,b,c,d])

    return configs

# -- compilar e treinar
def compile_and_fit(data_train, data_val, data_test, model, config):
    # unload config
    neurons, timesteps, batch_size, l_rate = config
    
    model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate), metrics = ['mae', rmse])
    history = model.fit(
        data_train, 
        validation_data=data_val,
        epochs=2,
        batch_size=batch_size,
        shuffle=False)
    metrics = model.evaluate(data_test)
    return [metrics,config]

# -- Grid search
def grid_search(data, grid, str=""):    
    # - Divide os dados em subsets, treino, validação e teste -
    train_df, val_df, test_df = splitData(data, 0.7, 0.2)

    scores = []
    for config in grid:
        # unload config
        neurons, timesteps, batch_size, l_rate = config
        # - To supervised
        #X, y  = to_supervised(train_df, timesteps)
        #valX, valy  = to_supervised(val_df, timesteps)
        #testX, testy  = to_supervised(val_df, timesteps)
        # TimeSeriesGenerator
        data1 = train_df
        targets = data1[:,3]
        dataTrain = TimeseriesGenerator(data1, targets,
                               length=timesteps,
                               batch_size=batch_size)
        data1 = val_df
        targets = data1[:,3]
        dataVal = TimeseriesGenerator(data1, targets,
                               length=timesteps,
                               batch_size=batch_size)
        data1 = test_df
        targets = data1[:,3]
        dataTest = TimeseriesGenerator(data1, targets,
                               length=timesteps,
                               batch_size=batch_size)
        # - create model
        model = createModel(neurons, timesteps)
        # - compile n fit
        scores.append(compile_and_fit(dataTrain, dataVal, dataTest, model, config))
        print(scores[-1])
    min = 100
    pos = 0
    # get min rmse
    for i in range(len(scores)):
        if scores[i][0][2] < min:
            min = scores[i][0][2]
            pos = i
    return grid[i]
# -------------------------------------------------
# ------------------- MAIN ------------------------
# -------------------------------------------------
percTrain = 0.7
neurons = [16]
timesteps = [6]
batch_size = [16,32]
l_rate = [0.1,]

# - Carrega dados -
data = loadData()
# - Normalização dos dados -
scaler = dataNorm(data)

#print(data.head())

# - generate grid
grid = grid_generate(neurons, timesteps, batch_size, l_rate)

# - search and train the grid
scores = grid_search(data, grid,'teste')
print(scores)