import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import csv

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

# -- Separar dados em Treino e Teste
def splitData(data, train, val):
    train_df = data[0 : int(len(data)*train)]
    val_df = data[int(len(data)*train)+1 : int(len(data)*train)+int(len(data)*val)]
    test_df = data[int(len(data)*train)+int(len(data)*val)+1 : len(data)]
    return train_df.values, val_df.values, test_df.values

# -- tornar dados em dados supervisionados
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
def createModel(h_neurons, timesteps, features=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    #tf.keras.utils.plot_model(model, 'Traffic_lstm.png', show_shapes=True)
    return model        
# GRU
def createModel2(h_neurons, timesteps,features=1):    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model
# CNN
def createModel3(timesteps=7, filters=16, kernel_size=5, pool_size=2,features=1):
    #timesteps = config[0]
    #timesteps = 7
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

# -- root mean squared error or rmse
def rmse(actual, predicted):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(actual - predicted)))

def grid_generate(neurons, timesteps, batch_size, l_rate):
    configs = []
    for a in neurons:
        for b in timesteps:
            for c in batch_size:
                for d in l_rate:
                    configs.append([a,b,c,d])

    return configs

def grid_search(data, grid, modelStr):    
    # - Divide os dados em subsets, treino, validação e teste -
    train_df, val_df, test_df = splitData(data, 0.7, 0.2)

    scores = []
    for config in grid:
        # unload config
        print('---------------------------')
        print(modelStr)
        print(config)
        neurons, timesteps, batch_size, l_rate = config
        # - To supervised
        X, y  = to_supervised(train_df, timesteps)
        valX, valy  = to_supervised(val_df, timesteps)
        testX, testy  = to_supervised(val_df, timesteps)
        # - create model
        if modelStr == 'lstm':
            model = createModel(neurons, timesteps)
        elif modelStr == 'gru':
            model = createModel2(neurons, timesteps)
        elif modelStr == 'cnn':
            model = createModel3(timesteps=timesteps)
        # - compile n fit
        scores.append(compile_and_fit((X,y), (valX, valy), (testX, testy), model, config))
    # get min rmse
    min = 100
    pos = 0
    for i in range(len(scores)):
        if scores[i][0][2] < min:            
            min = scores[i][0][2]
            pos = i
    return grid[pos]

# -- compilar e treinar
def compile_and_fit(data_train, data_val, data_test, model, config, save=False, str='a'):
    # unload config
    neurons, timesteps, batch_size, l_rate = config
    
    model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate), metrics = ['mae', rmse])
    history = model.fit(
        data_train[0],
        data_train[1], 
        validation_data=data_val,
        epochs=5,
        batch_size=batch_size,
        shuffle=False)
    metrics = model.evaluate(data_test[0], data_test[1])
    if save == True:
        model.save('../models/uni_'+str+'.h5')
        file = open('../models/uni_config_'+str+'.csv', 'w')
        with file:
            write = csv.writer(file) 
            write.writerows([config])

    return [metrics,config]


# -- treinar vários modelos
def massTrain(neurons, timesteps, batch_size, l_rate, data):
    train_df, val_df, test_df = splitData(data, 0.7, 0.2)
    

    models = ['gru', 'cnn']
    print('--------------------')
    print(type(models))
    print('--------------------')
    for m in models:

        # Gerar grid de acordo com o modelo
        if m != 'cnn':
            grid = grid_generate(neurons, timesteps, batch_size, l_rate)
        else:
            grid = grid_generate([1], timesteps, batch_size, l_rate)

        # Config com melhores resultados
        config = grid_search(data, grid, m)
        print(m)
        print(config)

        # - To supervised
        X, y  = to_supervised(train_df, config[1])
        valX, valy  = to_supervised(val_df, config[1])
        testX, testy  = to_supervised(val_df, config[1])
        # treinar modelo com os melhores parametros
        if m == 'lstm':
            model = createModel(config[0], config[1])
        if m == 'gru':
            model = createModel2(config[0], config[1])
        if m == 'cnn':
            model = createModel(config[1])

        compile_and_fit((X,y), (valX, valy), (testX,testy), model, config, save=True, str=m)
        # guardar modelo
        # guardar parametros

# -------------------------------------------------
# ------------------- MAIN ------------------------
# -------------------------------------------------
percTrain = 0.7
neurons = [16,32,64]
timesteps = [12,24]
batch_size = [16,32,64]
l_rate = [0.01,0.001]

# - Carrega dados -
data = loadData()
# - Normalização dos dados -
scaler = dataNorm()

# -- Treino em massa
massTrain(neurons=neurons, timesteps=timesteps, batch_size=batch_size, l_rate=l_rate, data=data)


# - generate grid
#grid = grid_generate(neurons, timesteps, batch_size, l_rate)

# - search and train the grid
#scores = grid_search(data, grid,'teste')
#print(scores)

