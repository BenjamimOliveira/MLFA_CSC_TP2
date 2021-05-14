from operator import index
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import shuffle
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
def createModel(h_neurons, timesteps):
    features=1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    #tf.keras.utils.plot_model(model, 'Traffic_lstm.png', show_shapes=True)
    return model        

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

def grid_search(data, grid):    
    # - Divide os dados em subsets, treino, validação e teste -
    train_df, val_df, test_df = splitData(data, 0.7, 0.2)

    scores = []
    for config in grid:
        # unload config
        neurons, timesteps, batch_size, l_rate = config
        # - To supervised
        X, y  = to_supervised(train_df, timesteps)
        valX, valy  = to_supervised(val_df, timesteps)
        testX, testy  = to_supervised(val_df, timesteps)
        # - create model
        model = createModel(neurons, timesteps)
        # - compile n fit
        scores.append(compile_and_fit((X,y), (valX, valy), (testX, testy), model, config))
    return scores

# -- compilar e treinar
def compile_and_fit(data_train, data_val, data_test, model, config):
    # unload config
    neurons, timesteps, batch_size, l_rate = config
    
    model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate), metrics = ['mae', rmse])
    history = model.fit(
        data_train[0],
        data_train[1], 
        validation_data=data_val,
        epochs=2,
        batch_size=batch_size,
        shuffle=False)
    metrics = model.evaluate(data_test[0], data_test[1])
    return [metrics,config]





# -------------------------------------------------
# ------------------- MAIN ------------------------
# -------------------------------------------------
percTrain = 0.7
neurons = [32]
timesteps = [3]
batch_size = [16,32]
l_rate = [0.01]

# - Carrega dados -
data = loadData()
# - Normalização dos dados -
scaler = dataNorm()

# - generate grid
grid = grid_generate(neurons, timesteps, batch_size, l_rate)

# - search and train the grid
scores = grid_search(data, grid)
print(scores)

