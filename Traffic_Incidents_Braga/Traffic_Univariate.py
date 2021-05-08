# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from numpy import nan
tf.random.set_seed(12345)

tf.keras.backend.clear_session()

path = 'Traffic_Braga.csv'
                #============================================================
                #                       Load DataSet                 
                #============================================================
#Carrega o Dataset
def load_data(path):
    raw_data = pd.read_csv(path,infer_datetime_format=True)
    return raw_data 



                #============================================================
                # 
                #                    Data Manipulation 
                #                                      
                #============================================================

#Divide o Dataset em validacao e treino
def split_data(training, perc=10):
    train_raw = np.arange(0, int(len(training)*(100-perc)/100)) #Splita de 0 ate o total menos o percentual neste caso 0 a 90
    validation_raw = np.arange(int(len(training)*(100-perc)/100+1), len(training)) #Splita  do total ate o fim, neste caso 90 ate 100
    return train_raw, validation_raw 

#Dropo todas as colunas
def data_Uni_Raw(df):
    df_1 = df.drop(columns=['cause_of_incident','city_name','description','cause_of_incident', 'from_road', 'to_road', 'affected_roads',
                           'incident_category_desc','magnitude_of_delay_desc', 'length_in_meters','delay_in_seconds',
                            'latitude','longitude'])
    return df_1
#Prepara os Dados Para incidentes Diarios, somatório de Incidentes por dia, com a data como index
def Data_Uni_Daily(df):
    df_1 = data_Uni_Raw(df)
    df_1['incident_date'] = df_1['incident_date'].str[:10]#Delete last 10 str
    df_1['Incidents'] = pd.DataFrame([1 for x in range(len(df_1.index))]) #Create a column with 1 to sum de incidents eh day
    df_1 = df_1.set_index('incident_date') #Set column incident_date to index
    df_1.index = pd.to_datetime(df_1.index) #Set Date string in index to Date type
    daily_groups = df_1.resample('D') #To sum groupy by each day
    daily_data = daily_groups.sum()
    return daily_data

    #Ver
    # 1 - Preencher dados em falta ao invés de dropa-los
    # 2 - Por horas
    # 3 - semanalmente
    # 4 - Considerar outro atributo para Univariate ?
    # 5 - Considerar quais atributos para Multivariate ?

                #============================================================
                # 
                #                    Data Preparation 
                #                                      
                #===========================================================

#Normaliza os Dados entre 1 e -1   
def data_normalization (df, norm_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    df[['Incidents']] = scaler.fit_transform(df[['Incidents']])
    return scaler

#Preparing Data for Build Model
#Transforma os dados em supervisionados
def to_supervised(df, timesteps):
    data = df.values
    x, y = list (), list ()
    dataset_size = len(data)
    for curr_pos in range(dataset_size):
        input_index = curr_pos + timesteps
        label_index = input_index +1
        if label_index < dataset_size:
            x.append(data[curr_pos:input_index,:])
            y.append(data[curr_pos:label_index,0])
    return np.array(x).astype('float32'), np.array(y).astype('float32')

#Prepare x,y for train,test, and validation
def prepare_train(df,config):
    timesteps, h_neurons, epochs, batch_size = config
    X, y = to_supervised(df, timesteps)
     #Time Series Cross Validator
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for train_index, test_index in tscv.split(X):
        train_idx, val_idx = split_data(train_index, perc=10) #further split into training and validation sets
        #build data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_index], y[test_index]
        #model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(), metrics = ['mae', rmse])
    return X_train,y_train,X_val, y_val,X_test, y_test
 


                #============================================================
                # 
                #                    Build Models, and Fit 
                #                                      
                #===========================================================   
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

#Modelo simples LSTM
def build_model(config):
    timesteps, h_neurons, epochs, batch_size = config
    features=1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    #model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_model.png', show_shapes=True)
    model.compile(loss = rmse, optimizer = 'Adam', metrics = ['mae', rmse])
    return model

#Modelo simples GRU
def build_model_2(config):
    timesteps, h_neurons, epochs, batch_size = config
    features=1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    #model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_model.png', show_shapes=True)
    model.compile(loss = rmse, optimizer = 'Adam', metrics = ['mae', rmse])
    return model

def compile_and_fit(df,model, config):
    timesteps, h_neurons, epochs, batch_size = config
    univariate = 1
    X_train,y_train,X_val, y_val,X_test, y_test = prepare_train(df, config)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
    epochs=epochs, batch_size=batch_size, shuffle=False) 
    metrics = model.evaluate(X_test, y_test)
    hist = history
    loss = metrics[2]
    plot_learning_curves(history.history['loss'], history.history['val_loss']) 
    return  hist, loss

#Recebe uma config por vez e passa para o model que por sua vez é passado para o compile_and_fit para o treino e verificação da performance
def call_models(df, config, n_repeats=1):
    timesteps, h_neurons, epochs, batch_size = config
    univariate = 1 #Quando fazer modelo multivariate mudar
    key = str(config)
    to_supervised(df, timesteps)
    #model = build_model(config) #Passar aqui as diferentes arquiteturas de models criadas
    model = build_model_2(config)
    hist, loss = compile_and_fit(df,model, config)
    print(f' Model: {key} Loss: {loss:.4f}')
    return (key, loss)

# Envia uma configuracao por vez para ser executada e recebe a configuracao com sua loss
#As configuracoes e loss são ordenadas em uma lista ex Configuracao: [3, 64, 40, 7]Loss: 0.386593371629
def grid_search(df,cfg_list):
    # evaluate configs
    scores = scores = [call_models(df, cfg) for cfg in cfg_list]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

#Executa todas possibilidades possiveis de acordo com os parametros passados (timestep,h_neurons,epochs,epochs)
def model_configs(timestep, h_neurons, epochs, batch_size):
    # define scope of configs
    timestep = timestep
    h_neurons = h_neurons
    epochs = epochs
    batch_size = batch_size
    # create configs
    configs = list()
    for i in timestep:
        for j in h_neurons:
            for k in epochs:
                for l in batch_size:
                        cfg = [i, j, k, l]
                        configs.append(cfg)
    print(f'Total de Configurações Definidas:{len(configs)}')
    return configs

'''
def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df.tail(timesteps).values
    input_seq = input_seq.reshape(1,timesteps,1)
    inp = input_seq
    forecasts = list()
    #multisteps tells us how many iterations we want to perform, i.e., how many days we want to predict
    for step in range(1,multisteps+1): #meu step vai de 1 a 5 (1,2,3,4,5)
        y_predicao = model.predict(inp[:, step:])[:, np.newaxis, :]
        inp = np.concatenate([inp, y_predicao],axis=1)
    forecasts = inp[:, timesteps:]
    print(forecasts)
    forecasts = forecasts.reshape(multisteps,1)
    forecasts = scaler.inverse_transform(forecasts)
    return forecasts
'''
#=====================================================================
#                                Plot Functions
#=====================================================================

#Plot time Series data
def plot_incidents(data):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data)
    plt.title('Number of incidents per Day')
    plt.ylabel('Incidents')
    plt.xlabel('Days')
    plt.show()

def plot_learning_curves(loss, val_loss):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



def plot_forecast(data, forecasts):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data, color='green', label='Confirmed')
    plt.plot(range(len(data)-1, len(data)+len(forecasts)-1), forecasts, color='red', label='Forecasts')
    plt.title('Number of Incidents')
    plt.ylabel('Incidents')
    plt.xlabel('Days')
    plt.legend()
    plt.show()


        #=====================================================================
        #
        #                            Main Execution
        #                               
        #=====================================================================



#Carrega o DataSet
df_raw = load_data(path)
df_D = Data_Uni_Daily(df_raw)

#Transform rows with 0 to nan and drop
df_1 = df_D.replace(0, np.nan)
df_1 = df_D.dropna(how='all', axis=0)


multisteps = 5 #number of days to forecast - we will forecast the next 5 days
cv_splits = 3

#Definir diferentes conjuntos de configurações para o model (neste exemplos existem 12 config possiveis)
timestep = [5,7]
h_neurons = [64, 128]
epochs = [10, 20]
batch_size = [5, 7]

plot_incidents(df_1) #Plot incident per day
scaler = data_normalization(df_1) #scaling data to [-1, 1]

cfg_list = model_configs(timestep, h_neurons, epochs, batch_size) #Recebe a lista das configurações
scores = grid_search(df_1,cfg_list) #Recebe uma lista de tuplas ordenadas com a configuracao e loss respectivamente
print("                                                                  ")
print("-----------------------Modelos Testados--------------------------" )
print("                                                                  ")
for cfg, loss in scores:
    print(f'Configuracao: {cfg}\n Loss: {loss:.4f}')
    
'''
forecasts = forecast(model, df_D_Uni, timesteps, multisteps, scaler)
#print(fore)
data_plot_forecast = scaler.inverse_transform(df_D_Uni)
plot_forecast(data_plot_forecast, forecasts)
#plot_multiple_forecasts(df_data, forecasts)
'''