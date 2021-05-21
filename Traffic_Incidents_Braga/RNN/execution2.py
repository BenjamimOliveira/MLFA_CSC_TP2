# -*- coding: utf-8 -*-

from data_processing import load_data, split_data, to_univariate,to_daily_uni_multi, missing_values, data_normalization, to_supervised, prepare_train, plot_incidents
from models import generate_configs, grid_search
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np

timestep =  [7,6,5,3]
features = [4] #Deixar somente 1 parametro,n = 1 -> univariate, (n > 1) -> Multivariate
h_neurons =  [128,64,32]
activations = ['relu', 'tanh']
epochs = [50,40,30,20,15]
batch_size = [10, 7, 5,2]


#Prametros Exclusivos para CNN------------------
filters = [16]                                  #
kernel_size = [5]                               #
pool_size = [2]                                 #
#-----------------------------------------------

#methods to select 'univariate', 'multivariate'
method = 'multivariate' #Some models need y in shape [x,x,x], and others [x,x]. Univariate = [x,x] Multivariate = [x,x,x]
path = 'Traffic_Braga.csv'

df_raw = load_data(path)        # load dataset
df_uni = to_daily_uni_multi(method,df_raw)
df_viz = df_uni 
df_uni = missing_values(df_uni) # deal with missing, comment if use masking
''' See if The Data is Stationary
df_viz_raw = missing_values(df_raw)

df_viz = to_daily(df_raw)    # convert data to univariate and group incidents by day
df_viz = missing_values(df_viz)
adf, pvalue, usedlag_,  nobs_, critical_values_, icbest_ = adfuller(df_viz)
print('pvalue = ', pvalue,'if above 0.05, data is not stationary')
'''

scaler = data_normalization(method,df_uni) # scaling data to [-1, 1]

#Models do select (lstm, gru, lstm_auto, gru_auto)
model_name = 'lstm' #Select Model (lstm, gru, srnn)

configs = generate_configs(timestep, features, h_neurons, activations, epochs, batch_size) # generate a list with all the possible configurations

#config_cnn = generate_configs(timestep, filters, kernel_size, pool_size)


scores = grid_search(df_uni, model_name, configs, method)                   # sort configuration by performance

for cfg, loss, loss_mae, loss_rmse in scores: 
    print(f'Configuracao: {cfg}\n loss: {loss:.4f}\n mae: {loss_mae:.4f}\n rmse: {loss_rmse:.4f}')

df_score = pd.DataFrame(scores)
df_score.to_csv('Score_models_Daily_GRU_encoder_Masking2.csv')

