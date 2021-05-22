# univariative
from numpy.lib.function_base import append
from metro_univariative2 import loadData as dataUni
from metro_univariative2 import splitData
from metro_univariative2 import rmse
from metro_univariative2 import to_supervised as to_supervised_uni
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import csv
from pickle import load
from matplotlib import pyplot as plt

def forecast_multi(model, df, timesteps, multisteps, scaler):
    X = df.reshape(-1,12,1)

    forecasts = []
    for x in range(multisteps):
        pred = float(model.predict(X))
        X = np.delete(X,0)
        X = np.append(X,pred)
        X = X.reshape(-1,12,1)
        forecasts.append(pred)
    return forecasts

# carregar dados
data = dataUni(path="../data/metro_multi_pred.csv")
# normalizar dados
scalertraffic = load(open('../models/scalerTraffic.pkl', 'rb'))
data[['traffic_volume']] = scalertraffic.transform(data[['traffic_volume']])

# dividir dados

# carregar parametros
file_CSV = open('../models/uni_config_gru.csv')
data_CSV = csv.reader(file_CSV)
config = list(data_CSV)
config = config[0]

# to supervised
testX, testy  = to_supervised_uni(data, int(config[1]))

# model
model = tf.keras.models.load_model('../models/uni_gru.h5', custom_objects={'rmse': rmse})

aaa = forecast_multi(model, testX[0], 12, 3, scalertraffic)
print("data")
print(data[0:15])
print("pred")
print(aaa)
data = data.values
aaa = [data[11]] + aaa

data = scalertraffic.inverse_transform(data)
aaa = np.reshape(aaa, (4,-1))
aaa = scalertraffic.inverse_transform(aaa)
print(aaa)
X = []
for x in range(11,15):
    X.append(x)

X_pred = []
for x in range(11,15):
    X_pred.append(x)

plt.plot(X,data[11:15])
plt.plot(X_pred,aaa)
plt.show()
# create model
#model = tf.keras.models.load_model('../models/uni_gru.h5', custom_objects={'rmse': rmse})
#model.summary()
#model.evaluate(testX,testy)