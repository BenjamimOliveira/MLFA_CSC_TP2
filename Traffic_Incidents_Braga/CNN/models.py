from typing import Sequence
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import itertools

from Data.data_processing import to_supervised, prepare_train

# for replicability purposes
tf.random.set_seed(91195003) 
np.random.seed(91190530)
# for an easy reset backend session state 
tf.keras.backend.clear_session()

'''
Define loss function (root mean square error)
'''
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

'''
Build a CNN model from a configuration
'''
def build_cnn(config, features):
    timesteps, _, _, filters, kernel_size, pool_size = config
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

'''
Compile model and fit it to the data
'''
def compile_and_fit(df, model, config):
    _, epochs, batch_size, _, _, _ = config
    # compile the model
    model.compile(loss=rmse, optimizer=tf.keras.optimizers.Adam(), metrics=['mae', rmse])
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_train(df, config)
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, shuffle=False) 
    metrics = model.evaluate(X_test, y_test)
    hist = history
    loss = metrics[0]
    loss_mae = metrics[1]
    loss_rmse = metrics[2]
    plot_learning_curves(history.history['loss'], history.history['val_loss']) 
    return hist, loss, loss_mae, loss_rmse

'''
Generate a list with all the possible configurations with the parameters timestep, h_neurons, epochs and batch_size
'''
def generate_configs(timesteps, epochs, batch_size, filters, kernel_size, pool_size):
    configs = [timesteps, epochs, batch_size, filters, kernel_size, pool_size]
    configs = list(itertools.product(*configs))
    print('Generated %s different configurations' % (len(configs)))
    return configs

'''
Check the training performances from a model for a configuration
'''
def call_model(df, config, features):
    timesteps, epochs, batch_size, filters, kernel_size, pool_size = config
    to_supervised(df, timesteps)
    model = build_cnn(config, features)
    hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config)
    print('Configuration: timesteps=%s, epochs=%s, batch_size=%s, filters=%s, kernel_size=%s, pool_size=%s' % (timesteps, epochs, batch_size, filters, kernel_size, pool_size))
    print('loss:', loss)
    print('mae:', loss_mae)
    print('rmse:', loss_rmse)
    return model, (str(config), loss_mae, loss_rmse)

'''
Train a model for each configuration and sort the configurations by performance
'''
def grid_search(df, configs, features):
    # evaluate configs
    scores = []
    for config in configs:
        scores.append(call_model(df, config, features)[1])
    # sort configs by error in ascending order
    scores.sort(key=lambda tup: tup[1])
    return scores

'''
Univariate single-step forecast
'''
def forecast_single(model, df, timesteps, scaler):
    sequence = df[-timesteps:].values        # get the last known sequence
    inp = sequence.reshape(-1, timesteps, 1) # reshape input
    prediction = model.predict(inp)          # predict number of incidents

    prediction_scaled = scaler.inverse_transform(prediction) # denormalize prediction
    prediction_scaled = prediction_scaled[0][0]

    return prediction_scaled

'''
Univariate recursive multi-step forecast
'''
def forecast_multi(model, df, timesteps, multisteps, scaler):
    sequence = df[-timesteps:].values # get the last known sequence
    forecasts = list()
    for step in range(1, multisteps+1):
        inp = sequence.reshape(-1, timesteps, 1) # reshape input
        prediction = model.predict(inp)          # predict number of incidents

        forecast = scaler.inverse_transform(prediction) # denormalize prediction
        forecast = forecast[0][0]
        forecasts.append(forecast)

        temp = np.concatenate((inp[0], prediction)) # insert prediction to the sequence
        temp = np.delete(temp, 0)                   # remove oldest value of the sequence
        sequence = temp.reshape((timesteps, 1))     # reshape sequence 
    return forecasts

'''
Multivariate single-step forecast
'''
def forecast_single_multi(model, df, timesteps, features, scaler):
    sequence = df[-timesteps:].values               # get the last known sequence
    inp = sequence.reshape(-1, timesteps, features) # reshape input
    prediction = model.predict(inp)                 # predict number of incidents

    temp = np.tile(prediction, (1, 4))    # duplicate prediction to get the right shape to denormalize
    temp = scaler.inverse_transform(temp) # denormalize prediction
    forecast = temp[0][0]
    return forecast

'''
Plot learning curves
'''
def plot_learning_curves(loss, val_loss):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

'''
Plot forecast
'''
def plot_forecast(data, forecasts):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data, color='green', label='Confirmed')
    plt.plot(range(len(data)-1, len(data)+len(forecasts)-1), forecasts, color='red', label='Forecasts')
    plt.title('Number of Incidents')
    plt.ylabel('Incidents')
    plt.xlabel('Days')
    plt.legend()
    plt.show()