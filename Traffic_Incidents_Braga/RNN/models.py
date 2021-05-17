import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from Data.data_processing import to_supervised, prepare_train

# for replicability purposes
tf.random.set_seed(91195003) 
np.random.seed(91190530)
# for an easy reset backend session state 
tf.keras.backend.clear_session()

multisteps = 5 # number of days to forecast (we will forecast the next 5 days)

'''
Define loss function (root mean square error)
'''
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

'''
Build a LSTM model from a configuration
'''
def build_lstm(config):
    timesteps, h_neurons, epochs, batch_size = config
    features=1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_lstm.png', show_shapes=True)
    return model

'''
Build a GRU model from a configuration
'''
def build_gru(config):
    timesteps, h_neurons, epochs, batch_size = config
    features=1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # show model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'Traffic_gru.png', show_shapes=True)
    return model

'''
Compile model and fit it to the data
'''
def compile_and_fit(df, model, config):
    timesteps, h_neurons, epochs, batch_size = config
    univariate = 1
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
def generate_configs(timestep, h_neurons, epochs, batch_size):
    # create configs
    configs = list()
    for i in timestep:
        for j in h_neurons:
            for k in epochs:
                for l in batch_size:
                    config = [i, j, k, l]
                    configs.append(config)
    print('Total number of defined configurations', len(configs))
    return configs

'''
Check the training performances from a model for a configuration
'''
def call_models(df, model_name, config, n_repeats=1):
    timesteps, h_neurons, epochs, batch_size = config
    univariate = 1 # Quando fazer modelo multivariate mudar
    to_supervised(df, timesteps)
    if model_name == 'lstm':
        model = build_lstm(config)
    else:
        model = build_gru(config)
    hist, loss, loss_mae, loss_rmse = compile_and_fit(df, model, config)
    print('Configuration: timestep=%s, h_neurons=%s, epochs=%s, batch_size=%s' % (config[0], config[1], config[2], config[3]))
    print('loss:', loss)
    print('mae:', loss_mae)
    print('rmse:', loss_rmse)
    return (str(config), loss_mae, loss_rmse)

'''
Train a model for each configuration and sort the configurations by performance
'''
def grid_search(df, model_name, configs):
    # evaluate configs
    scores = []
    for config in configs:
        scores.append(call_models(df, model_name, config))
    # sort configs by error in ascending order
    scores.sort(key=lambda tup: tup[1])
    return scores

'''
Recursive multi-step forecast
'''
def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df[-timesteps:].values # get the last known sequence
    inp = input_seq
    forecasts = list()
    for step in range(1, multisteps+1):
        print(inp.shape)
        prediction = model.predict(inp)
        forecasts.append(prediction)
        inp = np.append(inp, [[prediction]], axis=0) # insert prediction to the sequence
        np.delete(inp, 0)                            # remove oldest value of the sequence
    return forecasts

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