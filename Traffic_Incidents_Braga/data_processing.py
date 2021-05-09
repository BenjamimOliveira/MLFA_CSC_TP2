import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from numpy import nan

cv_splits = 3  # time series cross validator

'''
Load dataset
'''
def load_data(path):
    raw_data = pd.read_csv(path,infer_datetime_format=True)
    return raw_data 

'''
Split data into training and validation sets 
'''
def split_data(training, perc=10):
    train_raw = np.arange(0, int(len(training)*(100-perc)/100))                    # contains the first 90% of the data
    validation_raw = np.arange(int(len(training)*(100-perc)/100+1), len(training)) # contains the last 10%
    return train_raw, validation_raw 

'''
Convert data to univariate, using a single feature (incident_date) as input
'''
def to_univariate(df):
    df_uni = df.drop(columns=['cause_of_incident', 'city_name', 'description', 'cause_of_incident', 'from_road', 'to_road', 
                              'affected_roads', 'incident_category_desc','magnitude_of_delay_desc', 'length_in_meters',
                              'delay_in_seconds', 'latitude','longitude'])
    return df_uni

'''
Prepare data to have the number of daily incidents
'''
def to_daily(df):
    df_uni = to_univariate(df)
    df_uni['incident_date'] = df_uni['incident_date'].str[:10]                # delete the last 10 characters
    df_uni['Incidents'] = pd.DataFrame([1 for x in range(len(df_uni.index))]) # create a column with 1 to sum the incidents per day
    df_uni = df_uni.set_index('incident_date')                                # set the column incident_date to index
    df_uni.index = pd.to_datetime(df_uni.index)                               # convert the date in index from string to Date type
    daily_groups = df_uni.resample('D')                                       # sum groupy by day
    daily_data = daily_groups.sum()
    return daily_data

'''
Deal with missing values in the data
'''
def missing_values(df):
    df = df.replace(0, np.nan)        # replace instances with 0 incidents with NaN
    df = df.dropna(how='all', axis=0) # remove all instances with NaN
    return df

'''
Normalize the data to the range [-1, 1]
'''
def data_normalization(df, norm_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    df[['Incidents']] = scaler.fit_transform(df[['Incidents']])
    return scaler

'''
Convert the data to supervised
'''
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

'''
Prepare the training, validation and testing sets from a configuration
'''
def prepare_train(df, config):
    timesteps, h_neurons, epochs, batch_size = config
    X, y = to_supervised(df, timesteps)
    # time series cross validator
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for train_index, test_index in tscv.split(X):
        train_idx, val_idx = split_data(train_index, perc=10) # further split into training and validation sets
        # build data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_index], y[test_index]
        #model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(), metrics = ['mae', rmse])
    return X_train, y_train, X_val, y_val, X_test, y_test

'''
Plot Time Series data
'''
def plot_incidents(data):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data)
    plt.title('Number of incidents per Day')
    plt.ylabel('Incidents')
    plt.xlabel('Days')
    plt.show()