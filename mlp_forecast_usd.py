import numpy as np
import pandas as pd
import os
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def sample_generator_numeric(x_path, y_path, validation_split, batch_size):
    x = np.load(x_path)  # Already sequenced and scaled
    y = np.load(y_path)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, random_state=42)

    train_length = len(x_train)
    val_length = len(x_val)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(train_length).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(val_length).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_data, val_data, train_length, val_length

# Define MLP model
def build_mlp(hidden_units, dense_layers, dropout_rate):
    seq_size = 24  # Length of input sequence
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(seq_size,)))
    for _ in range(dense_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    return model

# Train models
def train_model(cfg):
    number_of_models = 1
    hidden_units = cfg['hidden_units']
    dense_layers = cfg['dense_layers']
    dropout_rate = cfg['dropout_rate']
    validation_split = cfg['validation_split']
    batch_size = cfg['batch_size']
    x_path = cfg['x_path']
    y_path = cfg['y_path']
    learning_rate = cfg['lng']
    esp = cfg['esp']

    for i in range(number_of_models):
        print(f'Train Forecasting model: {i + 1}')

        train_set, val_set, train_length, val_length = sample_generator_numeric(x_path, y_path, validation_split, batch_size)
        train_steps = train_length // batch_size + 1
        val_steps = val_length // batch_size + 1

        kmodel = build_mlp(hidden_units, dense_layers, dropout_rate)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=esp, min_delta=0.0001, restore_best_weights=True)
        kmodel.compile(optimizer=opt, loss='mae')

        history=kmodel.fit(train_set, epochs=50, validation_data=val_set, verbose=1, callbacks=[es], steps_per_epoch=train_steps, validation_steps=val_steps)

        model_path = f'/gpfs-home/p220127ma/Benchmark_Models/mlpusd_{i}.h5'
        kmodel.save(model_path)
        
        #Save training and Validation Loss to CSV
        loss_df=pd.DataFrame(history.history)
        loss_csv_path=f'/gpfs-home/p220127ma/Model_Loss/mlp_loss_history{i}.csv'
        loss_df.to_csv(loss_csv_path, index=False)
        print(f"Loss history saved to: {loss_csv_path}")

        tf.keras.backend.clear_session()
        gc.collect()

# Generate forecasts
def generate_forecasts(path_test, save_model_p, save_forecasts):
    number_of_models = 1

    # Load test data
    df_x_test = np.load(path_test)
    print(f"Original shape of df_x_test: {df_x_test.shape}")

    seq_length = 24
    df_x_test = df_x_test.reshape((-1, seq_length))
    print("Reshaped df_x_test for MLP: ", df_x_test.shape)

    y_hat_all = []
    for i in range(number_of_models):
        print(f'Forecasting with model: {i + 1}')
        path = save_model_p + str(i) + '.h5'
        model = tf.keras.models.load_model(path)
        y_hat = model.predict(df_x_test)
        y_hat_all.append(y_hat)
    y_hat_all = np.asarray(y_hat_all)
    
    path = '/gpfs-home/p220127ma/Meta_Data/gold_usd_test.csv'
    df_in = pd.read_csv(path, sep=',', decimal='.', usecols=[1])

    # Post-process forecasts
    #ts_in = np.asarray(df_in)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler.fit(np.reshape(ts_in, (-1, 1)))

    y_hat = y_hat_all[0, :]
    #y_hat = scaler.inverse_transform(np.reshape(y_hat, (-1, 1))).reshape(-1)

    np.save('/gpfs-home/p220127ma/Forecasts_Benchmarks/forecast_mlp_usd.npy', y_hat)
    if save_forecasts:
        path = '/gpfs-home/p220127ma/Forecasts_Benchmarks/frc_mlp_usd.csv'
        np.savetxt(path, y_hat, delimiter=',')

# Configuration
configuration = {
    'hidden_units': 64,
    'dense_layers': 1,
    'dropout_rate': 0.2,
    'batch_size': 64,
    'lng': 0.0005,
    'esp': 10,
    'validation_split': 0.2,
    'x_path': '/gpfs-home/p220127ma/Meta_Data/x_train_full.npy',
    'y_path': '/gpfs-home/p220127ma/Meta_Data/y_train_full.npy'
}

# Train and generate forecasts
train_model(configuration)

ts_path = '/gpfs-home/p220127ma/Meta_Data/x_test_full.npy'
md_path = '/gpfs-home/p220127ma/Benchmark_Models/mlpusd_'
generate_forecasts(ts_path, md_path, True)
