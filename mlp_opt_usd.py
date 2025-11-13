import os
import gc
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials
import pickle


def sample_generator_numeric(x_path, y_path, validation_split, batch_size):
    x = np.load(x_path)  # Already sequenced and scaled
    y = np.load(y_path)

   
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, random_state=42)

    train_length = len(x_train)
    val_length = len(x_val)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(train_length).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(val_length).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_data, val_data, train_length, val_length


# Define the MLP model
def build_mlp(hidden_units, dense_layers, dropout_rate):
    seq_size = 24
    model = Sequential()
    model.add(Flatten(input_shape=(seq_size, 1)))  # Flatten input to make it suitable for dense layers

    for _ in range(dense_layers):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
    return model

# Hyperparameter optimization objective function
def train_model(cfg):
    number_of_models = 1
    hidden_units = cfg['hidden_units']
    dense_layers = cfg['dense_layers']
    dropout_rate = cfg['dropout_rate']
    validation_split = cfg['validation_split']
    batch_size = cfg['batch_size']
    x_path = cfg['x_path']
    y_path = cfg['y_path']
    learning_rate = cfg['learning_rate']
    patience = cfg['patience']
    x_vali_path = cfg['x_vali']
    y_vali_path = cfg['y_vali']

    # Build and Train the MLP model
    for i in range(number_of_models):
        print('Train MLP model:', (i + 1))

        train_set, val_set, train_length, val_length = sample_generator_numeric(x_path, y_path, validation_split, batch_size)
        train_steps = train_length // batch_size + 1
        val_steps = val_length // batch_size + 1

        kmodel = build_mlp(hidden_units, dense_layers, dropout_rate)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, min_delta=0.0001, restore_best_weights=True)
        kmodel.compile(optimizer=opt, loss='mae')

        kmodel.fit(train_set, epochs=50, validation_data=val_set, verbose=1, callbacks=[es], steps_per_epoch=train_steps, validation_steps=val_steps)

        path = f'/gpfs-home/p220127ma/Benchmark_Models/mlp_usd_model_{i}.h5'
        kmodel.save(path)

    x_vali = np.load(x_vali_path)
    y_vali = np.load(y_vali_path)
    vali_length = len(x_vali)
    stepz = vali_length // batch_size + 1
    val = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
    val = val.shuffle(buffer_size=vali_length)
    val = val.repeat()
    val = val.batch(batch_size=64)
    val = val.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_loss = kmodel.evaluate(val, steps=stepz)

    # Clear memory
    tf.keras.backend.clear_session()
    gc.collect()

    return {'loss': val_loss, 'status': STATUS_OK}

# Define the search space for hyperparameters
space = {
    'validation_split': 0.2,
    'hidden_units': hp.choice('hidden_units', [32, 64, 128, 256]),
    'dense_layers': hp.choice('dense_layers', [1, 2, 3]),
    'dropout_rate': hp.choice('dropout_rate', [0.1, 0.2, 0.5]),
    'batch_size': hp.choice('batch_size', [64, 128, 256, 512]),
    'learning_rate': hp.choice('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01]),
    'patience': hp.choice('patience', [2, 5, 10]),
    'x_path': '/gpfs-home/p220127ma/Meta_Data/x_train_new.npy',
    'y_path': '/gpfs-home/p220127ma/Meta_Data/y_train_new.npy',
    'x_vali': '/gpfs-home/p220127ma/Meta_Data/x_vali.npy',
    'y_vali': '/gpfs-home/p220127ma/Meta_Data/y_vali.npy'
}

# Hyperparameter optimization
trials = SparkTrials(parallelism=4)
best = fmin(train_model, space, algo=tpe.suggest, max_evals=30, trials=trials)

# Save results
print(best)
with open('MLP_usd_opt_Gold_result.pkl', 'wb') as f:
    pickle.dump(best, f)
