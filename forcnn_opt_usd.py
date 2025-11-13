import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import gc
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials
import pickle

def identity_block(inp, filters, res):
    # First block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Second block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Third block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # If it is a residual block -> implement the identity shortcut
    if res:
        x_shortcut = inp
        x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.LeakyReLU()(x)

    return x

def convolutional_block(inp, filters, res):
    # First block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Second block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Third block of the MAIN path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # If it is a residual block -> implement the convolutional shortcut to match dimensions
    if res:
        x_shortcut = inp
        x_shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(x_shortcut)
        x_shortcut = tf.keras.layers.BatchNormalization()(x_shortcut)
        x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.LeakyReLU()(x)

    return x

def build_convolutional_model(bottleneck, blocks, l_per_block, starting_filters, residual):
    f = starting_filters
    inp = tf.keras.layers.Input(shape=(64, 64, 1))
    x = tf.keras.layers.BatchNormalization()(inp)

    # Encoder
    for i in range(blocks):
        x = convolutional_block(x, f, residual)
        for j in range(l_per_block - 1):
            x = identity_block(x, f, residual)
        x = tf.keras.layers.Conv2D(f, (2, 2), strides=(2, 2), padding='same')(x)
        f = f * 2

    # Bottleneck
    x = identity_block(x, int(f / 2), residual)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(bottleneck)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # MLP module
    x = tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(bottleneck, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='he_uniform')(x)

    return tf.keras.models.Model(inp, x)

def sample_generator(x_path, y_path, validation_split, batch_s):
    def process_images(x_im, y_h):
        x_im = (tf.cast(x_im, tf.float32) / 255)
        x_im = tf.reshape(x_im, (64, 64, 1))
        return x_im, y_h
    # Load Data
    x = np.load(x_path)
    y = np.load(y_path)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)

    train_length = len(x_train)
    val_length = len(x_val)

    # Training Sample

    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data_train = data_train.shuffle(buffer_size=train_length).map(process_images).repeat().batch(batch_size=batch_s).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Validation Sample
    data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    data_val = data_val.shuffle(buffer_size=val_length).map(process_images).repeat().batch(batch_size=batch_s).prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train, data_val, train_length, val_length

def train_models(cfg):
    number_of_models = 1

    bottleneck = cfg['bottleneck']
    num_blocks = cfg['number_blocks']
    num_layers = cfg['number_layers']
    batch_size = cfg['batch_size']
    val_split = cfg['validation_split']
    residual = cfg['residual']
    str_filters = cfg['starting_filters']
    x_path = cfg['x_path']
    y_path = cfg['y_path']
    lng =cfg['lng']
    esp = cfg['esp']
    x_vali = cfg['x_vali']
    y_vali = cfg['y_vali']


# Build and Train the forecasting model
    for i in range(number_of_models):
        print('Train Forecasting model:', (i + 1))

        train_set, val_set, train_length, val_length = sample_generator(x_path, y_path, val_split, batch_size)
        train_steps = train_length // batch_size + 1
        val_steps = val_length // batch_size + 1

        kmodel = build_convolutional_model(bottleneck, num_blocks, num_layers, str_filters, residual)
        opt = tf.keras.optimizers.Adam(learning_rate=lng, amsgrad=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=esp, min_delta=0.0001, restore_best_weights=True)
        kmodel.compile(optimizer=opt, loss='mae')


        kmodel.fit(train_set, epochs=50, validation_data=val_set, verbose=1, callbacks=[es], steps_per_epoch=train_steps, validation_steps=val_steps)
        
        path = '/gpfs-home/p220127ma/ForCNN_Models/ForCNN_SD/USD/forcnn_sd_USD_' + str(i) + '.h5'
        kmodel.save(path)

    def process_images(x_im, y_h):
            x_im = (tf.cast(x_im, tf.float32) / 255)
            x_im = tf.reshape(x_im, (64, 64, 1))
            return x_im, y_h


    x_vali = np.load(x_vali)
    y_vali = np.load(y_vali)
    vali_length = len(x_vali)
    stepz = vali_length // batch_size + 1
    val = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
    val = val.shuffle(buffer_size=vali_length)
    val = val.map(process_images)
    val = val.repeat()
    val = val.batch(batch_size=64)
    val = val.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_loss = kmodel.evaluate(val, steps=stepz)
    
    #del train_set, val_set, x_vali, y_vali, val
    # Clear memory
    tf.keras.backend.clear_session()
    gc.collect()
     
    return(val_loss)

space = {
    'bottleneck': 1024,
    'number_blocks': hp.choice('number_blocks', [2, 3, 4, 5]),
    'number_layers': hp.choice('number_layers', [1, 2, 3]),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'validation_split': 0.2,
    'residual': hp.choice('residual', [True, False]),
    'starting_filters': hp.choice('starting_filters', [8, 16, 32]),
    'lng': hp.choice('lng', [0.0001, 0.0005, 0.001]),
    'esp': hp.choice('esp',[2,5,10]),
    'x_path': '/gpfs-home/p220127ma/Images_USD/USD/x_train_new.npy',
    'y_path': '/gpfs-home/p220127ma/Images_USD/USD/y_train_new.npy',
    'x_vali': '/gpfs-home/p220127ma/Images_USD/USD/x_vali.npy',
    'y_vali':'/gpfs-home/p220127ma/Images_USD/USD/y_vali.npy'
}

# Add SparkTrials for parallel execution
trials = SparkTrials(parallelism=4)
 

best = fmin(train_models, space, algo=tpe.suggest, max_evals=30, trials=trials)

print(best)

with open('ForCNN_opt_USD_Gold_result.pkl','wb') as f:
    pickle.dump(best, f)