from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import tensorflow as tf
from keras.losses import MeanAbsoluteError
import pandas as pd
import numpy as np
import gc


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
    # Parameters
    f = starting_filters

    # AE construction
    inp = tf.keras.layers.Input(shape=(64, 64, 1))
    x = tf.keras.layers.BatchNormalization()(inp)

    # Encoder
    for i in range(blocks):
        # Convolutional layers
        x = convolutional_block(x, f, residual)
        for j in range(l_per_block-1):
            x = identity_block(x, f, residual)
        # Pooling layer
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

    # Load data
    x = np.load(x_path)
    y = np.load(y_path)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)

    train_length = len(x_train)
    val_length = len(x_val)

    # Training samples
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.map(process_images)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Validation samples
    data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    data_val = data_val.shuffle(buffer_size=val_length)
    data_val = data_val.map(process_images)
    data_val = data_val.repeat()
    data_val = data_val.batch(batch_size=batch_s)
    data_val = data_val.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train, data_val, train_length, val_length


def train_models(cfg):
    number_of_models = 1

    # Unpack configuration
    bottleneck = cfg['bottleneck']
    num_blocks = cfg['number_blocks']
    num_layers = cfg['number_layers']
    batch_size = cfg['batch_size']
    val_split = cfg['validation_split']
    residual = cfg['residual']
    lng =cfg['lng']
    esp = cfg['esp']
    str_filters = cfg['starting_filters']
    x_path = cfg['x_path']
    y_path = cfg['y_path']

    # Build & Train the Forecasting models
    for i in range(number_of_models):
        print('Train Forecasting model:', (i + 1))

        train_set, val_set, train_length, val_length = sample_generator(x_path, y_path, val_split, batch_size)
        train_steps = train_length // batch_size + 1
        val_steps = val_length // batch_size + 1
 
        kmodel = build_convolutional_model(bottleneck, num_blocks, num_layers, str_filters, residual)
        opt = tf.keras.optimizers.Adam(learning_rate=lng, amsgrad=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=esp,
                                              min_delta=0.0001, restore_best_weights=True)
        kmodel.compile(optimizer=opt, loss=MeanAbsoluteError())
        history=kmodel.fit(train_set, epochs=50, validation_data=val_set, verbose=1, callbacks=[es],
                   steps_per_epoch=train_steps, validation_steps=val_steps)
        model_path = '/gpfs-home/p220127ma/ForCNN_Models/forcnn_opt_usd_' + str(i) + '.h5'
        kmodel.save(model_path)
        
        #Save training and Validation Loss to CSV
        loss_df=pd.DataFrame(history.history)
        loss_csv_path=f'/gpfs-home/p220127ma/Model_Loss/forcnn_usd_loss_history{i}.csv'
        loss_df.to_csv(loss_csv_path, index=False)
        print(f"Loss history saved to: {loss_csv_path}")

        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()


def generate_forecasts(path_test, save_model_p, save_forecasts):
    number_of_models = 1

    # Read test data
    path = "/gpfs-home/p220127ma/Meta_Data/gold_usd_test.csv"
    df_in = pd.read_csv(path, sep=',', decimal='.', usecols=[1])  # Only read the relevant column

    df_x_test = np.load(path_test)
    df_x_test = df_x_test.astype(float) / 255
    df_x_test = df_x_test.reshape((len(df_x_test), 64, 64, 1))

    # Load forecasting models & ensemble
    y_hat_all = []
    for i in range(number_of_models):
        print('Forecasting with model:', (i + 1))
        model_path = save_model_p + str(i) + '.h5'
        model = tf.keras.models.load_model(model_path)
        y_hat = model.predict(df_x_test)
        y_hat_all.append(y_hat)
    
    y_hat_all = np.asarray(y_hat_all)

    # Load forecasting models & ensemble
    y_hat_all = list([])
    for i in range(number_of_models):
        print('Forecasting with model:', (i + 1))
        path = save_model_p + str(i) + '.h5'
        model = tf.keras.models.load_model(path)
        y_hat=model.predict(df_x_test)
        y_hat_all.append(y_hat)
        y_hat_all = np.asarray(y_hat_all)

    # Process forecasts (scale back to original level)
    
    #ts_in = np.asarray(df_in)  # Extract the entire time series from the first column of df_in


    #scaler = MinMaxScaler(feature_range=(0, 1))    # Fit the scaler on the last 18 values of the time series 
    #scaler.fit(np.reshape(ts_in, (-1, 1)))

    y_hat=y_hat_all[0,:]
    #y_hat = scaler.inverse_transform(np.reshape(y_hat, (-1, 1))).reshape(-1)


    np.save('/gpfs-home/p220127ma/Forecasts_ForCNN/forecast_forcnn_usd.npy', y_hat)
    
    if save_forecasts:
        path = '/gpfs-home/p220127ma/Forecasts_ForCNN/frc_forcnn_usd.csv'
        np.savetxt(path, y_hat, delimiter=',')


configuration = {
    'bottleneck': 1024,
    'number_blocks': 5,
    'number_layers': 2,
    'batch_size': 64,
    'lng': 0.001,
    'esp': 5,
    'validation_split': 0.2,
    'residual': False,
    'starting_filters': 32,
    'x_path': '/gpfs-home/p220127ma/Images_USD/USD/x_train.npy',
    'y_path': '/gpfs-home/p220127ma/Images_USD/USD/y_train.npy'
    }
train_models(configuration)

ts_path = '/gpfs-home/p220127ma/Images_USD/USD/x_test.npy'
md_path = '/gpfs-home/p220127ma/ForCNN_Models/forcnn_opt_usd_'
generate_forecasts(ts_path, md_path, True)
