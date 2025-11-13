from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import PIL.Image
import os

def fig2data(fig):
    """Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it."""
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)  # Note: shape should be (height, width, channels)

    # canvas.tostring_argb gives pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode.
    buf = np.roll(buf, 3, axis=2)
    return buf

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image in RGBA format and return it."""
    buf = fig2data(fig)
    w, h, d = buf.shape
    return PIL.Image.frombytes("RGBA", (h, w), buf.tobytes())

input_size = 24
forecasting_horizon = 1

# Directory setup
output_path = "/gpfs-home/p220127ma/Images_USD/USD"
os.makedirs(output_path, exist_ok=True)

# Load training data
df = pd.read_csv("/gpfs-home/p220127ma/Meta_Data/gold_usd_train.csv", usecols=['Close'])['Close'].values
ts = np.asarray(df).flatten()
print(len(ts))
df_x_train = []
df_y_train = []

# Generating training images
while len(ts) >= input_size + forecasting_horizon:
    y_train = ts[-forecasting_horizon:]
    x_train = ts[:-forecasting_horizon]
    x_train = x_train[-input_size:]

    # Min-Max scale input data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(np.reshape(x_train, (-1, 1))).reshape((-1))
    y_train = scaler.transform(np.reshape(y_train, (-1, 1))).reshape(-1)
    df_y_train.append(y_train)

    # Plot, process, and convert to image array
    fig = plt.figure(facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.plot(x_train, linewidth=5.0, color='white')
    ax.patch.set_facecolor('black')
    plt.xlim(-1, input_size)
    plt.ylim(-0.05, 1.05)
    
    # Convert plot to image
    im = fig2img(fig)
    im = im.convert(mode='L')
    im = im.resize((64, 64), PIL.Image.LANCZOS)
    df_x_train.append(np.asarray(im).flatten())
    plt.close(fig)

    # Trim the series
    ts = ts[:-1]

df_x_train = np.array(df_x_train)
df_y_train = np.array(df_y_train)

# Save the training arrays
np.save(os.path.join(output_path, 'x_train.npy'), df_x_train)
np.save(os.path.join(output_path, 'y_train.npy'), df_y_train)

print('x_train:', df_x_train.shape)
print('y_train:', df_y_train.shape)

# Load test data
test_data =  pd.read_csv("/gpfs-home/p220127ma/Meta_Data/gold_usd_test.csv", usecols=['Close'])['Close'].values
df_x_test = []
df_y_test=[]
df_x_test_numerical = []
df_y_test_numerical=[]


# Generating test images
while len(test_data) >=input_size:
    x_test = test_data[:input_size]
    df_x_test_numerical.append(x_test)

    if len(test_data) > input_size:  # Ensure there are enough data points for y_test
        y_test = test_data[input_size]
        df_y_test.append(y_test)
        df_y_test_numerical.append(y_test)
    else:
        break  # Exit loop if not enough data
    


    # Scale the test data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_test = scaler.fit_transform(np.reshape(x_test, (-1, 1))).reshape((-1))

    # Plot, process, and convert to image array
    fig = plt.figure(facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.plot(x_test, linewidth=5.0, color='white')
    ax.patch.set_facecolor('black')
    plt.xlim(-1, input_size)
    plt.ylim(-0.05, 1.05)

    # Convert plot to image
    im = fig2img(fig)
    im = im.convert(mode='L')
    im = im.resize((64, 64), PIL.Image.Resampling.LANCZOS)
    df_x_test.append(np.asarray(im).flatten())
    plt.close(fig)

    # Trim the series
    test_data = test_data[1:]

df_x_test = np.array(df_x_test)
df_x_test_numerical=np.array(df_x_test_numerical)
df_y_test=np.array(df_y_test)
df_y_test_numerical=np.array(df_y_test_numerical)


# Save the test arrays
np.save(os.path.join(output_path, 'x_test.npy'), df_x_test)
np.save(os.path.join(output_path, 'y_test.npy'), df_y_test)
np.save(os.path.join(output_path, 'x_test_numerical.npy'), df_x_test_numerical)
np.save(os.path.join(output_path, 'y_test_numerical.npy'), df_y_test_numerical)

print('x_test:', df_x_test.shape)
print('y_test:', df_y_test.shape)
print('x_test_numerical:', df_x_test_numerical.shape)
print('y_test_numerical:', df_y_test_numerical.shape)
