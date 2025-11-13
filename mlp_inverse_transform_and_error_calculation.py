import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# File paths
path = r"C:\Users\hp5cd\OneDrive\Desktop\Forecasts_Models\gold_usd_test.csv"
df_in = pd.read_csv(path, sep=',', decimal='.', usecols=[1])

path1 = r"C:\Users\hp5cd\OneDrive\Desktop\Forecasts_Models\frc_mlp_usd.csv" #before inverse transformation
y_hat = pd.read_csv(path1, sep=',', decimal='.', header=None, usecols=[0])

# Ensure y_hat is a numpy array for processing
y_hat = y_hat.values
# Ensure y_hat is a numpy array for processing
y_hat = y_hat.flatten()

# Initialize MinMaxScaler and process with sliding window
scaler = MinMaxScaler(feature_range=(0, 1))
num_windows = len(df_in) - 24  # Number of sliding windows of size 24

# Limit the iteration to the size of y_hat to avoid the IndexError
#num_windows = min(num_windows, len(y_hat))
inverse_transformed_y_hat = []

for i in range(num_windows):
    # Extract sliding window of size 24
    ts_in = np.array(df_in.iloc[i:i+24]).reshape(-1, 1)
    scaler.fit(ts_in)  # Fit scaler to the sliding window

    # Perform inverse transformation on corresponding y_hat value
    y_hat_transformed = scaler.inverse_transform(y_hat[i].reshape(-1, 1)).flatten()
    inverse_transformed_y_hat.append(y_hat_transformed)

# Convert the results to a numpy array
inverse_transformed_y_hat = np.array(inverse_transformed_y_hat)

# Save the resulting numpy array
np.save(r"C:\Users\hp5cd\OneDrive\Desktop\Forecasts_Models\forecast_mlp_usd_final.npy", inverse_transformed_y_hat)

# Save as CSV
save_forecasts = True
if save_forecasts:
    path = r"C:\Users\hp5cd\OneDrive\Desktop\Forecasts_Models\forecast_mlp_usd_final.csv"
    np.savetxt(path, inverse_transformed_y_hat, delimiter=',')

mlp_plt = pd.read_csv(path, header=None, usecols=[0])
#cnn_plt = pd.read_csv(r"C:\Users\hp5cd\chapter\forecast_cnn1d_sd_final.csv")
#forcnn_plt = pd.read_csv(r"C:\Users\hp5cd\chapter\forecast_forcnn_sd1.csv")  

  # Plot the data

plt.figure(figsize=(10, 6))
plt.plot(mlp_plt, label="mlp_usd", color='red')
#plt.plot(cnn_plt, label="cnn_1d", color='red')
#plt.plot(forcnn_plt, label="forcnn_plt", color='orange')
plt.plot(df_in, label="Actual", color='green')
# Add labels and title
plt.xlabel('Time in Hours')
plt.ylabel('Gold Price in USD')
plt.title('Actual Test Data vs Forecast by MLP')

# Add legend
plt.legend()

# Show the plot
plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def smape(y_actual, y_pred):
    return np.mean(200 * np.abs(y_pred - y_actual) / (np.abs(y_actual) + np.abs(y_pred)))


# Load data
# ErrorForcnn
y_test = np.load(r"C:\Users\hp5cd\OneDrive\Desktop\Forecasts_Models\y_test_numerical_usd.npy")  # Test actual data
print(y_test[1:5])
y_hat_test = np.load(r"C:\Users\hp5cd\OneDrive\Desktop\Forecasts_Models\forecast_mlp_usd_final.npy")  # Test predicted data

# Ensure both arrays have the same length by truncating the larger array
# min_len = min(len(y_test), len(y_hat_test))
# y_test = y_test[:min_len]
# y_hat_test = y_hat_test[:min_len]

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_hat_test)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate Normalized Root Mean Squared Error (NRMSE)
nrmse = rmse / (np.max(y_test) - np.min(y_test))

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_hat_test)

# Calculate SMAPE
smape_value = smape(y_test, y_hat_test)

# Print the results
print("MSE:", mse)
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("MAE:", mae)
print("SMAPE:", smape_value)
