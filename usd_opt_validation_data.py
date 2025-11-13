import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Paths to your train and test sets
train_path = "/gpfs-home/p220127ma/Meta_Data/gold_usd_train.csv"
test_path = "/gpfs-home/p220127ma/Meta_Data/gold_usd_test.csv"

# Load the train and test sets using pandas
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Remove non-numerical columns (e.g., timestamps) if present
train_data = train_data.select_dtypes(include=[np.number])
test_data = test_data.select_dtypes(include=[np.number])

# Convert to numpy arrays
train_data = train_data.values
test_data = test_data.values

# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define the sequence length
seq_length = 24

# Function to create sequences for x and y
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length, -1])  
    return np.array(x), np.array(y)

# Create x_train and y_train from the train data
x_train, y_train = create_sequences(train_data, seq_length)

# Create x_test and y_test from the test data
x_test, y_test = create_sequences(test_data, seq_length)

# Save y_test (numerical values) as a .npy file
directory = '/gpfs-home/p220127ma/Meta_Data/'

# Check if the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

y_test_numerical_path = os.path.join(directory, 'y_test_numerical.npy')
np.save(y_test_numerical_path, y_test)

# Now split the training data into a new training set and a validation set
val_size = 5820  # Validation set size

# Randomly select indices for validation set
np.random.seed(42)  # Optional: for reproducibility
indices = np.random.choice(len(x_train), size=val_size, replace=False)

# Create validation set
x_vali = x_train[indices]
y_vali = y_train[indices]

# Remove the selected data points from the training set
x_train_new = np.delete(x_train, indices, axis=0)
y_train_new = np.delete(y_train, indices, axis=0)

# Define file paths for saving the datasets in .npy format
x_vali_path = os.path.join(directory, 'x_vali.npy')
y_vali_path = os.path.join(directory, 'y_vali.npy')
x_train_new_path = os.path.join(directory, 'x_train_new.npy')
y_train_new_path = os.path.join(directory, 'y_train_new.npy')

# Save the validation and new training datasets as .npy
np.save(x_vali_path, x_vali)
np.save(y_vali_path, y_vali)
np.save(x_train_new_path, x_train_new)
np.save(y_train_new_path, y_train_new)

# Optional: Save the full x_train, y_train, x_test, y_test sets in .npy format
x_train_full_path = os.path.join(directory, 'x_train_full.npy')
y_train_full_path = os.path.join(directory, 'y_train_full.npy')
x_test_full_path = os.path.join(directory, 'x_test_full.npy')
y_test_full_path = os.path.join(directory, 'y_test_full.npy')

np.save(x_train_full_path, x_train)
np.save(y_train_full_path, y_train)
np.save(x_test_full_path, x_test)
np.save(y_test_full_path, y_test)

print(f"Datasets saved to {directory}")
