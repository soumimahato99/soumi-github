import numpy as np
import os

# Provide the full path to your x_train and y_train files
x_train_path = "/gpfs-home/p220127ma/Images_USD/USD/x_train.npy"
y_train_path= "/gpfs-home/p220127ma/Images_USD/USD/y_train.npy"

# Load your x_train and y_train datasets using the full path
x_train= np.load(x_train_path)  # Load x_train
y_train = np.load(y_train_path)  # Load y_train

# Step 1: Randomly select 3000 unique indices from x_train for validation set
np.random.seed(42)  # Optional: Set seed for reproducibility
indices = np.random.choice(len(x_train), size=4719, replace=False)

# Step 2: Create the validation set using the selected indices
x_vali = x_train[indices]
y_vali = y_train[indices]

# Step 3: Remove the selected data points from the training set
x_train_new = np.delete(x_train, indices, axis=0)
y_train_new = np.delete(y_train, indices, axis=0)

# Step 4: Define directory and file paths where you want to save the new datasets
directory = '/gpfs-home/p220127ma/Images_USD/USD'

# Check if the directory exists; if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Define file paths for saving the datasets
x_vali_path = os.path.join(directory, 'x_vali.npy')
y_vali_path = os.path.join(directory, 'y_vali.npy')
x_train_new_path = os.path.join(directory, 'x_train_new.npy')
y_train_new_path = os.path.join(directory, 'y_train_new.npy')

# Step 5: Save the new validation and training datasets
np.save(x_vali_path, x_vali)
np.save(y_vali_path, y_vali)
np.save(x_train_new_path, x_train_new)
np.save(y_train_new_path, y_train_new)

print(f"Validation and training datasets saved to {directory}")
