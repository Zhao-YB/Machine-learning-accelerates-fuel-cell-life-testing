import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
%matplotlib inline
plt.rcParams['figure.dpi'] = 300 
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.pylab import rcParams
from scipy.stats import pearsonr, spearmanr
import scipy.io as sio
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
from sklearn import cluster
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy.spatial.distance import euclidean
from collections import Counter

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
np.random.seed(42)

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.cluster import KMeans
import time
import h5py

from math import sqrt, log
import numpy as np
from numpy.linalg import norm
from numpy import inf, sin, cos, cosh, pi, exp, log10
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import time
from tabpfn import TabPFNRegressor
from TorchSisso import SissoModel

df = pd.read_csv('pemwe_single_cell_ast_1402h.csv', sep=';')
df['Re(Z)/Ohm'] = df['Re(Z)/Ohm'] * 1000
df['-Im(Z)/Ohm'] = df['-Im(Z)/Ohm'] * 1000
filtered_df = df[(df['freq/Hz'] < 101.97982) & (df['freq/Hz'] > 0)]

groups = [filtered_df.iloc[i:i+33] for i in range(0, len(filtered_df), 33)]
assert len(groups) == 330, "330"

# 选择特定组
selected_indices = [0, 6, 18, 30] + [30 + 12 * k for k in range(25)]
selected_groups = [groups[i] for i in selected_indices[:-1]]
assert len(selected_groups) == 28, "28"

new_groups = []
for group in selected_groups:
    start_idx = group.index[0]
    if start_idx - 2052 - 22 >= 0:
        part1 = df.iloc[start_idx-2052-22:start_idx-22]  
        part2 = df.iloc[start_idx-22:start_idx]  
        part3 = group  
        new_group = pd.concat([part1, part2, part3])
        assert len(new_group) == 2107, "2107"
        new_groups.append(new_group)
    else:
        print(f" {start_idx} ")

final_df = pd.concat(new_groups, ignore_index=True)

filtered_df = df[(df['freq/Hz'] < 101.97982) & (df['freq/Hz'] > 0)]

groups = [filtered_df.iloc[i:i+33] for i in range(0, len(filtered_df), 33)]
assert len(groups) == 330, "330"

selected_indices = [0, 6, 18, 30] + [30 + 12 * k for k in range(25)]
selected_groups = [groups[i] for i in selected_indices[:-1]]
assert len(selected_groups) == 28, "28"

new_groups = []
for group in selected_groups:
    start_idx = group.index[0]
    if start_idx - 22 >= 0:
        part1 = df.iloc[start_idx-22:start_idx-22]  
        part2 = df.iloc[start_idx-22:start_idx]  
        part3 = group  
        new_group = pd.concat([part1, part2, part3])
        assert len(new_group) == 55, "2107"
        new_groups.append(new_group)
    else:
        print(f" {start_idx} 2052+22")

final_df = pd.concat(new_groups, ignore_index=True)
group_size = 55 

mid_freq_end = 44

key_freqs = (39.138828, 6530.0005)


all_input_data_key = []
all_output_data_high = []
all_input_data_high_to_mid = []
all_output_data_mid = []

for i in range(len(final_df) // group_size):
    df_subset = final_df.iloc[i * group_size:(i + 1) * group_size].head(mid_freq_end).values
    max_Z_Re = df_subset[:, 4].max()  
    for freq in key_freqs:
        dff = df_subset[df_subset[:, 3] == freq]
        if len(dff) > 0:
            input_key.append(dff[0, 4:6])
    input_key = np.array(input_key)

    output_high = df_subset[0:mid_freq_end, 4:6] 

    all_input_data_key.append(input_key)
    all_output_data_high.append(output_high)
    
all_input_data_key = np.array(all_input_data_key)
all_output_data_high = np.array(all_output_data_high)

n_groups = 28

# Define the indices for the test set (4th, 8th, 12th, 16th, 20th, 24th, 28th in 1-based indexing)
test_indices = [3, 7, 11, 15, 19, 23, 27]  # Convert to 0-based indexing
train_indices = [i for i in range(n_groups) if i not in test_indices]  # All other indices for training

# Split the data based on the indices
train_input_key = all_input_data_key[train_indices]
test_input_key = all_input_data_key[test_indices]
train_output_high = all_output_data_high[train_indices]
test_output_high = all_output_data_high[test_indices]

# Proceed with standardization
scaler_key = StandardScaler()
scaler_high = StandardScaler()

# Standardize training data
train_input_key_scaled = scaler_key.fit_transform(train_input_key.reshape(train_input_key.shape[0], -1))
train_output_high_scaled = scaler_high.fit_transform(train_output_high.reshape(train_output_high.shape[0], -1))

# Standardize test data (using the same scalers)
test_input_key_scaled = scaler_key.transform(test_input_key.reshape(test_input_key.shape[0], -1))

# Train the RandomForestRegressor model
model_key_to_high_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_key_to_high_rf.fit(train_input_key_scaled, train_output_high_scaled)
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
all_predicted_curves = []
all_original_curves = []

for curve_index in range(len(test_input_key)):
    key_input_scaled = scaler_key.transform(test_input_key[curve_index].reshape(1, -1))
    key_input = test_input_key[curve_index].reshape(1, -1)

    predicted_high_scaled = model_key_to_high_rf.predict(key_input_scaled)
    predicted_high = scaler_high.inverse_transform(predicted_high_scaled).reshape(mid_freq_end, 2)

    predicted_curve = np.vstack((predicted_high))
    original_curve = np.vstack((test_output_high[curve_index]))

    all_predicted_curves.append(predicted_curve)
    all_original_curves.append(original_curve)

rmse_values = []
mae_values = []
mape_values = []
r2_values = []

for i in range(len(all_original_curves)):
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_original_curves[i], all_predicted_curves[i]))
    mae = mean_absolute_error(all_original_curves[i], all_predicted_curves[i])
    mape = mean_absolute_percentage_error(all_original_curves[i], all_predicted_curves[i]) * 100
    r2 = r2_score(all_original_curves[i], all_predicted_curves[i])
    
    # Store metrics
    rmse_values.append(rmse)
    mae_values.append(mae)
    mape_values.append(mape)
    r2_values.append(r2)

# Compute average metrics
avg_rmse = np.mean(rmse_values)
avg_mae = np.mean(mae_values)
avg_mape = np.mean(mape_values)
avg_r2 = np.mean(r2_values)

# Print results
print(f"Average RMSE on test set (RF): {avg_rmse:.4f}")
print(f"Average MAE on test set (RF): {avg_mae:.4f}")
print(f"Average MAPE on test set (RF): {avg_mape:.4f}%")
print(f"Average R? on test set (RF): {avg_r2:.4f}")

filtered_df = df[(df['freq/Hz'] < 101.97982) & (df['freq/Hz'] > 0)]

groups = [filtered_df.iloc[i:i+33] for i in range(0, len(filtered_df), 33)]
assert len(groups) == 330, "330"

selected_indices = [0, 6, 18, 30] + [30 + 12 * k for k in range(25)]
selected_groups = [groups[i] for i in selected_indices[:-1]]
assert len(selected_groups) == 28, "28"

new_groups = []
for group in selected_groups:
    start_idx = group.index[0]
    if start_idx - 22 >= 0:
        part1 = df.iloc[start_idx-22:start_idx-22]  
        part2 = df.iloc[start_idx-22:start_idx]  
        part3 = group  
        new_group = pd.concat([part1, part2, part3])
        assert len(new_group) == 55, "55"
        new_groups.append(new_group)
    else:
        print(f" {start_idx} 2052+22")

final_df = pd.concat(new_groups)

# Step 1: Divide final_df into 28 groups, each with 55 rows
groups = [final_df.iloc[i:i+55] for i in range(0, len(final_df), 55)]
assert len(groups) == 28, "There should be 28 groups, each with 55 rows"

# Step 2: Define 38 target i/A/cm^2 values
target_values = [3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 
                 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.09, 0.08, 0.07, 
                 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

# Step 3: Process each group
new_groups = []
for group in groups:
    start_idx = group.index[0]  # First row index of the group
    selected_rows = []
    last_idx = start_idx  # Record the index of last found row, initialized as group start index

    # Step 4: Search upward for each target value in sequence
    for target in target_values:
        # Search upward from last_idx, maximum 500 rows
        search_start = max(0, last_idx - 500)  # Limit search range to 500 rows
        search_df = df.iloc[search_start:last_idx] if last_idx > 0 else pd.DataFrame()
        
        if not search_df.empty:
            # Filter rows where freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm are all 0
            valid_rows = search_df[
                (search_df['freq/Hz'] == 0) & 
                (search_df['Re(Z)/Ohm'] == 0) & 
                (search_df['-Im(Z)/Ohm'] == 0)
            ]
            
            if not valid_rows.empty:
                # Find the row with i/A/cm^2 closest to target value
                valid_rows['diff'] = np.abs(valid_rows['i/A/cm^2'] - target)
                closest_row = valid_rows.sort_values('diff').head(1)  # Take the closest row
                selected_rows.append(closest_row.drop(columns=['diff']))
                last_idx = closest_row.index[0]  # Update last_idx to current found row index
            else:
                print(f"Group starting at index {start_idx}: No valid rows found for target value {target} in range {search_start}:{last_idx}")
                break  # No valid rows found, stop searching for this group
        else:
            print(f"Group starting at index {start_idx}: Insufficient search range for target value {target} in range {search_start}:{last_idx}")
            break  # Insufficient search range, stop searching for this group

        # If all 38 target values are found, immediately move to next group
        if len(selected_rows) == len(target_values):
            break

    # Step 5: Check if 38 rows are found, merge 38 rows with original 55 rows
    if len(selected_rows) == len(target_values):
        selected_df = pd.concat(selected_rows)
        assert len(selected_df) == 38, f"Group starting at index {start_idx} should have 38 rows, actually got {len(selected_df)} rows"
        new_group = pd.concat([selected_df, group])
        assert len(new_group) == 93, f"Group starting at index {start_idx} should have 93 rows in new group, actually got {len(new_group)} rows"
        new_groups.append(new_group)
    else:
        print(f"Group starting at index {start_idx} did not find all 38 target values, only found {len(selected_rows)}")

# Step 6: Combine all new groups into final DataFrame
if new_groups:
    final_iv_df = pd.concat(new_groups)
    print(f"Final DataFrame created successfully, containing {len(final_iv_df)} rows")
else:
    print("No groups were processed successfully")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


# Step 1: Set parameters
group_size = 93  # 93 rows per group
mid_freq_end = 44  # First 44 rows of the last 55 rows as input
n_groups = 28  # Total 28 groups

# Step 2: Split final_iv_df into 28 groups with 93 rows each
groups = [final_iv_df.iloc[i:i+group_size] for i in range(0, len(final_iv_df), group_size)]
assert len(groups) == n_groups, f"There should be {n_groups} groups, each with {group_size} rows"

# Step 3: Extract input and output data
all_input_data = []
all_output_data = []

for i, group in enumerate(groups):
    # Input: Last two columns of first 44 rows from last 55 rows (indices 38:82, last two columns)
    df_subset = group.iloc[38:38+mid_freq_end]  # First 44 rows of last 55 rows (indices 38:82)
    input_data = df_subset.iloc[:, -2:].values  # Last two columns, assumed to be Re(Z)/Ohm and -Im(Z)/Ohm
    # Output: Second column of first 38 rows (index 1, assumed to be Ecell/V)
    output_data = group.iloc[:38, 1].values  # Second column
    all_input_data.append(input_data)  # Shape: (44, 2)
    all_output_data.append(output_data)  # Shape: (38,)

# Convert to numpy arrays
all_input_data = np.array(all_input_data)  # Shape: (28, 44, 2)
all_output_data = np.array(all_output_data)  # Shape: (28, 38)

# Step 4: Data splitting
# Test set indices (1-based: 4, 8, 12, 16, 20, 24, 28 converted to 0-based)
test_indices = [3, 7, 11, 15, 19, 23, 27]
train_indices = [i for i in range(n_groups) if i not in test_indices]

# Split data based on indices
train_input = all_input_data[train_indices]  # Shape: (21, 44, 2)
test_input = all_input_data[test_indices]  # Shape: (7, 44, 2)
train_output = all_output_data[train_indices]  # Shape: (21, 38)
test_output = all_output_data[test_indices]  # Shape: (7, 38)

# Step 5: Standardization
scaler_input = StandardScaler()
scaler_output = StandardScaler()

# Standardize training data
train_input_scaled = scaler_input.fit_transform(
    train_input.reshape(train_input.shape[0], -1)
)  # Flatten to (21, 44*2)
train_output_scaled = scaler_output.fit_transform(
    train_output.reshape(train_output.shape[0], -1)
)  # Flatten to (21, 38)

# Standardize test data (using same scaler)
test_input_scaled = scaler_input.transform(
    test_input.reshape(test_input.shape[0], -1)
)  # Flatten to (7, 44*2)

# Step 6: Train Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_input_scaled, train_output_scaled)

# Step 7: Prediction and evaluation
test_output_pred_scaled = model.predict(test_input_scaled)
# Inverse transform predictions
test_output_pred = scaler_output.inverse_transform(test_output_pred_scaled)
test_output_true = test_output  # Confirm test_output is unstandardized original values

# Calculate evaluation metrics
mse = mean_squared_error(test_output_true, test_output_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_output_true, test_output_pred)
mape = mean_absolute_percentage_error(test_output_true, test_output_pred) * 100
r2 = r2_score(test_output_true, test_output_pred)

print(f"Test set Mean Squared Error (MSE): {mse:.4f}")
print(f"Test set Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Test set Mean Absolute Error (MAE): {mae:.4f}")
print(f"Test set Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"Test set R? score: {r2:.4f}")

test_input = np.array(all_predicted_curves)
test_input_scaled = scaler_input.transform(
    test_input.reshape(test_input.shape[0], -1))  

test_output_pred_scaled = model.predict(test_input_scaled)

test_output_pred_pred = scaler_output.inverse_transform(test_output_pred_scaled)
test_output_true = test_output  

mse = mean_squared_error(test_output_true, test_output_pred_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_output_true, test_output_pred_pred)
mape = mean_absolute_percentage_error(test_output_true, test_output_pred_pred) * 100
r2 = r2_score(test_output_true, test_output_pred_pred)

print(f"Test set Mean Squared Error (MSE): {mse:.4f}")
print(f"Test set Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Test set Mean Absolute Error (MAE): {mae:.4f}")
print(f"Test set Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"Test set R? score: {r2:.4f}")