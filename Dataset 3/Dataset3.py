import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.lines import Line2D
plt.rcParams['figure.dpi'] = 300 
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pylab import rcParams
from nptdms import TdmsFile
from matplotlib.pylab import rcParams
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn import cluster
from collections import Counter

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD

#from mlxtend.plotting import plot_decision_regions #用于画决策边界

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.svm import SVC, LinearSVC, NuSVC

from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, ADASYN
from imblearn.datasets import make_imbalance
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier

import time

import os, sys

if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('../')
    sys.path.insert(0, 'src/')

import numpy as np
import pandas as pd
import glob, re, pprint, random
from datetime import datetime
import pprint
import yaml

from scipy.stats import pearsonr, ttest_ind
from scipy.signal import savgol_filter
from scipy import interpolate

from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib import pyplot as plt
from tabpfn import TabPFNRegressor
import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import pandas as pd

# List of file paths
filepaths = ['ES10.mat', 'ES12.mat', 'ES14.mat']

all_results = []
all_labels = []
all_firsts = []

# Process each file
for filepath in filepaths:
    try:
        with h5py.File(filepath, 'r') as f:
            # Dynamically get the main group name (e.g., 'ES10', 'ES12', 'ES14')
            main_group_name = list(f.keys())[1]
            main_group = f[main_group_name]
            eis_data = main_group['EIS_Data']

            # Iterate through Cx groups (e.g., ES10C1, ES10C2, ..., ES14C8)
            for cx_index in range(1, 9):
                cx_name = f'{main_group_name}C{cx_index}'

                # Handle potential KeyError if a Cx group is missing
                if cx_name not in eis_data:
                    print(f"Warning: {cx_name} not found in {filepath}. Skipping.")
                    continue

                es10cx = eis_data[cx_name]
                eis_measurements = es10cx['EIS_Measurement']

                rows_to_extract = [1, 8, 68]
                cols_to_extract = [1, 2, 9]

                dataframes = {}

                for row_index, df_name in zip(rows_to_extract, ["0hrs", "125hrs", "3063hrs"]):
                    measurement_data_ref = eis_measurements['Data'][row_index, 0]
                    data_cells = f[measurement_data_ref]
                    first_cell_data_ref = data_cells[0, 0]
                    first_cell_data = f[first_cell_data_ref][:]

                    first_cell_data = first_cell_data.T

                    selected_data = first_cell_data[:, cols_to_extract]
                    df = pd.DataFrame(selected_data)
                    df.columns = ["Re(Z)", "-Im(Z)", "Cp"]
                    dataframes[df_name] = df

                re_z_diff = (dataframes["125hrs"]["-Im(Z)"] - dataframes["0hrs"]["-Im(Z)"]).values
                all_results.append(re_z_diff)
                
                first = dataframes["0hrs"]["Cp"].max()
                all_firsts.append(first)

                label = dataframes["3063hrs"]["Cp"].max() - dataframes["0hrs"]["Cp"].max()
                all_labels.append(label)

    except KeyError as e:
        print(f"Error: Key not found in the .mat file: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
    except OSError as e:
        print(f"Error reading HDF5 file. Possible file corruption or incorrect format: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Create the final DataFrame
final_df = pd.DataFrame(all_results)
final_df.columns = [f'Freq_{i+1}' for i in range(final_df.shape[1])]
final_df['label'] = all_labels
final_df['first'] = all_firsts
print("Successfully created the final DataFrame:")
print(f"\nShape of the final DataFrame: {final_df.shape}")

# Extract features (x_data) and target (y_data)
x_data = final_df.drop(['label', 'first'], axis=1).iloc[:, 8:]
y_data = final_df['label']

# Get total number of samples
total_samples = len(final_df)

# Create index lists for train/test sets
train_indices = []
test_indices = []

# For every 8 samples, take first 3 as test set and next 5 as training set
for i in range(0, total_samples, 8):  # Process in groups of 8 samples
    # First 3 samples in group go to test set
    test_indices.extend(range(i, min(i + 3, total_samples)))
    # Next 5 samples in group go to training set
    train_indices.extend(range(i + 3, min(i + 8, total_samples)))

# Split into train and test sets
train_input = x_data.iloc[train_indices].reset_index(drop=True)
test_input = x_data.iloc[test_indices].reset_index(drop=True)
train_output = y_data.iloc[train_indices].reset_index(drop=True)
test_output = y_data.iloc[test_indices].reset_index(drop=True)
feature_df = pd.DataFrame()
feature_cols = []

for i in range(x_data.shape[1]):
    col_name = f'im_{i}'
    feature_cols.append(col_name)


diff_feature_cols = []
num_cols = x_data.shape[1]
for i in range(num_cols):
    for j in range(i+1, num_cols):  
        col_name = f'im_{i}_minus_im_{j}'
        feature_df[col_name] = x_data.iloc[:, i] - x_data.iloc[:, j]
        diff_feature_cols.append(col_name)

sisso_features = feature_df[diff_feature_cols]  
sisso_target = pd.DataFrame(y_data.values, columns=['target'])
sisso_data = pd.concat([sisso_target, sisso_features], axis=1)

operators = ['+','-']

sisso_model = SissoModel(data=sisso_data, operators=operators, n_expansion = 2, use_gpu = True, k=10)
rmse, equation, r2, selected_features = sisso_model.fit()

im_41_minus_im_42 = x_data.iloc[:, 41] - x_data.iloc[:, 42]
im_42_minus_im_44 = x_data.iloc[:, 42] - x_data.iloc[:, 44]
im_7_minus_im_9 = x_data.iloc[:, 7] - x_data.iloc[:, 9]
im_14_minus_im_17 = x_data.iloc[:, 14] - x_data.iloc[:, 17]
im_1_minus_im_6 = x_data.iloc[:, 1] - x_data.iloc[:, 6]
im_3_minus_im_11 = x_data.iloc[:, 3] - x_data.iloc[:, 11]

im_6_minus_im_9 = x_data.iloc[:, 6] - x_data.iloc[:, 9]
im_3_minus_im_5 = x_data.iloc[:, 3] - x_data.iloc[:, 5]
im_7_minus_im_11 = x_data.iloc[:, 7] - x_data.iloc[:, 11]
im_37_minus_im_38 = x_data.iloc[:, 37] - x_data.iloc[:, 38]
im_40_minus_im_41 = x_data.iloc[:, 40] - x_data.iloc[:, 41]


term1 = -37193.8959810069*(im_6_minus_im_9-im_42_minus_im_44)
term2 = 74615.5713976286*(im_3_minus_im_5-im_7_minus_im_11)
term3 = -26559.1807914699*(im_37_minus_im_38-im_40_minus_im_41)
constant = 64.98637700790113
sisso_feature = term1 + term2 + term3 + constant
# Normalize SISSO feature
scaler = MinMaxScaler()
sisso_feature_normalized = scaler.fit_transform(sisso_feature.values.reshape(-1, 1))

# Split train and test sets
train_input = x_data.iloc[train_indices].reset_index(drop=True)
test_input = x_data.iloc[test_indices].reset_index(drop=True)
train_output = y_data.iloc[train_indices].reset_index(drop=True)
test_output = y_data.iloc[test_indices].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_input, test_input, train_output, test_output

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.ravel())

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train.ravel())

# Predict
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)


# Extract 'first' column values for test indices
first_values = final_df.iloc[test_indices]['first'].values

# Ensure the length of first_values matches the test set
assert len(first_values) == len(test_indices), "Length of first_values does not match test set size"

# Add 'first' column values to predictions
y_pred_rf_adjusted = y_pred_rf + first_values
y_pred_xgb_adjusted = y_pred_xgb + first_values
test_output = test_output+ first_values
# Calculate performance metrics
def calculate_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    return mae, mape, rmse, r2

# Evaluate adjusted predictions
mae_rf, mape_rf, rmse_rf, r2_rf = calculate_metrics(test_output, y_pred_rf_adjusted)
mae_xgb, mape_xgb, rmse_xgb, r2_xgb = calculate_metrics(test_output, y_pred_xgb_adjusted)

# Print results
print("\nRandom Forest Model Performance (Adjusted):")
print(f"MAE: {mae_rf:.4f}")
print(f"MAPE: {mape_rf:.4%}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R2: {r2_rf:.4f}")

print("\nXGBoost Model Performance (Adjusted):")
print(f"MAE: {mae_xgb:.4f}")
print(f"MAPE: {mape_xgb:.4%}")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"R2: {r2_xgb:.4f}")