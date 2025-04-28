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
from TorchSisso import SissoModel
from tabpfn import TabPFNRegressor

import pandas as pd
import os

base_path = './01-Data'
cycles = ['0cycles', '1000cycles', '5000cycles', '10000cycles', '20000cycles', '30000cycles']
all_data = []
expected_rows_41 = 41
expected_rows_81 = 81

for cell_num in range(1, 43):
    cell_folder = os.path.join(base_path, f'Cell_{cell_num:02}')
    eis_folder = os.path.join(cell_folder, '04-EIS_H2Air_RH100', '1000mAcm2')

    if os.path.exists(eis_folder):
        for cycle in cycles:
            file_path = os.path.join(eis_folder, cycle, 'EIS_data_raw.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    num_rows = len(df)
                    if num_rows == expected_rows_41:
                        df.columns = ['Frequency(Hz)', 'Z_Re(mOhm*cm2)', 'Z_Im(mOhm*cm2)']
                        df['Frequency(Hz)'] = pd.to_numeric(df['Frequency(Hz)'], errors='coerce')
                        df.sort_values(by='Frequency(Hz)', ascending=False, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        df['Cell'] = f'Cell_{cell_num:02}'
                        df['Cycle'] = cycle.replace('cycles', '')
                        all_data.append(df)
                    elif num_rows == expected_rows_81:
                        df.columns = ['Frequency(Hz)', 'Z_Re(mOhm*cm2)', 'Z_Im(mOhm*cm2)']
                        df['Frequency(Hz)'] = pd.to_numeric(df['Frequency(Hz)'], errors='coerce')
                        freq_41 = pd.read_csv(file_path, sep='\t', nrows=expected_rows_41)['f.C[Hz]'].astype(float).tolist()
                        df = df[df['Frequency(Hz)'].isin(freq_41)]
                        df.sort_values(by='Frequency(Hz)', ascending=False, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        df['Cell'] = f'Cell_{cell_num:02}'
                        df['Cycle'] = cycle.replace('cycles', '')
                        all_data.append(df)
                    else:
                        print(f"Warning: {file_path} has {num_rows} rows, expected {expected_rows_41} or {expected_rows_81}. Skipping.")
                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty. Skipping.")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"Warning: {file_path} does not exist. Skipping.")
    else:
        print(f"Warning: {eis_folder} does not exist. Skipping.")

combined_df = pd.concat(all_data, ignore_index=True) if all_data else None


base_path = './01-Data'
cycles = ['0cycles', '1000cycles', '5000cycles', '10000cycles', '30000cycles']
all_iv_data = []
interpolation_points = np.linspace(0, 3.1, 20)  

for cell_num in range(1, 43):
    cell_folder = os.path.join(base_path, f'Cell_{cell_num:02}')
    iv_folder = os.path.join(cell_folder, '03-IV-curve_RH100')

    if os.path.exists(iv_folder):
        for cycle in cycles:
            file_path = os.path.join(iv_folder, cycle, 'IV_data.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    df.columns = ['UC', 'IC'] 

                    df['IC'] = pd.to_numeric(df['IC'], errors='coerce')
                    df['UC'] = pd.to_numeric(df['UC'], errors='coerce')
                    
                    df.dropna(inplace=True)
                    
                    f = interp1d(df['IC'], df['UC'], kind='linear', fill_value="extrapolate") 
                    interpolated_u = f(interpolation_points)
                    interpolated_df = pd.DataFrame({'IC': interpolation_points, 'UC': interpolated_u})
                    interpolated_df['Cell'] = f'Cell_{cell_num:02}'
                    interpolated_df['Cycle'] = cycle.replace('cycles', '')
                    all_iv_data.append(interpolated_df)

                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty. Skipping.")
                except ValueError as e:
                    print(f"ValueError reading {file_path}: {str(e)}. Check data format and try again.")
                    print(f"DataFrame causing the error:\n{df}")  
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    print(f"DataFrame causing the error:\n{df}") 
            else:
                print(f"Warning: {file_path} does not exist. Skipping.")
    else:
        print(f"Warning: {iv_folder} does not exist. Skipping.")

combined_iv_df = pd.concat(all_iv_data, ignore_index=True) if all_iv_data else None


base_path = './01-Data'
cycles = ['0cycles', '1000cycles', '5000cycles', '10000cycles', '20000cycles', '30000cycles']
all_cv_data = []
interpolation_points = np.linspace(0.051, 0.40, 100) 

for cell_num in range(1, 43):
    cell_folder = os.path.join(base_path, f'Cell_{cell_num:02}')
    iv_folder = os.path.join(cell_folder, '10-CV')

    if os.path.exists(iv_folder):
        for cycle in cycles:
            file_path = os.path.join(iv_folder, cycle, 'CV_data_raw.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    df.columns = ['Time', 'UC', 'IC'] 
                    condition = df['UC'] <= 0.4

                    start_index = -1
                    end_index = -1

                    for i in range(len(df)):
                        if condition[i]:
                            if start_index == -1:
                                start_index = i
                            end_index = i
                        elif start_index != -1:
                            break 

                    if start_index != -1:
                        df = df.iloc[start_index:end_index+1].copy()
                    else:
                        print("Falut")

                    df['IC'] = pd.to_numeric(df['IC'], errors='coerce')
                    df['UC'] = pd.to_numeric(df['UC'], errors='coerce')
                    
                    df.dropna(inplace=True)
                    
                    f = interp1d(df['UC'],df['IC'], kind='linear', fill_value="extrapolate") # extrapolate参数外推
                    interpolated_u = f(interpolation_points)
                    interpolated_df = pd.DataFrame({'UC': interpolation_points, 'IC': interpolated_u})
                    interpolated_df['Cell'] = f'Cell_{cell_num:02}'
                    interpolated_df['Cycle'] = cycle.replace('cycles', '')
                    all_cv_data.append(interpolated_df)

                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty. Skipping.")
                except ValueError as e:
                    print(f"ValueError reading {file_path}: {str(e)}. Check data format and try again.")
                    print(f"DataFrame causing the error:\n{df}")  
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    print(f"DataFrame causing the error:\n{df}") 
            else:
                print(f"Warning: {file_path} does not exist. Skipping.")
    else:
        print(f"Warning: {iv_folder} does not exist. Skipping.")

combined_cv_df = pd.concat(all_cv_data, ignore_index=True) if all_cv_data else None

base_path = './01-Data'
cycles = ['0cycles', '30000cycles']
all_ilim_data = []

for cell_num in range(1, 43):
    cell_folder = os.path.join(base_path, f'Cell_{cell_num:02}')
    iv_folder = os.path.join(cell_folder, '02-Limiting_Current_Measurements')

    if os.path.exists(iv_folder):
        for cycle in cycles:
            file_path = os.path.join(iv_folder, cycle, 'iLim_data.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    df.columns = ['c_mean', 'iLim', 'p[bar]'] 
                    
                    df.dropna(inplace=True)
                    iLim = df[(df['c_mean']< 1.1) & (df['c_mean']>= 1) & (df['p[bar]']== 2.0)]['iLim'].values
                    interpolated_df = pd.DataFrame({'iLim': iLim})
                    interpolated_df['Cell'] = f'Cell_{cell_num:02}'
                    interpolated_df['Cycle'] = cycle.replace('cycles', '')
                    all_ilim_data.append(interpolated_df)

                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty. Skipping.")
                except ValueError as e:
                    print(f"ValueError reading {file_path}: {str(e)}. Check data format and try again.")
                    print(f"DataFrame causing the error:\n{df}")  
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    print(f"DataFrame causing the error:\n{df}") 
            else:
                print(f"Warning: {file_path} does not exist. Skipping.")
    else:
        print(f"Warning: {iv_folder} does not exist. Skipping.")

combined_ilim_df = pd.concat(all_ilim_data, ignore_index=True) if all_ilim_data else None

base_path = './01-Data'
cycles = ['0cycles', '30000cycles']
all_rt_data = []

for cell_num in range(1, 43):
    cell_folder = os.path.join(base_path, f'Cell_{cell_num:02}')
    iv_folder = os.path.join(cell_folder, '02-Limiting_Current_Measurements')

    if os.path.exists(iv_folder):
        for cycle in cycles:
            file_path = os.path.join(iv_folder, cycle, 'R_total_p.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, sep='\t')

                    df.dropna(inplace=True)
                    RT = df[(df['p[bar]']< 2.1) & (df['p[bar]']>= 1.9)]['R_T[s/m]'].values
                    interpolated_df = pd.DataFrame({'RT': RT})
                    interpolated_df['Cell'] = f'Cell_{cell_num:02}'
                    interpolated_df['Cycle'] = cycle.replace('cycles', '')
                    all_rt_data.append(interpolated_df)

                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty. Skipping.")
                except ValueError as e:
                    print(f"ValueError reading {file_path}: {str(e)}. Check data format and try again.")
                    print(f"DataFrame causing the error:\n{df}") 
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    print(f"DataFrame causing the error:\n{df}") 
            else:
                print(f"Warning: {file_path} does not exist. Skipping.")
    else:
        print(f"Warning: {iv_folder} does not exist. Skipping.")

combined_rt_df = pd.concat(all_rt_data, ignore_index=True) if all_rt_data else None

base_path = './01-Data'
cycles = ['0cycles', '10cycles', '100cycles', '1000cycles','3000cycles',  '5000cycles', '10000cycles', '20000cycles', '30000cycles']
all_ecsa_data = []

for cell_num in range(1, 43):
    cell_folder = os.path.join(base_path, f'Cell_{cell_num:02}')
    iv_folder = os.path.join(cell_folder, '10-CV')

    if os.path.exists(iv_folder):
        for cycle in cycles:
            file_path = os.path.join(iv_folder, cycle, 'ECSA_mean.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, sep='\t')

                    df.dropna(inplace=True)
                    ECSA = df['ECSA_90mV_mean [cm2/cm2]'].values
                    interpolated_df = pd.DataFrame({'ECSA': ECSA})
                    interpolated_df['Cell'] = f'Cell_{cell_num:02}'
                    interpolated_df['Cycle'] = cycle.replace('cycles', '')
                    all_ecsa_data.append(interpolated_df)

                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty. Skipping.")
                except ValueError as e:
                    print(f"ValueError reading {file_path}: {str(e)}. Check data format and try again.")
                    print(f"DataFrame causing the error:\n{df}")  
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    print(f"DataFrame causing the error:\n{df}") 
            else:
                print(f"Warning: {file_path} does not exist. Skipping.")
    else:
        print(f"Warning: {iv_folder} does not exist. Skipping.")

combined_ecsa_df = pd.concat(all_ecsa_data, ignore_index=True) if all_ecsa_data else None


eis_array = []
rt_array = []
rt_first_array = []  
eis_rows = 41
rt_rows = 1
cycles = ['0', '30000']

for cell_num in range(1, 43):  
    cell_id = f'Cell_{cell_num:02}'
    for cycle in cycles:
        eis_data = combined_df[(combined_df['Cell'] == cell_id) & (combined_df['Cycle'] == cycle)]
        rt_data = combined_rt_df[(combined_rt_df['Cell'] == cell_id) & (combined_rt_df['Cycle'] == cycle)]

        if len(eis_data) == eis_rows and len(rt_data) == rt_rows:
            if cycle == '0':
                rt_first_array.append(combined_rt_df[(combined_rt_df['Cell'] == cell_id) & (combined_rt_df['Cycle'] == '0')][['RT']].values)
            if cycle == '30000':
                rt_array.append(combined_rt_df[(combined_rt_df['Cell'] == cell_id) & (combined_rt_df['Cycle'] == '30000')][['RT']].values
                                - combined_rt_df[(combined_rt_df['Cell'] == cell_id) & (combined_rt_df['Cycle'] == '0')][['RT']].values)
        elif len(eis_data) != eis_rows:
            print(f"Warning: EIS data for {cell_id}, {cycle} has incorrect length ({len(eis_data)}). Skipping.")
        elif len(rt_data) != rt_rows:
            print(f"Warning: CV data for {cell_id}, {cycle} has incorrect length ({len(rt_data)}). Skipping.")

rt_array = np.array(rt_array)
rt_array = np.squeeze(rt_array)  
rt_first_array = np.array(rt_first_array)
rt_first_array = np.squeeze(rt_first_array)  

print("RT data shape (difference):", rt_array.shape)
print("RT first data shape (Cycle 0):", rt_first_array.shape)

eis_array = []
ilim_array = []
ilim_first_array = []  # New array for iLim values at Cycle == '0'
eis_rows = 41
ilim_rows = 1
cycles = ['0', '30000']

for cell_num in range(1, 43):  
    cell_id = f'Cell_{cell_num:02}'
    for cycle in cycles:
        eis_data = combined_df[(combined_df['Cell'] == cell_id) & (combined_df['Cycle'] == cycle)]
        ilim_data = combined_ilim_df[(combined_ilim_df['Cell'] == cell_id) & (combined_ilim_df['Cycle'] == cycle)]

        if len(eis_data) == eis_rows and len(ilim_data) == ilim_rows:
            if cycle == '0':
                # Store iLim value for Cycle == '0'
                ilim_first_array.append(combined_ilim_df[(combined_ilim_df['Cell'] == cell_id) & (combined_ilim_df['Cycle'] == '0')][['iLim']].values)
            if cycle == '30000':
                # Store difference in iLim values (Cycle 30000 - Cycle 0)
                ilim_array.append(combined_ilim_df[(combined_ilim_df['Cell'] == cell_id) & (combined_ilim_df['Cycle'] == '30000')][['iLim']].values
                                 - combined_ilim_df[(combined_ilim_df['Cell'] == cell_id) & (combined_ilim_df['Cycle'] == '0')][['iLim']].values)
        elif len(eis_data) != eis_rows:
            print(f"Warning: EIS data for {cell_id}, {cycle} has incorrect length ({len(eis_data)}). Skipping.")
        elif len(ilim_data) != ilim_rows:
            print(f"Warning: CV data for {cell_id}, {cycle} has incorrect length ({len(ilim_data)}). Skipping.")

# Convert lists to NumPy arrays
ilim_array = np.array(ilim_array)
ilim_array = np.squeeze(ilim_array)  # Remove single-dimensional entries
ilim_first_array = np.array(ilim_first_array)
ilim_first_array = np.squeeze(ilim_first_array)  # Remove single-dimensional entries

# Print shapes for verification
print("iLim data shape (difference):", ilim_array.shape)
print("iLim first data shape (Cycle 0):", ilim_first_array.shape)

eis_array = []
ecsa_array = []
ecsa_first_array = []  # New array for ECSA values at Cycle == '0'
eis_rows = 41
ecsa_rows = 1
cycles = ['0', '30000']

for cell_num in range(1, 43):  
    cell_id = f'Cell_{cell_num:02}'
    for cycle in cycles:
        eis_data = combined_df[(combined_df['Cell'] == cell_id) & (combined_df['Cycle'] == cycle)]
        ecsa_data = combined_ecsa_df[(combined_ecsa_df['Cell'] == cell_id) & (combined_ecsa_df['Cycle'] == cycle)]

        if len(eis_data) == eis_rows and len(ecsa_data) == ecsa_rows:
            if cycle == '0':
                # Store ECSA value for Cycle == '0'
                ecsa_first_array.append(combined_ecsa_df[(combined_ecsa_df['Cell'] == cell_id) & (combined_ecsa_df['Cycle'] == '0')][['ECSA']].values)
            if cycle == '30000':
                # Store difference in ECSA values (Cycle 30000 - Cycle 0)
                ecsa_array.append(combined_ecsa_df[(combined_ecsa_df['Cell'] == cell_id) & (combined_ecsa_df['Cycle'] == '30000')][['ECSA']].values
                                 - combined_ecsa_df[(combined_ecsa_df['Cell'] == cell_id) & (combined_ecsa_df['Cycle'] == '0')][['ECSA']].values)
        elif len(eis_data) != eis_rows:
            print(f"Warning: EIS data for {cell_id}, {cycle} has incorrect length ({len(eis_data)}). Skipping.")
        elif len(ecsa_data) != ecsa_rows:
            print(f"Warning: CV data for {cell_id}, {cycle} has incorrect length ({len(ecsa_data)}). Skipping.")

# Convert lists to NumPy arrays
ecsa_array = np.array(ecsa_array)
ecsa_array = np.squeeze(ecsa_array)  # Remove single-dimensional entries
ecsa_first_array = np.array(ecsa_first_array)
ecsa_first_array = np.squeeze(ecsa_first_array)  # Remove single-dimensional entries

# Print shapes for verification
print("ECSA data shape (difference):", ecsa_array.shape)
print("ECSA first data shape (Cycle 0):", ecsa_first_array.shape)

#Taking ECSA prediction as an example
cv_result = []
cell_ids = []  
for cell in combined_cv_df[combined_cv_df.Cell != 'Cell_06'].Cell.unique():
    cell_number = int(cell.split('_')[1])
    Cv0 = combined_cv_df[(combined_cv_df['Cell'] == cell) & (combined_cv_df['Cycle'] == '0')]['IC'].values
    Cv1 = combined_cv_df[(combined_cv_df['Cell'] == cell) & (combined_cv_df['Cycle'] == '1000')]['IC'].values
    Cv = Cv1 - Cv0
    cv_result.append(Cv)
    cell_ids.append(cell_number)
cv_result = np.array(cv_result)
cell_ids = np.array(cell_ids)
cv_result = cv_result[:-1]
ecsa_array_subset = ecsa_array[::2][:-1]
cell_ids = cell_ids[:][:-1]

test_mask = np.array([cell_id % 3 == 0 for cell_id in cell_ids])
train_mask = ~test_mask


feature_df = pd.DataFrame()
feature_cols = []


for i in range(cv_result.shape[1]):
    col_name = f'cv_{i}'
    feature_cols.append(col_name)

diff_feature_cols = []
num_cols = cv_result.shape[1]
for i in range(num_cols):
    for j in range(i+1, num_cols):  
        col_name = f'cv_{i}_minus_cv_{j}'
        feature_df[col_name] = cv_result[:, i] - cv_result[:, j]
        diff_feature_cols.append(col_name)



train_features = feature_df[diff_feature_cols].loc[train_mask].reset_index(drop=True)
train_target = pd.DataFrame(ilim_array_subset[train_mask], columns=['target']).reset_index(drop=True)
sisso_data_train = pd.concat([train_target, train_features], axis=1)

test_features = feature_df[diff_feature_cols].loc[test_mask].reset_index(drop=True)
test_target = ilim_array_subset[test_mask]

operators = ['+', '-']
sisso_model = SissoModel(data=sisso_data_train, operators=operators, n_expansion=2, use_gpu=True, k=10)
rmse, equation, r2, selected_features = sisso_model.fit()
