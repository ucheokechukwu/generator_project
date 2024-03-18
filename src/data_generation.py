import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from rich import pretty, print
pretty.install()

WB_RANGE = {f"WB {num + 1}": (num, 44 + num) for num in range(6)}


def data_generation():
    df = pd.read_pickle("data2/l_649_numbers_data.pkl")

    for col in df.columns[1:]:
        df[col] = df[col].astype('int')

    df = df.sort_values('Date')

    df['Day'] = df['Date'] - df['Date'].iloc[0]
    df['Day'] = df['Day'].dt.days
    df['Day'] = df['Day'] - df.Day.min()
    df['dayofweek'] = df['Date'].dt.weekday
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['is_leap_year'] = df['Date'].dt.is_leap_year
    df['weekofyear'] = df['Date'].dt.isocalendar()['week']

    df['is_leap_year'] = df['is_leap_year'].astype('int')
    df['weekofyear'] = df['weekofyear'].astype('int')

    # cutting off data from when the dayoftheweek was exclusively Friday for consistency??
    tmp_idx = df[df.dayofweek == 2].iloc[0].name
    df = df.loc[tmp_idx:]
    # binarizing dayoftheweek
    df['dayofweek'] = df['dayofweek'].apply(lambda x: 1 if x == 2 else 0)

    # selecting the relevant columnns only
    cols_to_keep = ['Day', 'WB 1', 'WB 2', 'WB 3', 'WB 4', 'WB 5', 'WB 6', 'GB',]
    data = df[cols_to_keep]

    return df, data

def make_train_test_splits(windows, labels, train_index, test_index):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    return windows[train_index], windows[test_index], labels[train_index], labels[test_index]

def transform_data(data, horizon=6):
    """
    input: pandas DataFrame
    output: windows and labels using numpy array slicing
    """
    
    data_array = data.to_numpy()
    # Calculate the number of slices
    num_slices = len(data) - horizon + 1
    # Use numpy's array slicing to create the transformed data
    data_transformed = np.array([data_array[i:i+horizon]
                                for i in range(num_slices)])
    # return windows and labels
    return data_transformed[:-1], data_array[horizon:]

def get_data(data, horizon):

    windows, labels = transform_data(data, horizon=horizon)
    windows = tf.expand_dims(windows, axis=-1)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
        windows=windows,
        labels=labels, test_split=0.2)
    return train_windows, test_windows, train_labels, test_labels


def get_training_data(data, HORIZON):
    X_train, X_test, train_labels, test_labels = get_data(data, HORIZON)
    y1_train, y2_train, y3_train, y4_train, y5_train, y6_train = train_labels[:, 0], train_labels[
        :, 1], train_labels[:, 2], train_labels[:, 3], train_labels[:, 4], train_labels[:, 5]
    y1_test, y2_test, y3_test, y4_test, y5_test, y6_test = test_labels[:, 0], test_labels[
        :, 1], test_labels[:, 2], test_labels[:, 3], test_labels[:, 4], test_labels[:, 5]
    y1_train.shape, y1_test.shape, X_train.shape, X_test.shape, train_labels.shape, test_labels.shape
    return X_train, X_test, y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, y1_test, y2_test, y3_test, y4_test, y5_test, y6_test


def make_index_for_input(target):
    """
    input: target in the form of 'WB N'
    used to create the index for input X6
    e.g. target=2, slice = 
    """
    try:
        t = int(target.split('WB ')[-1])
        assert t in range(6)
    except:
        return "Invalid target"  
    idx = [t]
    for x in range(6):
        if x != t:
            idx.extend((x, t))
    return idx

def scaling_data(data):
    """
    scales the target column to a range between 0-44
    """
    dt = data[WB_RANGE.keys()].copy()
    for k, v in WB_RANGE.items():
        dt[k] = (dt[k]-v[0])/44
    return dt

def get_X0_data(horizon, train_index, test_index,
                horizon_x0=20, target = 'WB 1', scaling=False):
    ### get transformer dataset
    df, _ = data_generation()
    
    ### new parts
    tmp = df.iloc[(horizon-horizon_x0):] # clip out the untrainable data
    tmp = tmp[target] # filter only the target label

    
    if scaling:
        tmp = (tmp-WB_RANGE[target][0])/44

    # get windowed data
    windows, labels = transform_data(tmp, horizon=horizon_x0)

    # splitting with KFold Stratification
    X0_train, X0_test, y_train, y_test = make_train_test_splits(
        np.expand_dims(windows, -1).astype(float), 
        labels, train_index, test_index)
    return X0_train, X0_test, y_train, y_test



def df_to_data(
        train_index, test_index, 
        target, 
        horizon_x0, horizon_x4, horizon_x5,
        scaling=False
        ):
    df, _ = data_generation()
    target_number = int(target[-1])
    
    # catching wrong configurations
    horizon = max(horizon_x0, horizon_x4, horizon_x5)  
    
    time_cols = ['Day', 'dayofweek', 'year', 'month',]
    target_cols = ['WB 1', 'WB 2', 'WB 3', 'WB 4', 'WB 5', 'WB 6']
    data = df[time_cols + target_cols]
    data['year'] = data['year'] - data.year.min() 
    
    ## geting X1, X2 and X3 training and test set
    tmp = data[[target]+time_cols]
    tmp = tmp.iloc[horizon:] # the length of data available depends on the max horizon
    tmp = tmp.reset_index(drop=True)   

    # extract labels
    X, y = tmp.drop(columns=[target]), tmp[target]
    X = X.astype('float')
    y = y.astype('int')


    # # splitting with KFold Stratification
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X1_train, X2_train, X3_train = X_train.iloc[:,:1], X_train.iloc[:,1:-1], X_train.iloc[:,-1:]
    X1_test, X2_test, X3_test = X_test.iloc[:,:1], X_test.iloc[:,1:-1], X_test.iloc[:,-1:]


    ############### getting the convoluted trains ========================
    
    tmp_data = data[target_cols].copy()
    if scaling:
        tmp_data = scaling_data(tmp_data)
    
        
    ## getting X4_train

    tmp = tmp_data.iloc[(horizon-horizon_x4):]
    windows, labels = transform_data(tmp, horizon_x4)
    # splitting with KFold Stratification
    X4_train, X4_test, train_labels, test_labels = make_train_test_splits(
              windows, labels, train_index, test_index)

    
    ## getting X5_train

    
    tmp = tmp_data.iloc[(horizon-horizon_x5):]
    windows, labels = transform_data(tmp, horizon_x5)   
    # splitting with KFold Stratification
    X5_train, X5_test, _, _ = make_train_test_splits(
              windows, labels, train_index, test_index)

    t = target_number - 1
    X5_train = X5_train[:,:,(0,t,5)]
    X5_test = X5_test[:,:,(0,t,5)]

    y1_train, y2_train, y3_train, y4_train, y5_train, y6_train = \
        train_labels[:, 0], train_labels[:, 1], train_labels[:, 2], train_labels[:, 3], train_labels[:, 4], train_labels[:, 5]
    y1_test, y2_test, y3_test, y4_test, y5_test, y6_test = \
        test_labels[:, 0], test_labels[:, 1], test_labels[:, 2], test_labels[:, 3], test_labels[:, 4], test_labels[:, 5]
    
          
            
    # get X0 dataset
    X0_train, X0_test, y0_train, y0_test = get_X0_data(horizon, train_index, test_index,
                horizon_x0=horizon_x0, target = target, scaling=scaling)
    
    # testing validity
    # only works if there's no scaling

    if scaling==False:
        
        assert all(y_train==y0_train) & all(y_test==y0_test)
    
        if target_number == 1:
            assert all(y_train==y1_train) and all(y_test==y1_test)
        if target_number == 2:
            assert all(y_train==y2_train) and all(y_test==y2_test)
        if target_number == 3:
            assert all(y_train==y3_train) and all(y_test==y3_test)
        if target_number == 4:
            assert all(y_train==y4_train) and all(y_test==y4_test)
        if target_number == 5:
            assert all(y_train==y5_train) and all(y_test==y5_test)
        if target_number == 6:
            assert all(y_train==y6_train) and all(y_test==y6_test)  
            
    y_train = y_train-(target_number)
    y_test = y_test-(target_number)
          
    return [X0_train, X0_test, X1_train, X2_train, X3_train, X1_test, X2_test, X3_test, X4_train, X4_test, 
            X5_train, X5_test, 
            y_train, y_test]



def sample_data(target = 'WB 1', 
                return_data=True,
                horizon_x0 = 100,
                horizon_x4=12, 
                horizon_x5=20, 
                scaling=False,
                docstring=False):

    df, _ = data_generation()
    
    # get train and test indices
    horizon = max(horizon_x0, horizon_x5, horizon_x4)
    tmp = df.iloc[horizon:]
    split = int(0.8*len(tmp))
    train_index = range(0,split)
    test_index  = range(split, len(tmp))
    
    if docstring:
        print("""Data is a list in this order:
        X0_train, X0_test, 
        X1_train, X2_train, X3_train, X1_test, X2_test, X3_test, 
        X4_train, X4_test, 
        X5_train, X5_test, 
        y_train, y_test""")
        print("""
        [bold]X0_train[/]\t\t\t - Time series of target column for attenuation, horizon_x0 for SubModel X0
        [bold]X1_train, X2_train, X3_train[/]\t - Datetime parameters for SubModelX123
        [bold]X4_train[/]\t\t\t - Windowed data of all 6 target columns with horizon_x4 for SubModelX4
        [bold]X5_train[/]\t\t\t - Windowed data of target column and 1st and 6th target with horizon_x5 for SubModelX4
        """)
    
    if return_data:
        return df_to_data(
               train_index = train_index,
               test_index = test_index,
               target = target,
               horizon_x0 = horizon_x0, 
               horizon_x4=horizon_x4,
               horizon_x5=horizon_x5, 
               scaling=scaling)

