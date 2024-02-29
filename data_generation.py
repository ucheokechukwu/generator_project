
from model_builder import *
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from rich import pretty, print
pretty.install()


def data_generation():
    df = pd.read_pickle("data/l_649_numbers_data.pkl")

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

    # selecting the relevant columnns only
    cols_to_keep = [
        'Day',
        'White Ball 0', 'White Ball 1', 'White Ball 2', 'White Ball 3',
        'White Ball 4', 'White Ball 5', 'Grey Ball',]
    data = df[cols_to_keep]

    return df, data


def get_training_data(data, HORIZON):
    X_train, X_test, train_labels, test_labels = get_data(data, HORIZON)
    y1_train, y2_train, y3_train, y4_train, y5_train, y6_train = train_labels[:, 0], train_labels[
        :, 1], train_labels[:, 2], train_labels[:, 3], train_labels[:, 4], train_labels[:, 5]
    y1_test, y2_test, y3_test, y4_test, y5_test, y6_test = test_labels[:, 0], test_labels[
        :, 1], test_labels[:, 2], test_labels[:, 3], test_labels[:, 4], test_labels[:, 5]
    y1_train.shape, y1_test.shape, X_train.shape, X_test.shape, train_labels.shape, test_labels.shape
    return X_train, X_test, y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, y1_test, y2_test, y3_test, y4_test, y5_test, y6_test


def get_data_for_attention_convolution(data, horizon_attention=100, horizon_convolution=10):
    if horizon_attention - horizon_convolution < 0:
        return "Check dimensions for training data"
    data_convolution = data.iloc[(horizon_attention-horizon_convolution):, :]

    # training data for attention
    X_train, X_test, y1_train_, y2_train_, y3_train_, y4_train_, y5_train_, y6_train_, y1_test, y2_test, y3_test, y4_test, y5_test, y6_test \
        = get_training_data(data, horizon_attention)

    X1_train, X2_train, X3_train, X4_train, X5_train, X6_train = \
        X_train[:, :, 1], X_train[:, :, 2], X_train[:, :,
                                                    3], X_train[:, :, 4], X_train[:, :, 5], X_train[:, :, 6],

    X1_test, X2_test, X3_test, X4_test, X5_test, X6_test = \
        X_test[:, :, 1], X_test[:, :, 2], X_test[:, :,
                                                 3], X_test[:, :, 4], X_test[:, :, 5], X_test[:, :, 6],

    # training data for convolution
    X_train, X_test, y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, y1_test, y2_test, y3_test, y4_test, y5_test, y6_test \
        = get_training_data(data_convolution, horizon_convolution)

    return [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train,
            X1_test, X2_test, X3_test, X4_test, X5_test, X6_test,
            y1_train, y2_train, y3_train, y4_train, y5_train, y6_train,
            y1_test, y2_test, y3_test, y4_test, y5_test, y6_test,
            X_train, X_test]
