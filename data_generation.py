
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from rich import pretty, print
pretty.install()

def data_generation():
    df = pd.read_pickle("l_649_numbers_data.pkl")


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
    tmp_idx = df[df.dayofweek==2].iloc[0].name
    df = df.loc[tmp_idx:]


    # selecting the relevant columnns only
    cols_to_keep = [
        'Day',
        'White Ball 0', 'White Ball 1', 'White Ball 2', 'White Ball 3',
        'White Ball 4', 'White Ball 5', 'Grey Ball',]
    data = df[cols_to_keep]

    return df, data
