
from rich import pretty, print, inspect
pretty.install()
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from random import random
from tqdm import tqdm


def parse_soup(soup_):
    
    soup = soup_.find('div', class_='col s_8_12')
    dates = [parsed_date.text for parsed_date in soup.find_all('div', class_='col s_3_12')]
    date_numbers = soup.find_all('div', class_='col s_9_12')
    white_balls = [[x.text for x in t.find_all('span', class_='white ball')] for t in date_numbers]
    grey_balls = [t.find('span', class_='grey ball').text[:-5] for t in date_numbers]

    return pd.DataFrame.from_dict({'Date': dates, 
                                   'White Balls': white_balls,
                                   'Grey Ball': grey_balls})



def get_records(years):
    links = ["https://www.lottodatabase.com/lotto-database/canadian-lotteries/lotto-649/draw-history/" + str(year) for year in years]
    dfs = []
    for link in tqdm(links):
        content = requests.get(link).content
        soup = BeautifulSoup(content, 'html.parser')
        dfs.append(parse_soup(soup))
        time.sleep(random()*20)
    display(pd.concat(dfs))
    return dfs

# dfs = get_records(years = list(range(1982, 2025)))


def original_parsing()
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)

    # df.rename(columns={'Grey Balls':'Grey Ball'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])


    tmp_cols = ["White Ball " + str(x) for x in range(6)]
    tmp_df = pd.DataFrame(df['White Balls'].to_list(), columns=tmp_cols)
    data = pd.concat([df.drop(columns=['White Balls']), tmp_df], axis=1)
    data = data[['Date',
     'White Ball 0',
     'White Ball 1',
     'White Ball 2',
     'White Ball 3',
     'White Ball 4',
     'White Ball 5',
     'Grey Ball']]
    # data.to_pickle("l_649_numbers_data.pkl")



def updating_dataset(dfs):
    """input:dfs -> recently scrapped data 
    output: backup existing dataset, updates with new records
    """
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)

    df['Date'] = pd.to_datetime(df['Date'])

    tmp_cols = ["White Ball " + str(x) for x in range(6)]
    tmp_df = pd.DataFrame(df['White Balls'].to_list(), columns=tmp_cols)
    data = pd.concat([df.drop(columns=['White Balls']), tmp_df], axis=1)
    data = data[['Date',
     'White Ball 0',
     'White Ball 1',
     'White Ball 2',
     'White Ball 3',
     'White Ball 4',
     'White Ball 5',
     'Grey Ball']]
    display(data)
    
    # load existing df and merge
    original_df = pd.read_pickle("data/l_649_numbers_data.pkl")
    updated_df = pd.concat(
        [original_df, data]).sort_values('Date').reset_index(drop=True).drop_duplicates()
    
    # backup old dataset
    filepath = f"data/l_649_numbers_data_backup_{int(time())}.pkl"
    original_df.to_pickle(filepath)
    
    # update dataset
    updated_df.to_pickle("data/l_649_numbers_data.pkl")
    print(f"Added {len(updated_df) - len(original_df)} new records", end=" ")
    print(f"from {str(original_df.Date.max()).split('00')[0]} to {str(updated_df.Date.max()).split('00')[0]}.")
    display(pd.read_pickle("data/l_649_numbers_data.pkl"))
    
    return f"Backup dataset pickled to {filepath}"
    





