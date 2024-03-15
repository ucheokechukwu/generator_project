
import sys
try:
    sys.path.append('../src')
except:
    sys.path.append('src')
    
from rich import pretty, print, inspect
pretty.install()
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

from ensemble_model_classes import *
from data_generation import *
from helper_files import *
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import KFold
from keras import callbacks
from keras.utils import set_random_seed
df, _ = data_generation()

from sklearn.model_selection import KFold
from keras import callbacks
from keras.utils import set_random_seed
set_random_seed(0)
