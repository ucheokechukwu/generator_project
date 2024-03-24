try:
    ROOT_DIR = "/Users/uche/Documents/BackUpFeb_5_2024/LotProj"
    import os
    if os.getcwd != ROOT_DIR:
        os.chdir(ROOT_DIR)
except:
    pass
    
import sys
sys.path.append('src')   
    
from warnings import filterwarnings
filterwarnings('ignore')
    
from rich import pretty, print, inspect
pretty.install()

from tqdm.auto import tqdm
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.utils import set_random_seed
set_random_seed(0)


