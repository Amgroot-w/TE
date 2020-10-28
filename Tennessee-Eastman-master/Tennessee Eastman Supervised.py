"""
Created on Tue May 14 22:44:05 2019

@author: Robot Hands (github.com/nvd919)
"""
import pandas as pd
import os
from sklearn.model_selection import RepeatedKFold as rkf
from sklearn.model_selection import train_test_split as tts

DATA_FOLDER = r"D:\Python Codes\Tennessee-Eastman-master\Tennessee Eastman Dataset\XGBoost\TE_process_dataset"
NUM_FILES = 22

training_files = [os.path.join(DATA_FOLDER, "d{:0>2}.dat".format(i)) for i in range(NUM_FILES)]
test_files = [os.path.join(DATA_FOLDER, "d{:0>2}_te.dat".format(i)) for i in range(NUM_FILES)]

training_data = [pd.read_csv(f, sep=' ', index_col = None) for f in training_files]
test_data = [pd.read_csv(f, sep=' ') for f in test_files]

# =============================================================================
#   Due to the relatively small training data (490 samples per fault) I think 
#   a K-fold strategy is necessary per fault. For that matter, I think 
#   splitting the training data to include a validation set in conjunction with
#   the KFold may be a wise move to avoid data leakage by tuning on the given
#   test set.
# =============================================================================
seed = 10
split_size = .30
rkf = rkf(n_splits = 5, n_repeats = 10, random_state = seed)
