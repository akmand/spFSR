
# spFtSel: Feature Selection and Ranking via SPSA
# Required Python version >= 3.6
# V. Aksakalli & Z. D. Yenice
# GPL-3.0, 2019
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589


from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import random
import numpy as np
import pandas as pd
import SpFtSel_engine

# make sure the results are repeatable
np.random.seed(8)
random.seed(8)

###################
# prepare input data:

# read in the ionosphere dataset
data_loc = 'http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
d = pd.read_csv(data_loc, header=None)

x = d.iloc[:, :-1].values  # x is all columns except the last
y = d.iloc[:, -1].values  # last column is y

x = preprocessing.StandardScaler().fit_transform(x)  # scale x with 0 mean, 1 std. deviation
y = preprocessing.LabelEncoder().fit_transform(y)  # encode y


###################
# run the algorithm:

# specify a wrapper to use
wrapper = GaussianNB()

# set the engine parameters
sp_engine = SpFtSel_engine.SpFtSel_engine(x, y, wrapper)

# - run the engine
# - first parameter is how many features to select:
# default value is 0 and it results in automatic feature selection
# - second parameter is the run mode:
# two run modes are available: regular (default) or extended
# - by default, no parallel processing will be performed.
# to activate parallel processing, please change the
# sp_params['n_jobs'] value in the SpFtSel_engine class.
sp_output = sp_engine.run(10, 'regular').results

# list of available keys in the engine output
print('Available keys:\n', sp_output.keys())

# performance value of the best feature set
print('Best value:', sp_output.get('best_value'))

# indices of selected features
print('Indices of selected features: ', sp_output.get('features'))

# importance of selected features
print('Importance of selected features: ', sp_output.get('importance').round(3))

# number of iterations for the optimal set
print('Total iterations for the optimal feature set:', sp_output.get('total_iter_for_opt'))

