
# SpFtSel: Feature Selection and Ranking via SPSA-BB
# Required Python version >= 3.6
# V. Aksakalli & Z. D. Yenice
# GPL-3.0, 2020
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589


import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import preprocessing

from SpFtSel import SpFtSel


#################################################
# ##### CLASSIFICATION EXAMPLE  #################
#################################################

# make sure the results are repeatable
np.random.seed(8)

df = load_breast_cancer()
x, y = df.data, df.target

# specify a _wrapper to use
wrapper = DecisionTreeClassifier()

# specify a metric to maximize
# (by default, sklearn metrics are defined as "higher is better")
# you need to make sure your _scoring metric is consistent with your problem type,
# based on whether it is a binary or multi-class classification problem
# example: accuracy, f1, roc_auc, etc.
# more info on the _scoring metrics can be found here:
# https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'accuracy'
###
n_jobs = -2
###
# set the engine parameters
sp_engine = SpFtSel(x, y, wrapper, scoring)

# run the engine
# available engine parameters:
# 1. num_features: how many features to select
#    (in addition to features to keep, if any)
#    default value is 0 and it results in automatic feature selection
# 2. iter_max: max number of iterations
#    for small datasets, iter_max = 150 works well (default)
#    for large datasets, iter_max = 300 works well
#    iteration stall limit will be iter_max/3
# 3. stratified_cv: whether CV should be stratified or not (default is True)
#    stratified_cv MUST be set to False for regression problems
# 4. n_jobs: number of cores to be used in cross-validation (default is 1)
# 5. print_freq: print frequency for the output (default is 5)
# 6. features_to_keep_indices: indices of features to keep: default is None
sp_run = sp_engine.run(num_features=5, n_jobs=n_jobs)

# get the results of the run
sp_results = sp_run.results

# list of available keys in the engine output
print('Available keys:\n', sp_results.keys())

# performance value of the best feature set
print('Best value:', sp_results.get('best_value'))

# indices of selected features
print('Indices of selected features: ', sp_results.get('features'))

# importance of selected features
print('Importance of selected features: ', sp_results.get('importance').round(3))

# number of iterations for the optimal set
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))


#################################################
# ##### REGRESSION EXAMPLE  #####################
#################################################

# make sure the results are repeatable
np.random.seed(8)

df = load_boston()

x, y = df.data, df.target

wrapper = DecisionTreeRegressor()

scoring = 'r2'

# for regression problems:
# you MUST set stratified_cv to False
# as the default value of True will not work
# you should also scale y to be between 0 and 1 for the algorithm to work properly!
y = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()
sp_engine = SpFtSel(x, y, wrapper, scoring)
sp_run = sp_engine.run(num_features=5, stratified_cv=False, n_jobs=n_jobs)

sp_results = sp_run.results

print('Best value:', sp_results.get('best_value'))
print('Indices of selected features: ', sp_results.get('features'))
print('Importance of selected features: ', sp_results.get('importance').round(3))
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))

