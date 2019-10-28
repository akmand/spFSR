
# SpFtWgt: Feature Weighting and Feature Selection via SPSA
# Required Python version >= 3.6
# G. Yeo & V. Aksakalli
# GPL-3.0, 2019

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from SpFtWgt import SpFtWgt

# Source for prepare_dataset_for_modeling.py:
# https://github.com/vaksakalli/datasets/blob/master/prepare_dataset_for_modeling.py
from prepare_dataset_for_modeling import prepare_dataset_for_modeling


#################################################
# ##### CLASSIFICATION EXAMPLE  #################
#################################################

x, y = prepare_dataset_for_modeling('sonar.csv')

# specify a wrapper to use
wrapper = KNeighborsClassifier(n_neighbors=1)

# specify a metric to maximize
# (by default, sklearn metrics are defined as "higher is better")
# you need to make sure your scoring metric is consistent with your problem type,
# based on whether it is a binary or multi-class classification problem
# example: accuracy, f1, roc_auc, etc.
# more info on the scoring metrics can be found here:
# https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'accuracy'

# set the engine parameters
sp_engine = SpFtWgt(x=x, y=y, wrapper=wrapper, scoring=scoring)

# make sure the results are repeatable
np.random.seed(123)

# run the engine
# available engine parameters:
# 1. num_features: how many features to select
#    (in addition to features to keep, if any)
#    default value is 0 and it results in automatic feature selection
# 2. run_mode: 'default' (default) or 'fast'
# 3. stratified_cv: whether CV should be stratified or not (default is True)
#    stratified_cv must be set to False for regression problems
# 4. n_jobs: number of cores to be used in cross-validation (default is 1)
# 5. print_freq: print frequency for the output (default is 20)
# 6. features_to_keep_indices: indices of features to keep: default is None
# 7. perturbation: the perturbation size (default is 0.151)
# 8. gain_maximum: upper bound for gains (default is 1.15)
# 9. fs_threshold: feature selection threshold (default is 0.5)
sp_results = sp_engine.run().results

# list of available keys in the engine output
print('Available keys:\n', sp_results.keys())

# performance value of the best feature set
print('Best value:', sp_results.get('best_value'))

# indices of selected features
print('Indices of selected features: ', sp_results.get('features'))

# importance of selected features
print('Weights of selected features: ', sp_results.get('importance').round(3))

# number of iterations for the optimal set
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))

#################################################
# ##### REGRESSION EXAMPLE  #####################
#################################################

x, y = prepare_dataset_for_modeling('boston_housing.csv', is_classification=False)

wrapper = KNeighborsRegressor(n_neighbors=1)

scoring = 'r2'

sp_engine = SpFtWgt(x=x, y=y, wrapper=wrapper, scoring=scoring)

# make sure the results are repeatable
np.random.seed(123)

# for regression problems, you must set stratified_cv to False
# because the default value of True will not work
sp_results = sp_engine.run(stratified_cv=False).results

print('Best value:', sp_results.get('best_value'))
print('Indices of selected features: ', sp_results.get('features'))
print('Weights of selected features: ', sp_results.get('importance').round(3))
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))
