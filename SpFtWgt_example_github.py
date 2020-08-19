
# SpFtWgt: Feature Weighting and Feature Selection via SPSA
# Required Python version >= 3.6
# G. Yeo & V. Aksakalli
# GPL-3.0, 2020

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score

from SpFtWgt import SpFtWgt

# Source for prepare_dataset_for_modeling.py:
# https://github.com/vaksakalli/datasets/blob/master/prepare_dataset_for_modeling_github.py
from prepare_dataset_for_modeling_github import prepare_dataset_for_modeling


#################################################
# ##### CLASSIFICATION EXAMPLE  #################
#################################################

dataset = 'sonar.csv'
n_neighbors = 1
####
n_jobs = -2
####
x, y = prepare_dataset_for_modeling(dataset, is_classification=True)
print("dataset = " + dataset + ", shape of x = ", x.shape)

# specify a wrapper classifier to use
wrapper = KNeighborsClassifier(n_neighbors=n_neighbors)

# specify a metric to maximize
# (by default, sklearn metrics are defined as "higher is better")
# you need to make sure your _scoring metric is consistent with your problem type,
# based on whether it is a binary or multi-class classification problem
# example: accuracy, f1, roc_auc, etc.
# more info on the _scoring metrics can be found here:
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
# 2. iter_max: max number of iterations
#    for small datasets, iter_max = 150 works well (default)
#    for large datasets, iter_max = 300 works well
#    iteration stall limit will be iter_max/3
# 3. stratified_cv: whether CV should be stratified or not (default is True)
#    stratified_cv MUST be set to False for regression problems
# 4. n_jobs: number of cores to be used in cross-validation (default is 1)
# 5. print_freq: print frequency for the output (default is 5)
# 6. features_to_keep_indices: indices of features to keep: default is None
# 7. fs_threshold: feature selection threshold (default is 0.5)
SpFtWgt_results = sp_engine.run(n_jobs=n_jobs).results

# list of available keys in the engine output
print('Available keys:\n', SpFtWgt_results.keys())
# performance value of the best feature set
print('Best value:', SpFtWgt_results.get('best_value'))
# indices of selected features
print('Indices of selected features: ', SpFtWgt_results.get('features'))
# importance of selected features
print('Weights of selected features: ', SpFtWgt_results.get('importance').round(3))
# # gain sequence used during optimization
# print('BB Gains:', SpFtWgt_results.get('iter_results').get('gains'))

#################################################
# ######   Comparisons   ########################
#################################################

# Unweighted KNN (with no feature selection)
np.random.seed(123)
cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)
unweighted_no_fs_score = cross_val_score(wrapper,
                                         x,
                                         y,
                                         cv=cv_method,
                                         scoring='accuracy')
print(f"unweighted_no_fs_score: {unweighted_no_fs_score.mean():.2f}")

# Unweighted KNN with only the selected features
np.random.seed(123)
x_fs = x[:, SpFtWgt_results.get('features')]
unweighted_fs_score = cross_val_score(wrapper,
                                      x_fs,
                                      y,
                                      cv=cv_method,
                                      scoring='accuracy')
print(f"unweighted_fs_score: {unweighted_fs_score.mean():.2f}")

# Weighted KNN with feature selection
# an independent run for validation
np.random.seed(123)
ft_weights = SpFtWgt_results.get('importance')
x_fs_weighted = ft_weights * x_fs
weighted_fs_score = cross_val_score(wrapper,
                                    x_fs_weighted,
                                    y,
                                    cv=cv_method,
                                    scoring='accuracy')
print(f"weighted_fs_score: {weighted_fs_score.mean():.2f}")

#################################################
# ##### REGRESSION EXAMPLE  #####################
#################################################

dataset = 'boston_housing.csv'
n_neighbors = 1
n_jobs = -1
####
x, y = prepare_dataset_for_modeling(dataset, is_classification=False)
print("dataset = " + dataset + ", shape of x = ", x.shape)

wrapper = KNeighborsRegressor(n_neighbors=n_neighbors)

scoring = 'r2'

sp_engine = SpFtWgt(x=x, y=y, wrapper=wrapper, scoring=scoring)

# make sure the results are repeatable
np.random.seed(123)

# for regression problems, you must set stratified_cv to False
# because the default value of True will not work
SpFtWgt_results = sp_engine.run(stratified_cv=False, n_jobs=n_jobs).results

print('Best value:', SpFtWgt_results.get('best_value'))
print('Indices of selected features: ', SpFtWgt_results.get('features'))
print('Weights of selected features: ', SpFtWgt_results.get('importance').round(3))

#################################################
# ######   Comparisons   ########################
#################################################

# Unweighted KNN (with no feature selection)
np.random.seed(123)
cv_method = RepeatedKFold(n_splits=5, n_repeats=5, random_state=123)
unweighted_no_fs_score = cross_val_score(wrapper,
                                         x,
                                         y,
                                         cv=cv_method,
                                         scoring=scoring)
print(f"unweighted_no_fs_score: {unweighted_no_fs_score.mean():.3f}")

# Unweighted KNN with only the selected features
np.random.seed(123)
x_fs = x[:, SpFtWgt_results.get('features')]
unweighted_fs_score = cross_val_score(wrapper,
                                      x_fs,
                                      y,
                                      cv=cv_method,
                                      scoring=scoring)
print(f"unweighted_fs_score: {unweighted_fs_score.mean():.3f}")

# Weighted KNN with feature selection
# an independent run for validation
np.random.seed(123)
ft_weights = SpFtWgt_results.get('importance')
x_fs_weighted = ft_weights * x_fs
weighted_fs_score = cross_val_score(wrapper,
                                    x_fs_weighted,
                                    y,
                                    cv=cv_method,
                                    scoring=scoring)
print(f"weighted_fs_score: {weighted_fs_score.mean():.3f}")
