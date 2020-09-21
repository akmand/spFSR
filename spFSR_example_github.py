
# SpFSR: Feature Selection and Ranking via SPSA-BB
# Required Python version >= 3.6
# V. Aksakalli & Z. D. Yenice
# GPL-3.0, 2020
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589


import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import preprocessing

from spFSR import SpFSR


#################################################
# ##### CLASSIFICATION EXAMPLE  #################
#################################################

df = load_breast_cancer()
x, y = df.data, df.target

# specify a wrapper to use
wrapper = DecisionTreeClassifier(random_state=1)

# specify a metric to maximize
# (by default, sklearn metrics are defined as "higher is better")
# you need to make sure your _scoring metric is consistent with your problem type,
# based on whether it is a binary or multi-class classification problem
# example: accuracy, f1, roc_auc, etc.
# more info on the _scoring metrics can be found here:
# https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'accuracy'
###

# set the engine parameters
sp_engine = SpFSR(x, y, wrapper, scoring)

# make sure the results are repeatable
np.random.seed(1)

# run the spFSR engine
# available engine parameters (with default values in parentheses):
# 1.  num_features (0): number of features to select (in addition to features to keep, if any)
#     a value of 0 results in automatic feature selection
# 2.  iter_max (300): max number of iterations - for small datasets, try: iter_max = 150
# 3.  stall_limit (100): when to restart the search (up to iter_max) - should be around iter_max/3
#     stall_limit will also apply to same feature counter during y-plus and y-minus calculations
# 4.  n_samples_max (5000): max number of randomly selected observations to be used during search
#     can be None for all observations
# 5.  ft_weighting (False): if features should be weighted by their importance values - this usually helps with kNN's.
# 6.  stratified_cv (True): whether cross-validation (CV) should be stratified or not
#     stratified_cv *** MUST *** be set to False for regression problems
# 7.  gain_type ('bb'): either 'bb' (Barzilai & Borwein) gains or 'mon' (monotone) gains as the step size during search
# 8.  cv_folds (5): number of folds to use during (perhaps repeated) CV both for evaluation and gradient evaluation
# 9.  num_grad_avg (4): number of gradient estimates to be averaged for determining search direction
#     for better gradient estimation, try increasing this to 8 or 10 - though this will slow down the search
# 10. cv_reps_eval (3): number of CV repetitions for evaluating a candidate feature set
# 11. cv_reps_grad (1): number of CV repetitions for evaluating y-plus and y-minus solutions during gradient estimation
# 12. stall_tolerance (1e-8): tolerance in objective function change for stalling
# 13. display_rounding (3): number of digits to display during algorithm execution
# 14. is_debug (False): whether detailed search info should be displayed for each iteration
# 15. n_jobs (1): number of cores to be used in CV - this will be passed into cross_val_score()
# 16. print_freq (10): print frequency for the algorithm output
# 17. starting_imps (None): if a hot start is required
# 18. features_to_keep_indices (None): indices of features to keep for sure, if any
sp_run = sp_engine.run(num_features=5, iter_max=150, stall_limit=50)

# get the results of the run
sp_results = sp_run.results

# list of available keys in the engine output
print('Available keys:\n', sp_results.keys())

# performance value of the best feature set
print('Best value:', sp_results.get('selected_ft_score_mean'))

# indices of selected features
print('Indices of selected features: ', sp_results.get('selected_features'))

# importance of selected features
print('Importance of selected features: ', sp_results.get('selected_ft_importance').round(3))

# number of iterations for the optimal set
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))


#################################################
# ##### REGRESSION EXAMPLE  #####################
#################################################

df = load_boston()

x, y = df.data, df.target

wrapper = DecisionTreeRegressor(random_state=1)

scoring = 'r2'

# for regression problems:
# you *** MUST *** set stratified_cv to False
# as the default value of True will not work
# you *** MUST *** also scale y to be between 0 and 1 for the algorithm to work properly
y = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

# set the engine parameters
sp_engine = SpFSR(x, y, wrapper, scoring)

# make sure the results are repeatable
np.random.seed(1)

sp_run = sp_engine.run(num_features=5, iter_max=150, stall_limit=50, stratified_cv=False)

sp_results = sp_run.results

print('Best value:', sp_results.get('selected_ft_score_mean'))
print('Indices of selected features: ', sp_results.get('selected_features'))
print('Importance of selected features: ', sp_results.get('selected_ft_importance').round(3))
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))

