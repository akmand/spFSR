
# SpFSR: Feature Selection and Ranking via SPSA-BB
# Required Python version >= 3.6
# D. Akman & Z. D. Yenice & A. Yeo
# GPL-3.0, 2022
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589


import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from spFSR import SpFSR


#################################################
# ##### CLASSIFICATION EXAMPLE  #################
#################################################

df = load_breast_cancer()
x, y = df.data, df.target

# specify prediction type
# pred_type needs to be 'c' for classification and 'r' for regression datasets
pred_type = 'c'

# specify a metric to maximize
# by default, sklearn metrics are defined as "higher is better"
# a 'None' value for scoring will default to 'accuracy' for classification and 'r2' (r-squared) for regression datasets
# you need to make sure your _scoring metric is consistent with your problem type
# example: accuracy, f1, roc_auc, etc.
# more info on the scoring metrics can be found here:
# https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'accuracy'

# specify a wrapper to use
# a 'None' value for the wrapper will default to a random forest wrapper
# this way spFSR will behave as a FILTER feature selection method
wrapper = DecisionTreeClassifier(random_state=1)

# set the engine parameters
sp_engine = SpFSR(x, y, pred_type=pred_type, scoring=scoring, wrapper=wrapper)

# run the spFSR engine
# available engine parameters (with default values in parentheses):
# 1.  num_features (0): number of features to select - a value of 0 results in automatic feature selection
# 2.  iter_max (100): max number of iterations
# 3.  stall_limit (35): when to restart the search (up to iter_max) - should be about iter_max/3
# 4.  n_samples_max (2500): max number of randomly selected observations to be used during search
#     can be 'None' for using all available observations
# 5.  ft_weighting (False): if features should be weighted by their importance values - this usually helps with kNN's.
# 6.  hot_start_num_ft_factor (15): factor for hot start: if user wants say 10 best features, hot start will result in
#     a maximum of 10*15 = 150 initial features for spFSR to choose from
# 7.  hot_start_max_auto_num_ft (75): max. number of initial hot start features for spFSR to choose from
#     if user selects auto feature selection
# 8.  use_hot_start (True): if hot start is to be used where initial feature importance vector is
#     determined by random forest importance (RFI) - this usually results in faster convergence
# 9.  hot_start_range (0.2): range for the initial feature importance vector in case of a hot start
#     for example: for a range of 0.2, most important RFI feature will have an imp. value of 0.1 and
#     the least important will have an imp. value of -0.1 - a value of 0 is also possible and
#     it will result in all RFI-selected features to have 0 imp. values
# 10. rf_n_estimators_hot_start (50): number of estimators for hot start RFI
# 11. rf_n_estimators_filter (5): number of estimators for prediction RFI in FILTER mode
# 12. gain_type ('bb'): either 'bb' (Barzilai & Borwein) gains or 'mon' (monotone) gains as the step size during search
# 13. cv_folds (5): number of folds to use during (perhaps repeated) CV both for evaluation and gradient evaluation
# 14. num_grad_avg (4): number of gradient estimates to be averaged for determining search direction
#     for better gradient estimation, try increasing this number - though this will slow down the search
# 15. cv_reps_eval (3): number of CV repetitions for evaluating a candidate feature set
# 16. cv_reps_grad (1): number of CV repetitions for evaluating y-plus and y-minus solutions during gradient estimation
# 17. stall_tolerance (1e-8): tolerance in objective function change for stalling
# 18. display_rounding (3): number of digits to display during algorithm execution
# 19. is_debug (False): whether detailed search info should be displayed for each iteration
# 20. random_state(1): seed for controlling randomness in the execution of the algorithm
# 21. n_jobs (1): number of cores to be used in CV - this will be passed into cross_val_score()
# 22. print_freq (10): iteration print frequency for the algorithm output
sp_run = sp_engine.run(num_features=5)

# get the results of the run
sp_results = sp_run.results

# list of available keys in the engine output
print('Available keys:', sp_results.keys())

# performance value of the best feature set
print('Best value:', sp_results.get('selected_ft_score_mean'))

# indices of selected features
print('Indices of selected features: ', sp_results.get('selected_features'))

# importance of selected features
print('Importance of selected features: ', sp_results.get('selected_ft_importance').round(3))

# number of iterations for the optimal set
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))

print('\n')

#################################################
# ##### REGRESSION EXAMPLE  #####################
#################################################

df = load_boston()

x, y = df.data, df.target

# set the engine parameters
sp_engine = SpFSR(x, y, pred_type='r', scoring='r2', wrapper=DecisionTreeRegressor(random_state=1))

sp_run = sp_engine.run(num_features=0)

sp_results = sp_run.results

print('Best value:', sp_results.get('selected_ft_score_mean'))
print('Indices of selected features: ', sp_results.get('selected_features'))
print('Importance of selected features: ', sp_results.get('selected_ft_importance').round(3))
print('Total iterations for the optimal feature set:', sp_results.get('total_iter_for_opt'))

