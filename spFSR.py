# SpFSR: SPSA for Feature Selection and Ranking
# V. Aksakalli & Z. D. Yenice & A. Yeo
# GPL-3.0, 2020
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589

import logging
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import time


class SpFSRLog:
    def __init__(self, is_debug):
        # if is_debug is set to True, debug information will also be printed
        self.is_debug = is_debug
        # create logger
        self.logger = logging.getLogger('SpFSR')
        # create console handler
        self.ch = logging.StreamHandler()
        # clear the logger to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        ####
        # set level
        if self.is_debug:
            self.logger.setLevel(logging.DEBUG)
            self.ch.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            self.ch.setLevel(logging.INFO)
        ####
        # create formatter
        self.formatter = logging.Formatter(fmt='{name}-{levelname}: {message}', style='{')
        # add formatter to console handler
        self.ch.setFormatter(self.formatter)
        # add console handler to logger
        self.logger.addHandler(self.ch)


class SpFSRKernel:

    def __init__(self, params):
        """
        algorithm parameters initialization
        """
        self._perturb_amount = 0.05
        #####
        self._gain_min = 0.01
        self._gain_max = 2.0
        #####
        self._change_min = 0.0
        self._change_max = 0.2
        #####
        self._bb_bottom_threshold = 1e-8
        #####
        self._mon_gain_A = 100  # SPSA-MONOTONE values from the 2016 PRL paper
        self._mon_gain_a = 0.75
        self._mon_gain_alpha = 0.6
        #####
        self._hot_start_num_ft_factor = 15
        #####
        self._use_hot_start = params['use_hot_start']
        self._hot_start_range = params['hot_start_range']
        self._gain_type = params['gain_type']
        self._num_features_selected = params['num_features']
        self._iter_max = params['iter_max']
        self._stall_limit = params['stall_limit']
        self._n_samples_max = params['n_samples_max']
        self._ft_weighting = params['ft_weighting']
        self._stratified_cv = params['stratified_cv']
        self._logger = SpFSRLog(params['is_debug'])
        self._stall_tolerance = params['stall_tolerance']
        self._rounding = params['display_rounding']
        self._n_jobs = params['n_jobs']
        self._print_freq = params['print_freq']
        self._random_state = params['random_state']
        ####
        self._num_cv_folds = params['cv_folds']
        self._num_cv_reps_eval = params['cv_reps_eval']
        self._num_cv_reps_grad = params['cv_reps_grad']
        self._num_grad_avg = params['num_grad_avg']
        #####
        self._input_x_all = None
        self._input_x_active = None  # active: after potential RF hot-start
        self._output_y = None
        self._pred_type = None  # 'c' or 'r'
        self._n_observations = None  # in the dataset
        self._n_samples = None  # after any sampling
        self._wrapper = None
        self._scoring = None
        self._idx_active = None
        self._imp_algo_start = None  # after potential RF hot-start
        self._imp = None
        self._imp_prev = None
        self._ghat = None
        self._cv_feat_eval = None
        self._cv_grad_avg = None
        self._p_all = None
        self._p_active = None
        self._best_value = -1 * np.inf
        self._best_std = -1
        self._stall_counter = 1
        self._run_time = -1
        self._best_iter = -1
        self._gain = -1
        self._selected_features = list()
        self._selected_features_prev = list()
        self._best_features_in_orig_data = list()
        self._best_features_active = list()
        self._best_imps = list()
        self._iter_results = self.prepare_results_dict()

    def set_inputs(self, x, y, pred_type, scoring, wrapper):
        self._input_x_all = x
        self._output_y = y
        self._pred_type = pred_type
        self._scoring = scoring
        self._wrapper = wrapper

    def shuffle_and_sample_data(self):
        if any([self._input_x_all is None, self._output_y is None]):
            raise ValueError('Error: There is no data inside shuffle_and_sample_data()')
        else:
            self._n_observations = self._input_x_all.shape[0]  # no. of observations in the dataset
            self._n_samples = self._input_x_all.shape[0]  # no. of observations after (any) sampling - initialization
            if self._n_samples_max and (self._n_samples_max < self._input_x_all.shape[0]):
                # don't sample more rows than what's in the dataset
                self._n_samples = self._n_samples_max
            self._input_x_all, self._output_y = shuffle(self._input_x_all,
                                                        self._output_y,
                                                        n_samples=self._n_samples,
                                                        random_state=self._random_state)
            self._input_x_active = self._input_x_all.copy()  # initialize after shuffling!

    @staticmethod
    def prepare_results_dict():
        iter_results = dict()
        iter_results['values'] = list()
        iter_results['st_devs'] = list()
        iter_results['gains'] = list()
        iter_results['importance'] = list()
        iter_results['feature_indices'] = list()
        return iter_results

    def print_algo_info(self):
        self._logger.logger.info(f'Wrapper: {self._wrapper}')
        self._logger.logger.info(f'Hot start: {self._use_hot_start}')
        if self._use_hot_start:
            self._logger.logger.info(f'Hot start range: {self._hot_start_range}')
        self._logger.logger.info(f'Feature weighting: {self._ft_weighting}')
        self._logger.logger.info(f'Scoring metric: {self._scoring}')
        self._logger.logger.info(f'Number of jobs: {self._n_jobs}')
        self._logger.logger.info(f"Number of observations in the dataset: {self._n_observations}")
        self._logger.logger.info(f"Number of observations used: {self._n_samples}")
        self._logger.logger.info(f"Number of features available: {self._input_x_all.shape[1]}")
        self._logger.logger.info(f"Number of features to select: {self._num_features_selected}")

    def prep_algo(self):
        self._p_all = self._input_x_all.shape[1]
        self._p_active = self._p_all  # initialization
        self._idx_active = list(range(self._p_all))  # initialization
        self._imp_algo_start = np.repeat(0.0, self._p_all)

        if self._pred_type == 'r':
            # for regression problems, we need to scale y to be between 0 and 1 for the algorithm to work properly
            self._output_y = preprocessing.MinMaxScaler().fit_transform(self._output_y.reshape(-1, 1)).flatten()

        if self._use_hot_start:
            if self._pred_type == 'c':
                hot_start_model = RandomForestClassifier(n_estimators=10, random_state=self._random_state)
            else:
                hot_start_model = RandomForestRegressor(n_estimators=10, random_state=self._random_state)

            hot_start_model.fit(self._input_x_all, self._output_y)

            if self._num_features_selected == 0:
                self._p_active = self._p_all  # select all the features in auto mode
            else:
                self._p_active = min(self._p_all, self._num_features_selected * self._hot_start_num_ft_factor)

            idx_hot_start_selected = np.argsort(hot_start_model.feature_importances_)[::-1][0:self._p_active].tolist()

            hot_ft_imp = hot_start_model.feature_importances_.tolist()

            hot_ft_imp_selected = [hot_ft_imp[i] for i in idx_hot_start_selected]

            self._idx_active = [self._idx_active[i] for i in idx_hot_start_selected]

            self._input_x_active = self._input_x_all[:, idx_hot_start_selected]

            if self._hot_start_range > 0:
                hot_range = (-0.5 * self._hot_start_range, + 0.5 * self._hot_start_range)
                self._imp_algo_start = preprocessing.minmax_scale(hot_ft_imp_selected, feature_range=hot_range)
            else:
                self._imp_algo_start = np.repeat(0.0, self._p_active)

        self._logger.logger.debug(f'Starting importance range: ({np.min(self._imp_algo_start)}, '
                                  f'{np.max(self._imp_algo_start)})')

    def init_parameters(self):
        self._imp = self._imp_algo_start.copy()
        self._imp_prev = self._imp.copy()
        self._ghat = np.repeat(0.0, self._p_active)

    def gen_cv_task(self):
        if self._stratified_cv:
            if self._num_cv_reps_grad > 1:
                self._cv_grad_avg = RepeatedStratifiedKFold(n_splits=self._num_cv_folds,
                                                            n_repeats=self._num_cv_reps_grad)
            else:
                self._cv_grad_avg = StratifiedKFold(n_splits=self._num_cv_folds)

            if self._num_cv_reps_eval > 1:
                self._cv_feat_eval = RepeatedStratifiedKFold(n_splits=self._num_cv_folds,
                                                             n_repeats=self._num_cv_reps_eval)
            else:
                self._cv_feat_eval = StratifiedKFold(n_splits=self._num_cv_folds)

        else:
            if self._num_cv_reps_grad > 1:
                self._cv_grad_avg = RepeatedKFold(n_splits=self._num_cv_folds,
                                                  n_repeats=self._num_cv_reps_grad)
            else:
                self._cv_grad_avg = KFold(n_splits=self._num_cv_folds)

            if self._num_cv_reps_eval > 1:
                self._cv_feat_eval = RepeatedKFold(n_splits=self._num_cv_folds,
                                                   n_repeats=self._num_cv_reps_eval)
            else:
                self._cv_feat_eval = KFold(n_splits=self._num_cv_folds)

    def get_selected_features(self, imp):
        """
        given the importance array, determine which features to select (as indices)
        :param imp: importance array
        :return: indices of selected features
        """
        if self._num_features_selected == 0:  # automated feature selection
            num_features_selected_actual = np.sum(imp >= 0.0)
            if num_features_selected_actual == 0:
                raise ValueError('Error: No features found with positive importance in auto mode.')
        else:  # user-supplied num_features_selected
            num_features_selected_actual = np.minimum(self._p_active, self._num_features_selected)
        return np.argsort(imp)[::-1][0:num_features_selected_actual].tolist()

    def eval_feature_set(self, cv_task, curr_imp):
        selected_features = self.get_selected_features(curr_imp)
        if self._ft_weighting:
            selected_ft_imp = curr_imp[selected_features]
            num_neg_imp = len(selected_ft_imp[selected_ft_imp < 0])
            if num_neg_imp > 0:
                raise ValueError(f'Error in feature weighting: {num_neg_imp} negative weights encountered' +
                                 ' - try reducing number of selected features or set it to 0 for auto.')
            x_active = curr_imp * self._input_x_active  # apply feature weighting
        else:
            x_active = self._input_x_active
        x_fs = x_active[:, selected_features]
        scores = cross_val_score(self._wrapper,
                                 x_fs,
                                 self._output_y,
                                 cv=cv_task,
                                 scoring=self._scoring,
                                 n_jobs=self._n_jobs)
        best_value_mean = scores.mean().round(self._rounding)
        best_value_std = scores.std().round(self._rounding)
        # sklearn metrics convention is that higher is always better
        # and SPSA here maximizes the obj. function
        return [best_value_mean, best_value_std]

    def clip_change(self, raw_change):
        # make sure change in the importance vector is bounded
        change_sign = np.where(raw_change > 0.0, +1, -1)
        change_abs_clipped = np.abs(raw_change).clip(min=self._change_min, max=self._change_max)
        change_clipped = change_sign * change_abs_clipped
        return change_clipped

    def run_kernel(self):
        np.random.seed(self._random_state)
        start_time = time.time()
        curr_iter_no = -1
        while curr_iter_no < self._iter_max:
            curr_iter_no += 1

            g_matrix = np.array([]).reshape(0, self._p_active)

            curr_imp_sel_ft_sorted = np.sort(self.get_selected_features(self._imp))

            for grad_iter in range(self._num_grad_avg):

                imp_plus, imp_minus = None, None

                # keep random perturbing until plus/ minus perturbation vectors are different from the current vector
                bad_perturb_counter = 0
                while bad_perturb_counter < self._stall_limit:  # use the global stall limit

                    delta = np.where(np.random.sample(self._p_active) >= 0.5, 1, -1)

                    imp_plus = self._imp + self._perturb_amount * delta
                    imp_minus = self._imp - self._perturb_amount * delta

                    imp_plus_sel_ft_sorted = np.sort(self.get_selected_features(imp_plus))
                    imp_minus_sel_ft_sorted = np.sort(self.get_selected_features(imp_minus))

                    # require both plus and minus to be different from current solution vector:
                    if (not np.array_equal(curr_imp_sel_ft_sorted, imp_plus_sel_ft_sorted)) and \
                            (not np.array_equal(curr_imp_sel_ft_sorted, imp_minus_sel_ft_sorted)):
                        # stop searching
                        break
                    else:
                        bad_perturb_counter += 1

                if bad_perturb_counter > 0:
                    self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, bad_perturb_counter: '
                                              f'{bad_perturb_counter} at gradient iteration {grad_iter}')

                # at this point, imp_plus *can* be the same as imp_minus if bad_perturb_counter reached the stall limit
                y_plus = self.eval_feature_set(self._cv_grad_avg, imp_plus)[0]
                y_minus = self.eval_feature_set(self._cv_grad_avg, imp_minus)[0]

                if y_plus != y_minus:
                    g_curr = (y_plus - y_minus) / (2 * self._perturb_amount * delta)
                    g_matrix = np.vstack([g_matrix, g_curr])
                else:
                    # unfortunately there will be one less gradient in the gradient averaging
                    self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                              f'y_plus == y_minus at gradient iteration {grad_iter}')

            if g_matrix.shape[0] < self._num_grad_avg:
                self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                          f'zero gradient(s) encountered: only {g_matrix.shape[0]} gradients averaged.')

            ghat_prev = self._ghat

            if g_matrix.shape[0] == 0:
                self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                          f'no proper gradient found, searching in the previous direction.')
                self._ghat = ghat_prev
            else:
                g_matrix_avg = g_matrix.mean(axis=0)
                if np.count_nonzero(g_matrix_avg) == 0:
                    self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                              f'zero gradient encountered, searching in the previous direction.')
                    self._ghat = ghat_prev
                else:
                    self._ghat = g_matrix_avg

            # gain calculation
            if self._gain_type == 'bb':
                if curr_iter_no == 0:
                    self._gain = self._gain_min
                else:
                    imp_diff = self._imp - self._imp_prev
                    ghat_diff = self._ghat - ghat_prev
                    bb_bottom = -1 * np.sum(imp_diff * ghat_diff)  # -1 due to maximization in SPSA
                    # make sure we don't end up with division by zero or negative gains:
                    if bb_bottom < self._bb_bottom_threshold:
                        self._gain = self._gain_min
                    else:
                        self._gain = np.sum(imp_diff * imp_diff) / bb_bottom
            elif self._gain_type == 'mon':
                self._gain = self._mon_gain_a / ((curr_iter_no + self._mon_gain_A) ** self._mon_gain_alpha)
            else:
                raise ValueError('Error: unknown gain type')

            # gain bounding
            self._gain = np.maximum(self._gain_min, (np.minimum(self._gain_max, self._gain)))

            self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                      f'iteration gain = {np.round(self._gain, self._rounding)}')

            self._imp_prev = self._imp

            # make sure change is not too much
            curr_change_raw = self._gain * self._ghat
            curr_change_clipped = self.clip_change(curr_change_raw)

            # we use "+" below so that SPSA maximizes
            self._imp = self._imp + curr_change_clipped

            self._selected_features_prev = self.get_selected_features(self._imp_prev)
            self._selected_features = self.get_selected_features(self._imp)

            sel_ft_prev_sorted = np.sort(self._selected_features_prev)

            # make sure we move to a new solution by going further in the same direction
            same_feature_counter = 0
            curr_imp_orig = self._imp.copy()
            same_feature_step_size = (self._gain_max - self._gain_min) / self._stall_limit
            while np.array_equal(sel_ft_prev_sorted, np.sort(self._selected_features)):
                same_feature_counter = same_feature_counter + 1
                curr_step_size = (self._gain_min + same_feature_counter * same_feature_step_size)
                curr_change_raw = curr_step_size * self._ghat
                curr_change_clipped = self.clip_change(curr_change_raw)
                self._imp = curr_imp_orig + curr_change_clipped
                self._selected_features = self.get_selected_features(self._imp)
                if same_feature_counter >= self._stall_limit:
                    break

            if same_feature_counter > 0:
                self._logger.logger.debug(f"=> iter_no: {curr_iter_no}, same_feature_counter = {same_feature_counter}")

            fs_perf_output = self.eval_feature_set(self._cv_feat_eval, self._imp)

            self._iter_results['values'].append(np.round(fs_perf_output[0], self._rounding))
            self._iter_results['st_devs'].append(np.round(fs_perf_output[1], self._rounding))
            self._iter_results['gains'].append(np.round(self._gain, self._rounding))
            self._iter_results['importance'].append(np.round(self._imp, self._rounding))
            self._iter_results['feature_indices'].append(self._selected_features)

            if self._iter_results['values'][curr_iter_no] >= self._best_value + self._stall_tolerance:
                self._stall_counter = 1
                self._best_iter = curr_iter_no
                self._best_value = self._iter_results['values'][curr_iter_no]
                self._best_std = self._iter_results['st_devs'][curr_iter_no]
                self._best_features_active = self._selected_features
                self._best_imps = self._imp[self._best_features_active]
            else:
                self._stall_counter += 1

            if curr_iter_no % self._print_freq == 0:
                self._logger.logger.info(f"iter_no: {curr_iter_no}, "
                                         f"num_ft: {len(self._selected_features)}, "
                                         f"value: {self._iter_results['values'][curr_iter_no]}, "
                                         f"st_dev: {self._iter_results['st_devs'][curr_iter_no]}, "
                                         f"best: {self._best_value} @ iter_no {self._best_iter}")

            if same_feature_counter >= self._stall_limit:
                # search stalled, start from scratch!
                self._logger.logger.info(f"===> iter_no: {curr_iter_no}, "
                                         f"same feature stall limit reached, initializing search...")
                self._stall_counter = 1  # reset the stall counter
                self.init_parameters()  # set _imp and _g_hat to vectors of zeros

            if self._stall_counter >= self._stall_limit:
                # search stalled, start from scratch!
                self._logger.logger.info(f"===> iter_no: {curr_iter_no}, "
                                         f"iteration stall limit reached, initializing search...")
                self._stall_counter = 1  # reset the stall counter to give this solution enough time
                self.init_parameters()

        self._run_time = round((time.time() - start_time) / 60, 2)  # report time in minutes
        self._logger.logger.info(f"SpFSR completed in {self._run_time} minutes.")
        self._logger.logger.info(
            f"Best value = {np.round(self._best_value, self._rounding)} with " +
            f"{len(self._best_features_active)} features and {len(self._iter_results.get('values')) - 1} total iterations.\n")

    def parse_results(self):
        self._best_features_in_orig_data = [self._idx_active[i] for i in self._best_features_active]
        return {'wrapper': self._wrapper,
                'scoring': self._scoring,
                'iter_results': self._iter_results,
                'selected_data': self._input_x_all[:, self._best_features_in_orig_data],
                'selected_features': self._best_features_in_orig_data,
                'selected_ft_importance': self._best_imps,
                'selected_num_features': len(self._best_features_in_orig_data),
                'total_iter_overall': len(self._iter_results.get('values')),
                'total_iter_for_opt': np.argmax(np.array(self._iter_results.get('values'))),
                'selected_ft_score_mean': self._best_value,
                'selected_ft_score_std': self._best_std,
                'run_time': self._run_time,
                }


class SpFSR:
    def __init__(self, x, y, pred_type, scoring=None, wrapper=None):
        self._x = x
        self._y = y
        self._pred_type = pred_type
        self._scoring = scoring
        self._wrapper = wrapper
        self.results = None

    def run(self,
            num_features=0,
            iter_max=100,
            stall_limit=30,
            n_samples_max=5000,
            ft_weighting=False,
            use_hot_start=True,
            hot_start_range=0.2,
            gain_type='bb',
            cv_folds=5,
            num_grad_avg=4,
            cv_reps_eval=3,
            cv_reps_grad=1,
            stall_tolerance=1e-8,
            display_rounding=3,
            is_debug=False,
            random_state=1,
            n_jobs=1,
            print_freq=10):

        if self._pred_type == 'c':
            stratified_cv = True
            if self._wrapper is None:
                self._wrapper = RandomForestClassifier(n_estimators=10, random_state=random_state)
            if self._scoring is None:
                self._scoring = 'accuracy'
        elif self._pred_type == 'r':
            stratified_cv = False
            if self._wrapper is None:
                self._wrapper = RandomForestRegressor(n_estimators=10, random_state=random_state)
            if self._scoring is None:
                self._scoring = 'r2'
        else:
            raise ValueError("Error: prediction type needs to be either 'c' for classification or 'r' for regression.")

        # define a dictionary to initialize the SpFSR kernel
        sp_params = dict()
        sp_params['stratified_cv'] = stratified_cv
        sp_params['use_hot_start'] = use_hot_start
        sp_params['hot_start_range'] = hot_start_range
        sp_params['gain_type'] = gain_type
        sp_params['num_features'] = num_features
        sp_params['iter_max'] = iter_max
        sp_params['stall_limit'] = stall_limit
        sp_params['n_samples_max'] = n_samples_max
        sp_params['ft_weighting'] = ft_weighting
        sp_params['cv_folds'] = cv_folds
        sp_params['num_grad_avg'] = num_grad_avg
        sp_params['cv_reps_eval'] = cv_reps_eval
        sp_params['cv_reps_grad'] = cv_reps_grad
        sp_params['stall_tolerance'] = stall_tolerance
        sp_params['display_rounding'] = display_rounding
        sp_params['is_debug'] = is_debug
        sp_params['random_state'] = random_state
        sp_params['n_jobs'] = n_jobs
        sp_params['print_freq'] = print_freq
        ######################################

        kernel = SpFSRKernel(sp_params)

        kernel.set_inputs(x=self._x,
                          y=self._y,
                          pred_type=self._pred_type,
                          scoring=self._scoring,
                          wrapper=self._wrapper)

        kernel.shuffle_and_sample_data()
        kernel.prep_algo()
        kernel.print_algo_info()
        kernel.init_parameters()
        kernel.gen_cv_task()
        kernel.run_kernel()
        self.results = kernel.parse_results()
        return self

            
