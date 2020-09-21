# SpFSR: SPSA for Feature Selection and Ranking
# V. Aksakalli & Z. D. Yenice & A. Yeo
# GPL-3.0, 2020
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589

import logging
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.utils import shuffle
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
        self._print_freq = params.get('print_freq')
        self._starting_imps = params.get('starting_imps')
        self._features_to_keep_indices = params['features_to_keep_indices']
        ####
        self._num_cv_folds = params['cv_folds']
        self._num_cv_reps_eval = params['cv_reps_eval']
        self._num_cv_reps_grad = params['cv_reps_grad']
        self._num_grad_avg = params['num_grad_avg']
        #####
        self._input_x = None
        self._output_y = None
        self._n_observations = None  # in the dataset
        self._n_samples = None  # after any sampling
        self._wrapper = None
        self._scoring = None
        self._curr_imp_prev = None
        self._imp = None
        self._ghat = None
        self._cv_feat_eval = None
        self._cv_grad_avg = None
        self._curr_imp = None
        self._p = None
        self._best_value = -1 * np.inf
        self._best_std = -1
        self._stall_counter = 1
        self._run_time = -1
        self._best_iter = -1
        self._gain = -1
        self._selected_features = list()
        self._selected_features_prev = list()
        self._best_features = list()
        self._best_imps = list()
        self._iter_results = self.prepare_results_dict()

    def set_inputs(self, x, y, wrapper, scoring):
        self._input_x = x
        self._output_y = y
        self._wrapper = wrapper
        self._scoring = scoring

    def shuffle_and_sample_data(self):
        if any([self._input_x is None, self._output_y is None]):
            raise ValueError('There is no data inside shuffle_and_sample_data()')
        else:
            self._n_observations = self._input_x.shape[0]  # no. of observations in the dataset
            self._n_samples = self._input_x.shape[0]  # no. of observations after (any) sampling - initialization
            if self._n_samples_max and (self._n_samples_max < self._input_x.shape[0]):
                # don't sample more rows than what's in the dataset
                self._n_samples = self._n_samples_max
            self._input_x, self._output_y = shuffle(self._input_x, self._output_y, n_samples=self._n_samples)

    @staticmethod
    def prepare_results_dict():
        iter_results = dict()
        iter_results['values'] = list()
        iter_results['st_devs'] = list()
        iter_results['gains'] = list()
        iter_results['importance'] = list()
        iter_results['feature_indices'] = list()
        return iter_results

    def init_parameters(self):
        self._p = self._input_x.shape[1]
        if self._starting_imps:
            self._curr_imp = self._starting_imps
            self._logger.logger.info(f'Starting importance range: ({self._curr_imp.min()}, {self._curr_imp.max()})')
        else:
            self._curr_imp = np.repeat(0.0, self._p)  # initialize starting importance to (0,...,0)
        self._ghat = np.repeat(0.0, self._p)
        self._curr_imp_prev = self._curr_imp

    def print_algo_info(self):
        self._logger.logger.info(f'Wrapper: {self._wrapper}')
        self._logger.logger.info(f'Feature weighting: {self._ft_weighting}')
        self._logger.logger.info(f'Scoring metric: {self._scoring}')
        self._logger.logger.info(f'Number of jobs: {self._n_jobs}')
        self._logger.logger.info(f"Number of observations in the dataset: {self._n_observations}")
        self._logger.logger.info(f"Number of observations used: {self._n_samples}")
        self._logger.logger.info(f"Number of features available: {self._p}")
        self._logger.logger.info(f"Number of features to select: {self._num_features_selected}")

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
        selected_features = imp.copy()  # init_parameters
        if self._features_to_keep_indices is not None:
            selected_features[self._features_to_keep_indices] = np.max(imp)  # keep these by setting their imp to max

        if self._num_features_selected == 0:  # automated feature selection
            num_features_to_select = np.sum(selected_features >= 0.0)
            if num_features_to_select == 0:
                num_features_to_select = 1  # select at least one!
        else:  # user-supplied num_features_selected
            if self._features_to_keep_indices is None:
                num_features_to_keep = 0
            else:
                num_features_to_keep = len(self._features_to_keep_indices)

            num_features_to_select = np.minimum(self._p, (num_features_to_keep + self._num_features_selected))

        return selected_features.argsort()[::-1][:num_features_to_select]

    def eval_feature_set(self, cv_task, curr_imp):
        selected_features = self.get_selected_features(curr_imp)
        if self._ft_weighting:
            selected_ft_imp = curr_imp[selected_features]
            num_neg_imp = len(selected_ft_imp[selected_ft_imp < 0])
            if num_neg_imp > 0:
                raise ValueError(f'Error in feature weighting: {num_neg_imp} negative weights encountered' +
                                 ' - try reducing number of selected features or set it to 0 for auto.')
            x_all = curr_imp * self._input_x  # apply feature weighting
        else:
            x_all = self._input_x
        x_fs = x_all[:, selected_features]
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
        start_time = time.time()

        curr_iter_no = -1
        while curr_iter_no < self._iter_max:
            curr_iter_no += 1

            g_matrix = np.array([]).reshape(0, self._p)

            curr_imp_sel_ft_sorted = np.sort(self.get_selected_features(self._curr_imp))

            for grad_iter in range(self._num_grad_avg):

                imp_plus, imp_minus = None, None

                # keep random perturbing until plus/ minus perturbation vectors are different from the current vector
                bad_perturb_counter = 0
                while bad_perturb_counter < self._stall_limit:  # use the global stall limit

                    delta = np.where(np.random.sample(self._p) >= 0.5, 1, -1)

                    imp_plus = self._curr_imp + self._perturb_amount * delta
                    imp_minus = self._curr_imp - self._perturb_amount * delta

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

            ghat_prev = self._ghat.copy()

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
                    imp_diff = self._curr_imp - self._curr_imp_prev
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

            self._curr_imp_prev = self._curr_imp.copy()

            # make sure change is not too much
            curr_change_raw = self._gain * self._ghat
            curr_change_clipped = self.clip_change(curr_change_raw)

            # we use "+" below so that SPSA maximizes
            self._curr_imp = self._curr_imp + curr_change_clipped

            self._selected_features_prev = self.get_selected_features(self._curr_imp_prev)
            self._selected_features = self.get_selected_features(self._curr_imp)

            sel_ft_prev_sorted = np.sort(self._selected_features_prev)

            # make sure we move to a new solution by going further in the same direction
            same_feature_counter = 0
            curr_imp_orig = self._curr_imp.copy()
            same_feature_step_size = (self._gain_max - self._gain_min) / self._stall_limit
            while np.array_equal(sel_ft_prev_sorted, np.sort(self._selected_features)):
                same_feature_counter = same_feature_counter + 1
                curr_step_size = (self._gain_min + same_feature_counter * same_feature_step_size)
                curr_change_raw = curr_step_size * self._ghat
                curr_change_clipped = self.clip_change(curr_change_raw)
                self._curr_imp = curr_imp_orig + curr_change_clipped
                self._selected_features = self.get_selected_features(self._curr_imp)
                if same_feature_counter >= self._stall_limit:
                    break

            if same_feature_counter > 0:
                self._logger.logger.debug(f"=> iter_no: {curr_iter_no}, same_feature_counter = {same_feature_counter}")

            fs_perf_output = self.eval_feature_set(self._cv_feat_eval, self._curr_imp)

            self._iter_results['values'].append(np.round(fs_perf_output[0], self._rounding))
            self._iter_results['st_devs'].append(np.round(fs_perf_output[1], self._rounding))
            self._iter_results['gains'].append(np.round(self._gain, self._rounding))
            self._iter_results['importance'].append(np.round(self._curr_imp, self._rounding))
            self._iter_results['feature_indices'].append(self._selected_features)

            if self._iter_results['values'][curr_iter_no] >= self._best_value + self._stall_tolerance:
                self._stall_counter = 1
                self._best_iter = curr_iter_no
                self._best_value = self._iter_results['values'][curr_iter_no]
                self._best_std = self._iter_results['st_devs'][curr_iter_no]
                self._best_features = self._selected_features
                self._best_imps = self._curr_imp[self._best_features]
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
                self.init_parameters()

            if self._stall_counter >= self._stall_limit:
                # search stalled, start from scratch!
                self._logger.logger.info(f"===> iter_no: {curr_iter_no}, "
                                         f"iteration stall limit reached, initializing search...")
                self._stall_counter = 1  # reset the stall counter to give this solution enough time
                self.init_parameters()  # set _curr_imp and _g_hat to vectors of zeros

        self._run_time = round((time.time() - start_time) / 60, 2)  # report time in minutes
        self._logger.logger.info(f"SpFSR completed in {self._run_time} minutes.")
        self._logger.logger.info(
            f"Best value = {np.round(self._best_value, self._rounding)} with " +
            f"{len(self._best_features)} features and {len(self._iter_results.get('values')) - 1} total iterations.\n")

    def parse_results(self):
        return {'wrapper': self._wrapper,
                'scoring': self._scoring,
                'iter_results': self._iter_results,
                'selected_data': self._input_x[:, self._best_features],
                'selected_features': self._best_features,
                'selected_ft_importance': self._best_imps,
                'selected_num_features': len(self._best_features),
                'total_iter_overall': len(self._iter_results.get('values')),
                'total_iter_for_opt': np.argmax(np.array(self._iter_results.get('values'))),
                'selected_ft_score_mean': self._best_value,
                'selected_ft_score_std': self._best_std,
                'run_time': self._run_time,
                }


class SpFSR:
    def __init__(self, x, y, wrapper, scoring):
        self._x = x
        self._y = y
        self._wrapper = wrapper
        self._scoring = scoring
        self.results = None

    def run(self,
            num_features=0,  # a value of zero results in automatic feature selection
            iter_max=300,
            stall_limit=100,  # should be about 1/3 of iter_max
            n_samples_max=5000,  # if more rows than this in input data, a subset of data will be used - can be None
            ft_weighting=False,
            stratified_cv=True,  # *** MUST *** be set to False for regression problems
            gain_type='bb',  # either 'bb' (Barzilai & Borwein) (default) or 'mon' (monotone)
            cv_folds=5,
            num_grad_avg=4,  # for better gradient estimation, try increasing num_grad_avg to 8 or 10
            cv_reps_eval=3,
            cv_reps_grad=1,
            stall_tolerance=1e-8,
            display_rounding=3,
            is_debug=False,
            n_jobs=1,
            print_freq=10,
            starting_imps=None,
            features_to_keep_indices=None):

        # define a dictionary to initialize the SpFSR kernel
        sp_params = dict()
        sp_params['gain_type'] = gain_type
        sp_params['num_features'] = num_features
        sp_params['iter_max'] = iter_max
        sp_params['stall_limit'] = stall_limit
        sp_params['n_samples_max'] = n_samples_max
        sp_params['ft_weighting'] = ft_weighting
        sp_params['stratified_cv'] = stratified_cv
        sp_params['cv_folds'] = cv_folds
        sp_params['num_grad_avg'] = num_grad_avg
        sp_params['cv_reps_eval'] = cv_reps_eval
        sp_params['cv_reps_grad'] = cv_reps_grad
        sp_params['stall_tolerance'] = stall_tolerance
        sp_params['display_rounding'] = display_rounding
        sp_params['is_debug'] = is_debug
        sp_params['n_jobs'] = n_jobs
        sp_params['print_freq'] = print_freq
        sp_params['starting_imps'] = starting_imps
        sp_params['features_to_keep_indices'] = features_to_keep_indices
        ######################################

        kernel = SpFSRKernel(sp_params)

        kernel.set_inputs(x=self._x,
                          y=self._y,
                          wrapper=self._wrapper,
                          scoring=self._scoring)

        kernel.shuffle_and_sample_data()
        kernel.init_parameters()
        kernel.print_algo_info()
        kernel.gen_cv_task()
        kernel.run_kernel()
        self.results = kernel.parse_results()
        return self
