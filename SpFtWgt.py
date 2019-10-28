
# spFtWgt: Feature Selection and Weighting via SPSA
# V. Aksakalli & G. Yeo
# GPL-3.0, 2019

import logging
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.externals.joblib import parallel_backend
import time

#######################################
#######################################


class SpFtWgtLog:

    # create logger
    logger = logging.getLogger('spFtWgt')
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        fmt='{name}-{levelname}: {message}',
        style='{',
    )

    # add formatter to CH
    ch.setFormatter(formatter)

    # add CH to logger
    logger.addHandler(ch)

#######################################
#######################################


class SpFtWgtKernel:

    def __init__(self, params):
        """
        algorithm parameters initialization
        """
        self._run_mode = params['run_mode']
        self._features_to_keep_indices = params['features_to_keep_indices']
        self._iter_max = params['iter_max']
        self._stall_limit = params['stall_limit']
        self._num_grad_avg = params['num_grad_avg']
        self._num_gain_smoothing = params['num_gain_smoothing']
        self._stratified_cv = params['stratified_cv']
        self._num_cv_reps_grad = params['cv_reps_grad']
        self._num_cv_reps_eval = params['cv_reps_eval']
        self._num_cv_folds = params['cv_folds']
        self._n_jobs = params['n_jobs']
        self._num_features_selected = params['num_features']
        self._fs_threshold = params['fs_threshold']
        self._print_freq = params['print_freq']
        #############
        self._perturb_amount = 0.15
        self._gain_min = 0.05
        self._gain_max = 1.15
        self._imp_max = 99.9
        ############
        self._imp_min = 0.0
        self._same_count_max = self._iter_max
        self._stall_tolerance = 10e-7
        self._decimals = 5  # for rounding, minimum required = 3
        #####
        self._input_x = None
        self._output_y = None
        self._wrapper = None
        self._scoring = None
        self._curr_imp_prev = None
        self._imp = None
        self._ghat = None
        self._cv_feat_eval = None
        self._cv_grad_avg = None
        self._curr_imp = None
        self._p = None
        self._stall_counter = 1
        self._best_iter = -1
        self._gain = -1
        self._run_time = -1
        self._best_value = -1 * np.inf
        self._best_std = np.inf
        self._selected_features = list()
        self._selected_features_prev = list()
        self._best_features = list()
        self._best_imps = list()
        self._raw_gain_seq = list()
        self._iter_results = self.prepare_results_dict()
        self._plotting_values = list()

    def set_inputs(self, x, y, wrapper, scoring):
        self._input_x = x
        self._output_y = y
        self._wrapper = wrapper
        self._scoring = scoring

    def shuffle_data(self):
        if any([self._input_x is None, self._output_y is None]):
            raise ValueError('There is no data inside shuffle_data()')
        else:
            self._input_x, self._output_y = shuffle(self._input_x, self._output_y)

    @staticmethod
    def prepare_results_dict():
        iter_results = dict()
        iter_results['values'] = list()
        iter_results['stds'] = list()
        iter_results['gains'] = list()
        iter_results['gains_raw'] = list()
        iter_results['importances'] = list()
        iter_results['feature_indices'] = list()
        return iter_results

    def init_parameters(self):
        self._p = self._input_x.shape[1]
        self._curr_imp = np.repeat(0.5, self._p)
        self._ghat = np.repeat(0.0, self._p)
        self._curr_imp_prev = self._curr_imp
        SpFtWgtLog.logger.info(f'Algorithm run mode: {self._run_mode}')
        SpFtWgtLog.logger.info(f'Wrapper: {self._wrapper}')
        SpFtWgtLog.logger.info(f'Scoring metric: {self._scoring}')
        SpFtWgtLog.logger.info(f"Number of features: {self._p}")
        SpFtWgtLog.logger.info(f"Number of observations: {self._input_x.shape[0]}")

    def get_selected_features(self, imp):
        """
        given the importance array, determine which features to select (as indices)
        :param imp: importance array
        :return: indices of selected features
        """
        selected_features = imp.copy()  # init_parameters
        if self._features_to_keep_indices is not None:
            selected_features[self._features_to_keep_indices] = 1.0  # keep these for sure by setting their imp to 1

        if self._num_features_selected == 0:  # automated feature selection
            num_features_to_select = np.sum(selected_features >= self._fs_threshold)
            if num_features_to_select == 0:
                num_features_to_select = 1  # select at least one!

        else:  # user-supplied _num_features_selected
            if self._features_to_keep_indices is None:
                num_features_to_keep = 0
            else:
                num_features_to_keep = len(self._features_to_keep_indices)

            num_features_to_select = \
                np.minimum(self._p, (num_features_to_keep + self._num_features_selected))

        return selected_features.argsort()[::-1][:num_features_to_select]

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
                self._cv_grad_avg = RepeatedKFold(n_splits=self._num_cv_folds, n_repeats=self._num_cv_reps_grad)
            else:
                self._cv_grad_avg = KFold(n_splits=self._num_cv_folds)

            if self._num_cv_reps_eval > 1:
                self._cv_feat_eval = RepeatedKFold(n_splits=self._num_cv_folds, n_repeats=self._num_cv_reps_eval)
            else:
                self._cv_feat_eval = KFold(n_splits=self._num_cv_folds)

    def eval_feature_set(self, cv_task, c_imp):
        x_weighted = c_imp*self._input_x
        selected_features = self.get_selected_features(c_imp)
        x_fs = x_weighted[:, selected_features]
        scores = cross_val_score(self._wrapper,
                                 x_fs,
                                 self._output_y,
                                 cv=cv_task,
                                 scoring=self._scoring,
                                 n_jobs=self._n_jobs)
        best_value_mean = scores.mean().round(self._decimals-2)
        best_value_std = scores.std().round(self._decimals-2)
        del scores
        return [-1 * best_value_mean, best_value_std]

    def run_kernel(self):
        start_time = time.time()
        for iter_i in range(self._iter_max):
            g_matrix = np.array([]).reshape(0, self._p)

            for g in range(self._num_grad_avg):
                delta = np.where(np.random.sample(self._p) >= 0.5, 1, -1)

                imp_plus = self._curr_imp + self._perturb_amount * delta
                imp_plus = np.maximum(imp_plus, self._imp_min)
                imp_plus = np.minimum(imp_plus, self._imp_max)

                imp_minus = self._curr_imp - self._perturb_amount * delta
                imp_minus = np.maximum(imp_minus, self._imp_min)
                imp_minus = np.minimum(imp_minus, self._imp_max)

                y_plus = self.eval_feature_set(self._cv_grad_avg, imp_plus)[0]
                y_minus = self.eval_feature_set(self._cv_grad_avg, imp_minus)[0]

                g_curr = (y_plus - y_minus) / (2 * self._perturb_amount * delta)

                g_matrix = np.vstack([g_matrix, g_curr])

            ghat_prev = self._ghat.copy()
            self._ghat = g_matrix.mean(axis=0)

            if np.count_nonzero(self._ghat) == 0:
                self._ghat = ghat_prev

            if iter_i == 0:
                self._gain = self._gain_min
                self._raw_gain_seq.append(self._gain)
            else:
                imp_diff = self._curr_imp - self._curr_imp_prev
                ghat_diff = self._ghat - ghat_prev
                bb_bottom = np.sum(imp_diff * ghat_diff)
                if bb_bottom < 10 ** (-7):  # make sure we don't end up with division by zero
                    self._gain = self._gain_min
                else:
                    self._gain = np.sum(imp_diff * imp_diff) / bb_bottom
                    self._gain = np.maximum(self._gain_min, (np.minimum(self._gain_max, self._gain)))
                self._raw_gain_seq.append(self._gain)
                if iter_i >= self._num_gain_smoothing:
                    raw_gain_seq_recent = self._raw_gain_seq[-self._num_gain_smoothing:]
                    self._gain = np.mean(raw_gain_seq_recent)

            SpFtWgtLog.logger.debug(f'iteration gain raw = {self._raw_gain_seq[-1]:1.4f}')
            SpFtWgtLog.logger.debug(f'iteration gain smooth = {self._gain:1.4f}')

            self._curr_imp_prev = self._curr_imp.copy()
            self._curr_imp = (self._curr_imp - self._gain * self._ghat).clip(min=self._imp_min, max=self._imp_max)

            self._selected_features_prev = self.get_selected_features(self._curr_imp_prev)
            self._selected_features = self.get_selected_features(self._curr_imp)

            same_feature_counter = 1
            curr_imp_orig = self._curr_imp.copy()
            while np.array_equal(self._selected_features_prev, self._selected_features):
                self._curr_imp = curr_imp_orig - same_feature_counter * self._gain_min * self._ghat
                self._curr_imp = self._curr_imp.clip(min=self._imp_min, max=self._imp_max)
                self._selected_features = self.get_selected_features(self._curr_imp)
                if same_feature_counter >= self._same_count_max:
                    break
                same_feature_counter = same_feature_counter + 1
            SpFtWgtLog.logger.debug(f"same_feature_counter = {same_feature_counter}")
            fs_perf_output = self.eval_feature_set(self._cv_feat_eval, self._curr_imp)

            self._iter_results['values'].append(round(-1 * fs_perf_output[0], self._decimals))
            self._iter_results['stds'].append(round(fs_perf_output[1], self._decimals))
            self._iter_results['gains'].append(round(self._gain, self._decimals))
            self._iter_results['gains_raw'].append(round(self._raw_gain_seq[-1], self._decimals))
            self._iter_results['importances'].append(self._curr_imp)
            self._iter_results['feature_indices'].append(self._selected_features)
            self._plotting_values.append(round(-1 * fs_perf_output[0], self._decimals))

            if self._iter_results['values'][iter_i] >= self._best_value + self._stall_tolerance:
                self._stall_counter = 1
                self._best_iter = iter_i
                self._best_value = self._iter_results['values'][iter_i]
                self._best_std = self._iter_results['stds'][iter_i]
                self._best_features = self._selected_features
                self._best_imps = self._curr_imp[self._best_features]
            else:
                self._stall_counter = self._stall_counter + 1

            if iter_i % self._print_freq == 0:
                SpFtWgtLog.logger.info(f"iter: {iter_i}, value: {self._iter_results['values'][iter_i]}, "
                                        f"std: {self._iter_results['stds'][iter_i]}, "
                                        f"num. features: {len(self._selected_features)}, "
                                        f"best value: {self._best_value} ({self._best_std}), "
                                        f"best iter: {self._best_iter}")
            if self._stall_counter > self._stall_limit:
                SpFtWgtLog.logger.info(f"iter: {iter_i}, value: {self._iter_results['values'][iter_i]}, "
                                        f"std: {self._iter_results['stds'][iter_i]}, "
                                        f"num. features: {len(self._selected_features)}, "
                                        f"best value: {self._best_value} ({self._best_std}), "
                                        f"best iter: {self._best_iter}")
                break

        self._run_time = round(time.time() - start_time, 2)
        SpFtWgtLog.logger.info(f"spFtWgt run completed in {self._run_time:.1f} seconds with {len(self._iter_results.get('values'))-1} total iterations.")

    def parse_results(self):
        selected_data = self._input_x[:, self._best_features]
        results_values = np.array(self._iter_results.get('values'))
        total_iter_for_opt = np.argmax(results_values)

        return {'_wrapper': self._wrapper,
                '_scoring': self._scoring,
                'selected_data': selected_data,
                'iter_results': self._iter_results,
                'features': self._best_features,
                'importance': self._best_imps,
                'num_features': len(self._best_features),
                'total_iter_overall': len(self._iter_results.get('values'))-1,
                'total_iter_for_opt': total_iter_for_opt,
                'best_value': self._best_value,
                'best_std': self._best_std,
                'run_time': self._run_time,
                }

#######################################
#######################################


class SpFtWgt:
    def __init__(self, x, y, wrapper, scoring='accuracy'):
        self._x = x
        self._y = y
        self._wrapper = wrapper
        self._scoring = scoring
        self.results = None

    def run(self,
            num_features=0,
            run_mode='extended',
            stratified_cv=True,
            n_jobs=1,
            print_freq=10,
            features_to_keep_indices=None,
            fs_threshold=0.5):

        # define a dictionary to initialize the SpFtWgt kernel
        sp_params = dict()

        sp_params['num_features'] = num_features
        sp_params['run_mode'] = run_mode
        sp_params['stratified_cv'] = stratified_cv
        sp_params['n_jobs'] = n_jobs
        sp_params['print_freq'] = print_freq
        sp_params['features_to_keep_indices'] = features_to_keep_indices
        sp_params['fs_threshold'] = fs_threshold

        if run_mode == 'extended':
            sp_params['cv_folds'] = 5
            sp_params['iter_max'] = 250
            sp_params['stall_limit'] = 50
            sp_params['num_grad_avg'] = 10
            sp_params['cv_reps_grad'] = 1
            sp_params['cv_reps_eval'] = 5
            sp_params['num_gain_smoothing'] = 1
        elif run_mode == 'regular':
            sp_params['cv_folds'] = 5
            sp_params['iter_max'] = 100
            sp_params['stall_limit'] = 25
            sp_params['num_grad_avg'] = 2
            sp_params['cv_reps_grad'] = 1
            sp_params['cv_reps_eval'] = 2
            sp_params['num_gain_smoothing'] = 2
        else:
            raise ValueError('Error: Unknown run mode')

        kernel = SpFtWgtKernel(sp_params)

        kernel.set_inputs(x=self._x,
                          y=self._y,
                          wrapper=self._wrapper,
                          scoring=self._scoring)

        kernel.shuffle_data()

        kernel.init_parameters()

        kernel.gen_cv_task()

        with parallel_backend('multiprocessing'):
            kernel.run_kernel()

        self.results = kernel.parse_results()

        return self