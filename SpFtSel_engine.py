
# spFtSel: Feature Selection and Ranking via SPSA
# V. Aksakalli & Z. D. Yenice
# GPL-3.0, 2019
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589

from sklearn.externals.joblib import parallel_backend
import sklearn.metrics.scorer as scorer
from SpFtSel_kernel import SpFtSel_kernel


class SpFtSel_engine:
    def __init__(self, x, y, wrapper):
        self.x = x
        self.y = y
        self.wrapper = wrapper
        self.results = None

    def run(self, num_features=0, run_mode='regular'):

        # define a dictionary to initialize the SPFSR kernel
        sp_params = dict()

        sp_params['num_features'] = num_features

        # how many cores to use for parallel processing during cross validation
        # this value is directly passed in to cross_val_score()
        sp_params['n_jobs'] = 1

        # two gain types are available: bb (barzilai & borwein) or mon (monotone)
        sp_params['gain_type'] = 'bb'

        if run_mode == 'extended':
            sp_params['iter_max'] = 200
            sp_params['stall_limit'] = 50
            sp_params['num_grad_avg'] = 10
            sp_params['cv_reps_grad'] = 1
            sp_params['cv_reps_eval'] = 5
            sp_params['num_gain_smoothing'] = 1
        elif run_mode == 'regular':
            sp_params['iter_max'] = 100
            sp_params['stall_limit'] = 25
            sp_params['num_grad_avg'] = 2
            sp_params['cv_reps_grad'] = 1
            sp_params['cv_reps_eval'] = 2
            sp_params['num_gain_smoothing'] = 2
        else:
            raise ValueError('Error: Unknown SPFSR run mode.')

        # set other algorithm parameters
        sp_params['run_mode'] = run_mode
        sp_params['print_freq'] = 5  # how often do you want to print iteration results
        sp_params['cv_folds'] = 5
        sp_params['scoring_metric'] = scorer.accuracy_scorer
        sp_params['stratified_cv'] = True
        sp_params['maximize_score'] = True
        # two performance eval methods are available: cv or resub
        sp_params['perf_eval_method'] = 'cv'
        #####
        kernel = SpFtSel_kernel(sp_params)
        kernel.set_inputs(x=self.x, y=self.y, wrapper=self.wrapper)
        kernel.shuffle_data()
        kernel.init_parameters()
        kernel.gen_cv_task()
        with parallel_backend('multiprocessing'):
            kernel.run_spFtSel()
        self.results = kernel.parse_results()

        return self

