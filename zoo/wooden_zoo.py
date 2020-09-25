import xgboost as xgb
from fastFM.mcmc import FMRegression
from scipy.sparse import csc_matrix

class XGBoostStackFM(object):
    def __init__(self, xgb_iter, fm_iter=1000, fm_rank=2, xgb_param={}):
        self.xgb_iter = xgb_iter
        self.fm_iter = fm_iter
        self.fm_rank = fm_rank
        self.xgb_param = xgb_param

    def fit(self, x, y, x_val, y_val):
        train_dmatrix = xgb.DMatrix(x, label=y)
        val_dmatrix = xgb.DMatrix(x_val, label=y_val)
        evallist = [(train_dmatrix, 'train'), (test_dmatrix, 'test')]
        bst = xgb.train(self.xgb_param, train_dmatrix, self.xgb_iters, evallist)


class XGBoostBlendFM(object):
    def __init__(self, total_iter, xgb_iters=100, fm_rank=2, fm_iter=1000, xgb_param={}):
        self.total_itet = total_iter
        self.xgb_iters = xgb_iters
        self.fm_rank = fm_rank
        self.fm_iter = fm_iter
        self.xgb_param = xgb_param
        self.xgbs = []
        self.fms = []

    def fit(self, x, y, x_val, y_val):
