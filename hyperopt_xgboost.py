# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn import datasets, cross_validation
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

iris=datasets.load_iris()
trainX=iris.data[0::2,:]
trainY=iris.target[0::2]
testX=iris.data[1::2,:]
testY=iris.target[1::2]

np.random.seed(131)

def score(params):
	print "Training with params : "
	print params
	Sum=0.0
	ite=0.0
	for train, test in skf:
	  X_train, X_test, y_train, y_test = trainX[train], trainX[test], trainY[train], trainY[test]
	  dtrain = xgb.DMatrix(X_train, label=y_train)
	  dvalid = xgb.DMatrix(X_test, label=y_test)
	  watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
	  model = xgb.train(params, dtrain, num_boost_round=150,evals=watchlist,early_stopping_rounds=10)
	  predictions = model.predict(dvalid).reshape((X_test.shape[0], 3))
	  ite+=model.best_iteration
	  Sum+=model.best_score
	Sum/=len(skf)
	ite/=len(skf)
	print "\tAverage of best iteration {0}\n".format(ite)
	print "\tScore {0}\n\n".format(Sum)
	return {'loss': Sum, 'status': STATUS_OK}

# hp.quniform('変数名',最小値,最大値,間隔) の形式

def optimize(trials):
  space = {
    'eta' : hp.quniform('eta', 0.1, 0.5, 0.05),
    'max_depth' : hp.quniform('max_depth', 1, 10, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma' : hp.quniform('gamma', 0, 1, 0.05),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'num_class' : 3,
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'lambda': 1e-5,
    'alpha': 1e-5,
    'silent' : 1
    }
  best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)
  print best

skf = cross_validation.StratifiedKFold(trainY, n_folds=3, shuffle=True,random_state=123)
trials = Trials()
optimize(trials)
