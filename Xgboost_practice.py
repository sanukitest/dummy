# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix, log_loss
iris=datasets.load_iris()
# python は0 base, R は1 base
# 偶数番目をトレーニングデータ
trainX=iris.data[1::2,:]
trainY=iris.target[1::2]
# 奇数番目を検証用データ
testX=iris.data[2::2,:]
testY=iris.target[2::2]

# DMatrix 型に変換
trainX=pd.DataFrame(trainX)
trainX.columns=iris.feature_names
dtrain = xgb.DMatrix(trainX.as_matrix(),label=trainY.tolist()) # マトリックス表示に変換する
testX=pd.DataFrame(testX)
testX.columns=iris.feature_names
dtest=xgb.DMatrix(testX.as_matrix())

np.random.seed(130) # シードを固定

# パラメータの設定
# http://puyokw.hatenablog.com/entry/2015/04/11/040941参照
params={'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'eta': 0.3, # default=0.3
        'max_depth': 6, # 木の深さの最大値 [default=6]
        'min_child_weight': 1, # [default=1]
        'subsample': 1, # [default=1]
        'colsample_bytree': 1, # [default=1]
        'num_class': 3
        }
# トレーニング
# num_boost_round 学習器の数（n_estimators）
bst=xgb.train(params,dtrain,num_boost_round=18)
# 変数重要度を求める
imp=bst.get_fscore()
print(imp)
# 以下のものが出力される
# {'f0': 15, 'f1': 7, 'f2': 57, 'f3': 28}
# このやり方だとアウトプットがめんどくさい
cv=xgb.cv(params,dtrain,num_boost_round=40,nfold=10)

# 目的変数の予測
bst=xgb.train(params,dtrain,18)
pred=bst.predict(dtest)
pred=pd.DataFrame(pred)

print confusion_matrix(testY, pred.idxmax(axis=1))


# パラメータのチューニングについてはhyperopt参照する
for i in range(100):
    print "test"
    
