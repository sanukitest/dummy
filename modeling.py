
# coding: utf-8
from utils import Utility
import pandas as pd
import numpy as np
import os
import cPickle
import datetime
import logging
import sys
import random
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn import linear_model, grid_search, cross_validation, preprocessing
from sklearn.neighbors import KNeighborsRegressor

VERSION = "ver56"
IS_FILTER = True
model_n = 5
featureset_n = 4
random.seed(71)


def exclude_bydate(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data["date_start"] = data["date"].apply(lambda x: x >= start_date.date())
    data["date_end"] = data["date"].apply(lambda x: x <= end_date.date())
    data = data.loc[~(data["date_start"] & data["date_end"]),:]
    data.drop(["date_start", "date_end"], axis=1, inplace=True)
    return data

class Model:
    def __init__(self, obj, kfold=3):
        self.kfold = kfold
        self.obj = obj
        self.best_params_ = [[0 for i in range(featureset_n)] for j in range(model_n)]
        self.predicted = [[0 for i in range(featureset_n)] for j in range(model_n)]
        self.clf = [[0 for i in range(featureset_n)] for j in range(model_n)]
        self.test_predicted = [[0 for i in range(featureset_n)] for j in range(model_n)]
        self.parameters = [0 for i in range(model_n)]
        self.featureset_master = dict({0:"raw", 1:"tfidf", 2:"pca", 3:"infra_holiday"})
        self.clf_master = dict({0:"RF", 1:"KNN", 2:"Elastic", 3:"GBR", 4:"ETR"})


    def set_feature(self):
        infra_cols = utility.cols_extractcols(data_train, ["infra"],["PCA", "tfidf"])
        weather_cols = utility.cols_extractcols(data_train, ["wea"])
        tfidf_infra_cols = utility.cols_extractcols(data_train, ["tfidf", "infra"])
        tfidf_snslocation_cols = utility.cols_extractcols(data_train, ["tfidf", "snslocation"])
        pca_infra_cols = utility.cols_extractcols(data_train, ["PCA", "infra"])
        pca_clim_cols = utility.cols_extractcols(data_train, ["PCA", "clim"])
        pca_snsraw_cols = utility.cols_extractcols(data_train, ["PCA", "snsraw"])

        loc_cols = utility.cols_extractcols(data_train, ["loc_"])
        jpy_cols = utility.cols_extractcols(data_train, ["JPY"])
        season_cols = utility.cols_extractcols(data_train,["season"])
        snslocation_cols = utility.cols_extractcols(data_train, ["snslocation"], ["tfidf", "PCA"])
        clim_cols = utility.cols_extractcols(data_train, ["clim"], ["tfidf", "PCA"])

        snsraw_cols = utility.cols_extractcols(data_train, ["twitter"], ["PCA"])
        holiday_cols = ["is_holiday", "is_weekend", "is_dayoff", "is_dayoff_mean"]
        holiday_cols_abroad = ["is_holiday_abroad", "is_weekend", "is_dayoff_abroad", "is_dayoff_mean_abroad"]

        if self.obj == "total":
            self.features = [0] * featureset_n
            self.features[0] = infra_cols + loc_cols + clim_cols + holiday_cols + snslocation_cols + ["snsfeature"]
            self.features[1] = tfidf_infra_snslocation_cols + holiday_cols
            self.features[2] = pca_infra_clim_snsraw_cols + holiday_cols
            self.features[3] = infra_cols + holiday_cols + ["snsfeature"]

        elif self.obj == "inbound":
            self.features = [0] * featureset_n
            self.features[0] = infra_cols + loc_cols + clim_cols + jpy_cols + holiday_cols_abroad + ["snsfeature"]
            self.features[1] = tfidf_infra_cols + holiday_cols_abroad
            self.features[2] = pca_infra_cols + pca_clim_cols + holiday_cols_abroad
            self.features[3] = infra_cols + holiday_cols_abroad + ["snsfeature"]

        elif self.obj == "japan":
            self.features = [0] * featureset_n
            self.features[0] = infra_cols + loc_cols + clim_cols + holiday_cols + ["snsfeature"]
            self.features[1] = tfidf_infra_cols +  holiday_cols
            self.features[2] = pca_infra_cols + pca_clim_cols + pca_snsraw_cols + holiday_cols
            self.features[3] = infra_cols + holiday_cols+ ["snsfeature"]

    def setmodel_stage1(self):
        self.clf[0] = [RandomForestRegressor(random_state=71)] * featureset_n
        self.parameters[0] = {'n_estimators':np.arange(50, 450, 100),"max_features":np.arange(3,12,3),"max_depth":np.arange(7,13,3)}

        self.clf[1] = [KNeighborsRegressor()] * featureset_n
        self.parameters[1] = {'n_neighbors':np.arange(4,15,2), "weights":["uniform", "distance"]}

        self.clf[2] = [linear_model.ElasticNet(max_iter=10000)] *  featureset_n
        self.parameters[2] = {'alpha': np.linspace(0.01, 1500, num=10), "l1_ratio": np.linspace(0.01,1,5)}

        self.clf[3] = [GradientBoostingRegressor(random_state=71)] * featureset_n
        self.parameters[3] = {'n_estimators':np.arange(200, 400, 100),"max_features":np.arange(6,12,3),"max_depth":np.arange(7,13,3)}

        self.clf[4] = [ExtraTreesRegressor(random_state=71)] * featureset_n
        self.parameters[4] = {'n_estimators':np.arange(100, 400, 100),"max_features":np.arange(6,12,3),"max_depth":np.arange(4,13,3)}

    def parametersearch_stage1(self):
        cv = cross_validation.KFold(len(data_train), n_folds=self.kfold, shuffle=True, random_state=1)
        for i in range(model_n):
            for j in range(featureset_n):
                grid = grid_search.GridSearchCV(self.clf[i][j], self.parameters[i], cv=cv, n_jobs=1, scoring="mean_absolute_error")
                if self.clf_master[i] == "KNN":
                    scaler = preprocessing.StandardScaler().fit(data_train[self.features[j]])
                    tmp = pd.DataFrame(scaler.transform(data_train[self.features[j]]), columns=data_train[self.features[j]].columns)
                    grid.fit(tmp, data_train[self.obj])
                    print "{0}_{1} params: {2} score:{3}".format(self.clf_master[i], self.featureset_master[j], grid.best_params_, grid.best_score_)
                else:
                    grid.fit(data_train[self.features[j]], data_train[self.obj])
                    print "{0}_{1} params: {2} score:{3}".format(self.clf_master[i], self.featureset_master[j], grid.best_params_, grid.best_score_)
                self.best_params_[i][j] = grid.best_params_

    def predict_stage1(self):
        for i in range(model_n):
            for j in range(featureset_n):
                self.clf[i][j].set_params(**self.best_params_[i][j])
                cv = cross_validation.KFold(len(data_train), n_folds=self.kfold, shuffle=True, random_state=71)
                if self.clf_master[i] == "KNN":
                    scaler = preprocessing.StandardScaler().fit(data_train[self.features[j]])
                    tmp = pd.DataFrame(scaler.transform(data_train[self.features[j]]), columns=data_train[self.features[j]].columns)
                    self.predicted[i][j] = cross_validation.cross_val_predict(self.clf[i][j], tmp, data_train[self.obj], cv=cv)
                    self.clf[i][j].fit(tmp, data_train[self.obj])
                    tmp_test = pd.DataFrame(scaler.transform(data_test[self.features[j]]), columns=data_test[self.features[j]].columns)
                    self.test_predicted[i][j] = self.clf[i][j].predict(tmp_test)

                else:
                    self.predicted[i][j] = cross_validation.cross_val_predict(self.clf[i][j], data_train[self.features[j]], data_train[self.obj], cv=cv)
                    self.clf[i][j].fit(data_train[self.features[j]], data_train[self.obj])
                    self.test_predicted[i][j] = self.clf[i][j].predict(data_test[self.features[j]])
                    # exclude outlier data
                    self.test_predicted[i][j][self.test_predicted[i][j]<0] = 0
                    self.test_predicted[i][j][self.test_predicted[i][j] > data_train[self.obj].max()] = data_train[self.obj].max()

                if (self.clf_master[i] == "RF") | (self.clf_master[i] == "GBC"):
                    self.__show_featureimportance(self.clf[i][j], data_train[self.features[j]].columns.tolist(), "ver" + str(i) + str(j), self.obj)

    def makedata_stage2(self):
        self.data_train2 = pd.DataFrame([])
        self.data_test2 = pd.DataFrame([])
        for i in range(model_n):
            for j in range(featureset_n):
                self.data_train2[self.clf_master[i] + "_" + self.featureset_master[j] ] = self.predicted[i][j]
                self.data_test2[self.clf_master[i] + "_" + self.featureset_master[j] ] = self.test_predicted[i][j]

                if self.obj=="japan":
                    infra_cols = utility.cols_extractcols(data_train, ["infra"],["PCA", "tfidf"])
                    holiday_cols = ["is_holiday", "is_weekend", "is_dayoff", "is_dayoff_mean"]
                    clim_cols = utility.cols_extractcols(data_train, ["clim"], ["tfidf", "PCA"])
                    loc_cols = utility.cols_extractcols(data_train, ["loc_"])

                    self.data_train2 = pd.concat([self.data_train2, data_train[infra_cols + holiday_cols + ["snsfeature"]]], axis=1)
                    self.data_test2 = pd.concat([self.data_test2, data_test[infra_cols + holiday_cols + ["snsfeature"] ]], axis=1)

                elif self.obj=="inbound":
                    infra_cols = utility.cols_extractcols(data_train, ["infra"],["PCA", "tfidf"])
                    holiday_cols_abroad = ["is_holiday_abroad", "is_weekend", "is_dayoff_abroad", "is_dayoff_mean_abroad"]
                    loc_cols = utility.cols_extractcols(data_train, ["loc_"])
                    clim_cols = utility.cols_extractcols(data_train, ["clim"], ["tfidf", "PCA"])
                    self.data_train2 = pd.concat([self.data_train2, data_train[infra_cols + holiday_cols_abroad +  ["snsfeature"] ]], axis=1)
                    self.data_test2 = pd.concat([self.data_test2, data_test[infra_cols + holiday_cols_abroad  +  ["snsfeature"]]], axis=1)




    def parametersearch_stage2(self):
        cv = cross_validation.KFold(len(self.data_train2), n_folds=self.kfold, shuffle=True, random_state=1)
        self.final_clf = RandomForestRegressor(random_state=71)
        self.final_params = {'n_estimators':np.arange(250, 350, 100),"max_features":np.arange(8,9,1),"max_depth":np.arange(6,7,1)}
        grid = grid_search.GridSearchCV(self.final_clf, self.final_params, cv=cv, scoring="mean_absolute_error")
        grid.fit(self.data_train2, data_train[self.obj])
        print "Final RF params: {0} score:{1}".format(grid.best_params_, grid.best_score_)
        self.final_best_param = grid.best_params_

    def predict_stage2(self, is_filter=True):
        self.final_clf.set_params(**self.final_best_param)
        self.final_clf.fit(self.data_train2, data_train[self.obj])
        self.__show_featureimportance(self.final_clf, self.data_train2.columns.tolist(), "final", self.obj)
        predicted = self.final_clf.predict(self.data_test2)
        if is_filter:
            predicted[predicted < 0] = 0
            predicted[predicted > data_train[self.obj].max()] = data_train[self.obj].max()
        return predicted

    def savefile(self):
        data_test_piv = data_test[["predicted", "location", "date"]].pivot(index="date", columns="location",
                                                                         values="predicted")
        with open("../output/{0}_{1}.pkl".format(self.obj, VERSION), "wb") as f:
            cPickle.dump(data_test_piv, f, -1)

    def makesubmit(self):
        with open("../output/japan_{0}.pkl".format(VERSION), "rb") as f:
            data_japan = cPickle.load(f)
        with open("../output/inbound_{0}.pkl".format(VERSION), "rb") as f:
            data_inbound = cPickle.load(f)
        data_total = data_japan + data_inbound
        data_total = data_total[utility.location_names]
        data_inbound = data_inbound[utility.location_names]

        submit = pd.concat([data_total, data_inbound], axis=1)
        submit.to_csv("../submit/submit_total_{0}_inbound_{0}.csv".format(VERSION), header=None, index_label=None)

    @staticmethod
    def __show_featureimportance(clf, features, version, obj):

        TOPFEATURES = 10
        feature_coef=dict(zip(features, clf.feature_importances_ ))
        top_features = sorted(feature_coef.items(), key=lambda x: x[1])[-TOPFEATURES:]
        top_features = [ k for k, value in top_features]
        with open("../output/stage2_topfeatures{0}_{1}.pkl".format(obj, version), "wb") as f:
            cPickle.dump(sorted(feature_coef.items(), key=lambda x: x[1]), f, -1)


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
utility = Utility()

logging.debug("create model")
for OBJ in ["japan", "inbound"]:
    with open("../feature/data_train.pkl", "rb") as f:
        data_train = cPickle.load(f)
    with open("../feature/data_test.pkl", "rb") as f:
        data_test = cPickle.load(f)
    if IS_FILTER:
        if OBJ=="japan":
            data_train = exclude_bydate(data_train, "2014-12-27", "2015-01-04")
        data_train.reset_index(drop=True, inplace=True)

    model = Model(OBJ)
    model.set_feature()
    model.setmodel_stage1()
    model.parametersearch_stage1()
    model.predict_stage1()

    model.makedata_stage2()
    model.parametersearch_stage2()
    data_test["predicted"] = model.predict_stage2()
    model.savefile()

model.makesubmit()
