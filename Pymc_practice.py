# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pymc3 as pm
import theano.tensor as tt

if __name__ == "__main__":

    # 学習用データの作成
    traindata_2015 = pd.read_csv("./input/2015_J1_Result.txt")
    traindata_2014 = pd.read_csv("./input/2014_J1_Result.txt")
    traindata = pd.concat([traindata_2015,traindata_2014], axis=0)
    traindata.columns = ['matchID', "date", 'cupID', 'session', 'hometeam_ID', 'hometeam_name',
                      'hometeam_abbname','awayteam_ID', 'awayteam_name', 'awayteam_abbname',
                      'home_score', 'away_score']
    # モデリングの準備
    traindata = traindata[["hometeam_name", "awayteam_name", "home_score", "away_score"]]
    teams = traindata["hometeam_name"].unique()
    teams = pd.DataFrame(teams, columns=['team'])
    teams['i'] = teams.index
    traindata = pd.merge(traindata, teams, left_on='hometeam_name', right_on='team', how='left')
    traindata = traindata.rename(columns = {'i': 'i_home'}).drop('team', 1)
    traindata = pd.merge(traindata, teams, left_on='awayteam_name', right_on='team', how='left')
    traindata = traindata.rename(columns = {'i': 'i_away'}).drop('team', 1)
    observed_home_goals = traindata["home_score"].values
    observed_away_goals = traindata["away_score"].values
    home_team = traindata["i_home"].values
    away_team = traindata["i_away"].values
    num_teams = len(traindata["i_home"].unique())
    num_games = len(home_team)
    g = traindata.groupby('i_away')
    att_starting_points = np.log(g.away_score.mean())
    g = traindata.groupby('i_home')
    def_starting_points = -np.log(g.away_score.mean())

    # 階層ベイズモデリングの作成
    model = pm.Model()
    # モデル内を記述する
    with pm.Model() as model:
        # 事前分布
        tau_home    = pm.Gamma("tau_home", .1, .1)
        tau_away    = pm.Gamma("tau_away", .1, .1)
        tau_att     = pm.Gamma('tau_att',   .1, .1)
        tau_def     = pm.Gamma('tau_def',   .1, .1)
        intercept   = pm.Normal('intercept', 0, .0001)
        # チーム毎のパラメータ
        atts_star   = pm.Normal("atts_star",
                               mu   =0,
                               tau  =tau_att,
                               shape=num_teams)
        defs_star   = pm.Normal("defs_star",
                               mu   =0,
                               tau  =tau_def,
                               shape=num_teams)
        home_star   = pm.Normal("home_star",
                               mu   =0,
                               tau  =tau_home,
                               shape=num_teams)
        away_star   = pm.Normal("away_star",
                               mu   =0,
                               tau  =tau_away,
                               shape=num_teams)

        atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
        defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))
        home = pm.Deterministic('home', home_star - tt.mean(home_star))
        away = pm.Deterministic('away', away_star - tt.mean(away_star))

        home_theta = tt.exp(intercept + home[home_team] + atts[home_team] + defs[away_team])
        away_theta = tt.exp(intercept + away[away_team] + atts[away_team] + defs[home_team])

        # ポアソン分布を用いる
        home_points = pm.Poisson('home_points', mu=home_theta, observed=observed_home_goals)
        away_points = pm.Poisson('away_points', mu=away_theta, observed=observed_away_goals)

    with model:
        start = pm.find_MAP()
        # サンプリング方法はNUTS
        step = pm.NUTS(state=start)
        trace = pm.sample(2000, step, start=start, progressbar=True)

    # チーム毎のパラメータをまとめる
    teams["atts_star"] = pm.find_MAP(model=model)["atts_star"]
    teams["defs_star"] = pm.find_MAP(model=model)["defs_star"]
    teams["home"] = pm.find_MAP(model=model)["home_star"]
    teams["away"] = pm.find_MAP(model=model)["away_star"]
    teams["intercept"] = pm.find_MAP(model=model)["intercept"]*1.0

    # 最終節の予測
    result_submit = pd.read_csv("./rawdata/2015_J1_final_game.txt")
    result_submit.columns = ['matchID', "date", 'cupID', 'session', 'hometeam_ID', 'hometeam_name',
                      'hometeam_abbname','awayteam_ID', 'awayteam_name', 'awayteam_abbname',
                      'home_score', 'away_score']
    for i,row in result_submit.iterrows():
        atts_star = teams.ix[teams["team"]==row["hometeam_name"], "atts_star"].values[0]
        defs_star = teams.ix[teams["team"]==row["awayteam_name"], "defs_star"].values[0]
        home = teams.ix[teams["team"]==row["hometeam_name"], "home"].values[0]
        intercept = teams["intercept"].unique()[0]
        result_submit.loc[i, "home_score_predict"] = math.exp(home + defs_star + atts_star + intercept)

        atts_star = teams.ix[teams["team"]==row["awayteam_name"], "atts_star"].values[0]
        defs_star = teams.ix[teams["team"]==row["hometeam_name"], "defs_star"].values[0]
        away = teams.ix[teams["team"]==row["awayteam_name"], "away"].values[0]
        intercept = teams["intercept"].unique()[0]
        result_submit.loc[i, "away_score_predict"] = math.exp(away + defs_star + atts_star + intercept)

    # 提出用ファイルの作成
    hometeam_predicted = result_submit[["hometeam_ID","home_score_predict"]]
    awayteam_predicted = result_submit[["awayteam_ID","away_score_predict"]]
    hometeam_predicted.columns = ["team_ID","score_predicted"]
    awayteam_predicted.columns = ["team_ID","score_predicted"]
    submit_data = pd.concat([hometeam_predicted,awayteam_predicted],axis=0)
    submit_data.sort(["team_ID"], inplace=True)
    submit_data.to_csv("./output/submission.csv", index=False, header=None)
