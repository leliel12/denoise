#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import joblib


import itertools as it
import datetime as dt

import numpy as np

import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split)
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from libs.container import Container
import dataset

BEST_PARAMS = joblib.load("results/hp_selection.pkl.bz2")


def run_svm_linear():
    cache_path = "results/rfecv_svm_linear.pkl.bz2"    
    svcl_params = BEST_PARAMS["svc_linear"].best_params_
    
    return _run(cache_path=cache_path, clf_class=SVC, **svcl_params)
    
    
def run_rf():
    cache_path = "results/rfecv_rf_linear.pkl.bz2"    
    rf_params = BEST_PARAMS["rf"].best_params_
    
    return _run(cache_path=cache_path, clf_class=RandomForestClassifier, **rf_params)
    

def _run(cache_path, clf_class, **best_params):
    
    if os.path.exists(cache_path):
        print("Cached")
        res = joblib.load(cache_path)
    else:
        res = {}
        
    change = False

    cpu = joblib.cpu_count()


    names = ["b261", "b277", "b278", "b360"]
    combs = list(it.combinations(names, 2))
    print(combs)

    data = dataset.load_tile_clf()

    
    for t0, t1 in combs:
        if (t0, t1) in res:
            print("!!! Skip", t0, t1)
            continue
            
        print(dt.datetime.now(), t0, t1)

        df = pd.concat([data[t0], data[t1]])
        
        unscaled, df,  scl = dataset.scale(df)

        cls = {name: idx for idx, name in enumerate(df.tile.unique())}
        print(cls)
        df["cls"] = df.tile.apply(cls.get)

        X = df[dataset.FEATURES].values
        y = df.cls.values

        clf = clf_class(**best_params)
        sel = RFECV(clf, n_jobs=-1, cv=5)

        print("fit")
        sel.fit(X, y)

        print("storing")
        res[(t0, t1)] = {"selector": sel, "unscaled": unscaled, "scaled": df, "scl": scl}
        joblib.dump(res, cache_path, compress=3)
        
        print("-----------------------------")

    
    return res




