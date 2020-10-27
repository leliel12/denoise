#!/usr/bin/env python
# coding: utf-8

# In[39]:

import datetime as dt
import itertools as it 
import os
import dataset
import pandas as pd
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

from libs.container import Container
import dataset


# In[44]:

#no_feats = ["tile", "cls"]
feats = ['n09_hk_color', 'c89_hk_color', 'c89_m2', 'n09_m4', 'n09_m2',
       'n09_jh_color', 'c89_m4', 'c89_jh_color']
noises = [f"noise_{idx}" for idx in range(30)]
all_feats = feats + noises

def load_noise():
    random = np.random.RandomState(seed=42)
    cats, scats, scl = dataset.load_tile_clf()
    wnoise = {}
    for tile, tdf in scats.items():
        tdf2 = tdf[["tile"] + feats].copy()
        for noise in noises:
            tdf2[noise] = random.uniform(len(tdf2))
        wnoise[tile] = tdf2
    print("foo")
    return wnoise


# In[45]:


def _run(cache_path, clf_class, **best_params):
    
    if os.path.exists(cache_path):
        print("Cached")
        return joblib.load(cache_path)    

    cpu = joblib.cpu_count()


    names = ["b261", "b277", "b278", "b360"]
    combs = list(it.combinations(names, 2))
    print(combs)

    sdata = load_noise()

    res = {}
#    return
    for t0, t1 in combs:
        print(dt.datetime.now())

        df = pd.concat([sdata[t0], sdata[t1]])

        cls = {name: idx for idx, name in enumerate(df.tile.unique())}
        print(cls)
        df["cls"] = df.tile.apply(cls.get)

        X = df[all_feats].values
        y = df.cls.values

        clf = clf_class(**best_params)
        sel = RFECV(clf, n_jobs=-1, cv=5)

        print("fit")
        sel.fit(X, y)

        print("storing")
        res[(t0, t1)] = sel

        print("-----------------------------")

    joblib.dump(res, cache_path, compress=3)    
    return res


# In[46]:


BEST_PARAMS = joblib.load("results/hp_selection.pkl.bz2")

def run_svm_noise():
    cache_path = "results/rfecv_noise_svm_linear.pkl.bz2"    
    svcl_params = BEST_PARAMS["svc_linear"].best_params_
    
    return _run(cache_path=cache_path, clf_class=SVC, **svcl_params)


# In[47]:


run_svm_noise()


# In[ ]:




