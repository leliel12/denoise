import os
import pathlib

import joblib


PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


def _load(filename):
    path = PATH / filename
    print("Reading '{path}')
    return path
    

def load_raw():
  return _load("full.pkl.bz2")
  
  
def load_scaled():
    return _load("full_scaled.pkl.bz2")


def load_scaler():
    return _load("scaler_full.pkl.bz2")
