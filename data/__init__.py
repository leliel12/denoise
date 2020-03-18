import os
import pathlib

import joblib

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

print("Reading 'full.pkl.bz2' -> data.raw")
raw = joblib.load("full.pkl.bz2")

print("Reading 'full_scaled.pkl.bz2' -> data.scaled")
scaled = joblib.load("full_scaled.pkl.bz2")

print("Reading 'scaler_full.pkl.bz2' -> data.scaler")
scaler = joblib.load("scaler_full.pkl.bz2")