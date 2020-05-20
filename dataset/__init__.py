import os
import pathlib

import joblib

import numpy as np

import diskcache as dcache
import carpyncho

from libs.container import Container


PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

CACHE_DIR = PATH / "_carpyncho_cache_"

CACHE_SIZE_LIMIT = int(1e10)  # (10TB)


NO_FEATURES = [
    'id', 'tile', 'cnt', 'ra_k', 'dec_k', 
    'vs_type', 'vs_catalog', 'cls']


FEATURES = [
    'Amplitude', 'Autocor_length', 'Beyond1Std', 'Con', 'Eta_e', 
    'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35',
    'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65',
    'FluxPercentileRatioMid80', 'Freq1_harmonics_amplitude_0',
    'Freq1_harmonics_amplitude_1', 'Freq1_harmonics_amplitude_2',
    'Freq1_harmonics_amplitude_3', 'Freq1_harmonics_rel_phase_1',
    'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3',
    'Freq2_harmonics_amplitude_0', 'Freq2_harmonics_amplitude_1',
    'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3',
    'Freq2_harmonics_rel_phase_1', 'Freq2_harmonics_rel_phase_2',
    'Freq2_harmonics_rel_phase_3', 'Freq3_harmonics_amplitude_0',
    'Freq3_harmonics_amplitude_1', 'Freq3_harmonics_amplitude_2',
    'Freq3_harmonics_amplitude_3', 'Freq3_harmonics_rel_phase_1',
    'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3',
    'Gskew', 'LinearTrend', 'MaxSlope', 'Mean', 'MedianAbsDev', 'MedianBRP', 
    'PairSlopeTrend', 'PercentAmplitude', 'PercentDifferenceFluxPercentile', 
    'PeriodLS', 'Period_fit', 'Psi_CS', 'Psi_eta', 'Q31', 'Rcs', 'Skew', 
    'SmallKurtosis', 'Std', 'c89_c3', 'c89_hk_color', 'c89_jh_color', 
    'c89_jk_color', 'c89_m2', 'c89_m4', 'n09_c3', 'n09_hk_color', 
    'n09_jh_color', 'n09_jk_color', 'n09_m2', 'n09_m4', 'ppmb']


def _load(filename, raw=False):
    path = PATH / filename
    print(f"Reading '{path}'")
    
    sample = joblib.load(path)
    if raw:
        return sample
    grouped = sample.groupby("tile")
    data = Container({
        k: grouped.get_group(k).copy() for k in grouped.groups.keys()})
    del grouped, sample
    return data
    

def load_raw():
    return _load("full.pkl.bz2")
  

def load_scaled():
    return _load("full_scaled.pkl.bz2")


def load_scaler():
    return _load("scaler_full.pkl.bz2", raw=True)


cache = dcache.Cache(directory=CACHE_DIR, size_limit=CACHE_SIZE_LIMIT)
client = carpyncho.Carpyncho(cache=cache, parquet_engine="fastparquet")


def load_catalogs(*tiles):
    def gen_class(v):
        if v == "":
            return 0
        elif v.startswith("RRLyr-"):
            return 1
        return -1
    
    cats = {}
    for tile in tiles:
        feats = client.get_catalog(tile, "features")
        
        feats["tile"] = tile
        feats["cls"] = feats.vs_type.apply(gen_class)
        
        feats = feats[NO_FEATURES + FEATURES].copy()
        feats[FEATURES] = feats[FEATURES].astype(np.float32)
        feats = feats[
            (feats.cls >= 0) &
            (feats.Mean.between(12, 16.5))
        ].copy()
        
        feats = feats[~np.isinf(feats.Period_fit.values)]
        feats = feats[~feats.Gskew.isnull()]
        
        cats[tile] = feats
    return Container(cats)
