import os
import pathlib

import joblib

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

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


TILES = ['b206',
 'b214',
 'b216',
 'b220',
 'b228',
 'b234',
 'b247',
 'b248',
 'b261',
 'b262',
 'b263',
 'b264',
 'b277',
 'b278',
 'b360',
 'b396']


cache = dcache.Cache(directory=CACHE_DIR, size_limit=CACHE_SIZE_LIMIT)
client = carpyncho.Carpyncho(cache=cache, parquet_engine="fastparquet")


def load_catalogs(*tiles, flt=None):
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
        if flt is not None:
            feats = feats[feats.id.isin(flt)]
            
        feats[FEATURES] = feats[FEATURES].astype(np.float32)
        feats = feats[
            (feats.cls >= 0) &
            (feats.Mean.between(12, 16.5))
        ].copy()
        
        feats = feats[~np.isinf(feats.Period_fit.values)]
        feats = feats[~feats.Gskew.isnull()]
        
        cats[tile] = feats
    
    cats = Container(cats)
    
    ## Scaling
    scl = StandardScaler()
    all_df = pd.concat(cats.values())
    
    all_df[FEATURES] = scl.fit_transform(all_df[FEATURES].values)
    
    # eliminamos lo que da nan o inf en lo normalizado
    for x in all_df.columns:
        if all_df[x].dtype == object:
            continue
        if np.isnan(all_df[x].values).sum():
            all_df = all_df[~np.isnan(all_df[x].values)]
        if np.isinf(all_df[x].values).sum():
            all_df = all_df[~np.isinf(all_df[x].values)]
    
    # removemos de los catalogos lo que estuvo en inf en lo normalizado
    for tname, tile in cats.items():
        cats[tname] = tile[tile.id.isin(all_df.id)].copy()
    
    # split
    scats = Container({
        gn: gdf.copy() for gn, gdf in all_df.groupby("tile")})
    
    return cats, scats, scl   
    
    

def load_tile_clf():
    # read
    flt = joblib.load(PATH / "sep_tile.pkl.bz2")
    return load_catalogs(*TILES, flt=flt)