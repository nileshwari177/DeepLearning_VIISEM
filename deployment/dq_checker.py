# Standalone data-quality checker built on the trained Gaussian-Bernoulli RBM (NumPy only).
import os, pickle, numpy as np, pandas as pd
from scipy.stats import norm

def _sigmoid(x):  return 1.0 / (1.0 + np.exp(-x))
def _softplus(x): return np.logaddexp(0, x)

class DataQualityChecker:
    def __init__(self, path=None):
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "rbm_dq_model.pkl")
        with open(path, "rb") as f: self.__dict__.update(pickle.load(f))

    def assess(self, df, top=3):
        missing_cols = [c for c in self.feature_names if c not in df.columns]
        if missing_cols: raise ValueError(f"CSV is missing required columns: {missing_cols[:5]}...")
        X = df[self.feature_names].apply(pd.to_numeric, errors="coerce")
        n_missing = X.isna().sum(axis=1).values
        X = X.fillna(pd.Series(self.scaler_mean, index=self.feature_names))     # fill gaps with training mean
        V = (X.values - self.scaler_mean) / self.scaler_scale
        fe = 0.5 * ((V - self.b) ** 2).sum(axis=1) - _softplus(V @ self.W + self.c).sum(axis=1)   # free energy
        err = (V - (_sigmoid(V @ self.W + self.c) @ self.W.T + self.b)) ** 2                        # per-feature error
        out = pd.DataFrame({"free_energy": fe,
                            "anomaly_probability": norm.cdf((fe - self.fe_mu) / self.fe_sd),
                            "flagged": fe >= self.threshold, "missing_cells": n_missing}, index=df.index)
        out["suspicious_features"] = [", ".join(np.array(self.feature_names)[np.argsort(-e)[:top]]) for e in err]
        return out
