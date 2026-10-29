"""Feature engineering and AI/ML models for the condition assessment of the 33 m bridge.

Everything that depends on "normal behaviour" (temperature compensation, reference levels, the
anomaly detector) is fitted on healthy training campaigns only: in practice that is the first years of
monitoring of this bridge.  Truth columns (kind, severity, p_*, ...) are never used as inputs.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression

GIRDERS = [f"G{j}" for j in range(1, 7)]
TRUTH = ["campaign", "kind", "severity", "girders", "end", "pl_cracked", "f1_true"]

# ---------------------------------------------------------------------- feature groups (engineered names)
GROUPS = {
    "vibration": ["df1_pct", "df2_pct", "df3_pct", "f2_over_f1", "a_rms", "a_max"],
    "load_test": [f"lt_ratio_{g}" for g in GIRDERS] + ["lt_ratio_max", "lt_ratio_mean", "lt_ratio_min",
                                                       "lt_strain_ratio_max", "lt_shape_asym"],
    "monitoring": ["vw_res_G1", "vw_res_G3", "vw_res_G6", "vw_res_spread", "camber_res_G1", "camber_res_G3",
                   "camber_res_G4", "camber_res_G6", "camber_res_max", "camber_res_spread",
                   "brg_slope_ratio_min"],
    "ndt": ["upv_girder_mean", "upv_girder_min", "upv_deck_min", "rebound_girder_mean_c", "rebound_girder_min_c",
            "hcp_min_c", "hcp_mean_c", "hcp_end_minus_mid", "hcp_soffit_mid_min_c", "resistivity_min_c",
            "resistivity_median_c", "chloride_end_max", "n_upv_doubtful", "n_hcp_high_risk"],
    "visual": ["vis_hi", "vis_cs_girders", "vis_cs_bearings", "vis_cs_joints", "vis_cs_deck", "vis_n_obs",
               "vis_flexural_crack", "vis_duct_crack", "vis_leakage_girder_end", "vis_bearing_defect",
               "vis_surface_deterioration", "vis_spalling"],
}
GROUPS["all"] = sum(GROUPS.values(), [])

PHYSICAL_MEANING = {
    "df1_pct": "change of the 1st bending frequency after temperature compensation: global flexural stiffness / boundary restraint",
    "df2_pct": "change of the 1st torsion frequency: torsional stiffness, asymmetric damage, bearings",
    "df3_pct": "change of the transverse (distortion) frequency: deck slab and cross-girder stiffness",
    "f2_over_f1": "torsion / bending ratio: damage that affects the two modes differently (one girder vs. all)",
    "lt_ratio_max": "largest load-test deflection / healthy reference: local stiffness loss of a girder",
    "lt_ratio_min": "smallest deflection ratio: added restraint (seized bearing) stiffens the span",
    "lt_shape_asym": "change of the transverse deflection shape (G1-G3 vs G4-G6): which side is softer",
    "camber_res_max": "sag beyond the long-term reference curve: loss of prestress (or stiffness under permanent load)",
    "vw_res_spread": "difference between girders of the embedded-gauge strain residual: prestress lost in one girder only",
    "camber_res_spread": "difference between girders of the camber residual: localised loss in one girder",
    "vw_res_G1": "embedded-gauge strain residual: elongation of the bottom flange when prestress is lost",
    "brg_slope_ratio_min": "bearing movement per degC / alpha L: falls when a free bearing is restrained",
    "upv_girder_min": "lowest pulse velocity on the girders: cracked or poor-quality concrete",
    "hcp_min_c": "most negative half-cell potential (season-corrected): active corrosion somewhere",
    "hcp_end_minus_mid": "end zones vs mid-span potentials: corrosion fed by a leaking joint",
    "chloride_end_max": "chloride at end-block cover depth: risk of depassivation at anchorages",
    "vis_bearing_defect": "inspector recorded a bearing defect",
    "vis_cs_bearings": "worst condition state given to a bearing",
    "vis_cs_girders": "worst condition state on the girders",
    "vis_cs_joints": "worst condition state of the expansion joints (leaking seal = source of chloride)",
    "vis_flexural_crack": "inspector recorded transverse flexural cracks: the girder has cracked under load",
    "vis_duct_crack": "crack along a duct line: corrosion of the tendon or its surrounding steel",
    "vis_surface_deterioration": "scaling / surface loss: poor or deteriorating concrete",
    "vis_hi": "visual health index of the whole bridge",
    "upv_girder_mean": "mean pulse velocity on the girders: overall concrete quality",
    "hcp_soffit_mid_min_c": "most negative potential on the mid-span soffit (season-corrected): corrosion along the ducts",
    "hcp_mean_c": "mean potential (season-corrected): extent of corrosion activity",
    "resistivity_min_c": "lowest resistivity (season-corrected): wet, chloride-contaminated concrete",
    "rebound_girder_mean_c": "mean rebound corrected for carbonation hardening: surface hardness / strength",
    "lt_ratio_G1": "load-test deflection of G1 / healthy reference",
    "lt_strain_ratio_max": "largest soffit-strain ratio in the load test: local stiffness loss",
    "vis_leakage_girder_end": "leakage / efflorescence on girder end blocks",
}


@dataclass
class Baseline:
    """Healthy reference behaviour learnt from healthy training campaigns."""
    temp_models: dict = field(default_factory=dict)      # feature -> (intercept at 30 degC, slope per degC)
    wet_shift: dict = field(default_factory=dict)        # feature -> mean wet - dry
    ref: dict = field(default_factory=dict)              # reference values of raw features
    age_slope: dict = field(default_factory=dict)        # rebound carbonation hardening per sqrt(year)

    def fit(self, h):
        T = h.T_deck.values - 30.0
        for f in ["f1", "f2", "f3"] + [f"lt_defl_{g}" for g in GIRDERS] + [f"lt_strain_{g}" for g in GIRDERS]:
            lr = LinearRegression().fit(T[:, None], h[f].values)
            self.temp_models[f] = (float(lr.intercept_), float(lr.coef_[0]), float(lr.score(T[:, None], h[f].values)))
        for f in ["hcp_min", "hcp_mean", "hcp_soffit_mid_min"]:
            self.wet_shift[f] = float(h[h.wet][f].mean() - h[~h.wet][f].mean())
        for f in ["resistivity_min", "resistivity_median"]:   # multiplicative
            self.wet_shift[f] = float(np.log(h[h.wet][f]).mean() - np.log(h[~h.wet][f]).mean())
        sa = np.sqrt(h.age_years.values)
        for f in ["rebound_girder_mean", "rebound_girder_min"]:
            lr = LinearRegression().fit(sa[:, None], h[f].values)
            self.age_slope[f] = float(lr.coef_[0])
        for f in ["brg_slope_G1", "brg_slope_G6"]:
            self.ref[f] = float(h[f].mean())
        return self

    def compensated(self, df, f):
        a, b, _ = self.temp_models[f]
        return df[f] - b * (df.T_deck - 30.0), a


def engineer(df, base):
    """Physics-based features from the raw campaign columns."""
    X = pd.DataFrame(index=df.index)
    comp = {}
    for i, f in enumerate(["f1", "f2", "f3"], 1):
        v, ref = base.compensated(df, f)
        comp[f] = v
        X[f"df{i}_pct"] = 100 * (v / ref - 1)
    X["f2_over_f1"] = comp["f2"] / comp["f1"]
    X["a_rms"] = df.a_rms / df.traffic_per_min.clip(lower=1) ** 0.5       # normalise by traffic intensity
    X["a_max"] = df.a_max
    ratios, sratios = [], []
    for g in GIRDERS:
        v, ref = base.compensated(df, f"lt_defl_{g}")
        X[f"lt_ratio_{g}"] = v / ref
        ratios.append(X[f"lt_ratio_{g}"])
        s, sref = base.compensated(df, f"lt_strain_{g}")
        sratios.append(s / sref)
    R = pd.concat(ratios, axis=1)
    # G6 deflects only ~0.6 mm under the eccentric test: its ratio is noisy, use G1-G5 for max/min
    X["lt_ratio_max"] = R.iloc[:, :5].max(axis=1)
    X["lt_ratio_min"] = R.iloc[:, :5].min(axis=1)
    X["lt_ratio_mean"] = R.iloc[:, :5].mean(axis=1)
    X["lt_strain_ratio_max"] = pd.concat(sratios, axis=1).iloc[:, :5].max(axis=1)
    X["lt_shape_asym"] = R.iloc[:, :3].mean(axis=1) - R.iloc[:, 3:5].mean(axis=1)
    for c in ["vw_res_G1", "vw_res_G3", "vw_res_G6", "camber_res_G1", "camber_res_G3", "camber_res_G4", "camber_res_G6"]:
        X[c] = df[c]
    vw = df[["vw_res_G1", "vw_res_G3", "vw_res_G6"]]
    cam = df[["camber_res_G1", "camber_res_G3", "camber_res_G4", "camber_res_G6"]]
    X["vw_res_spread"] = vw.max(axis=1) - vw.min(axis=1)
    X["camber_res_max"] = cam.max(axis=1)
    X["camber_res_spread"] = cam.max(axis=1) - cam.min(axis=1)
    X["brg_slope_ratio_min"] = np.minimum(df.brg_slope_G1 / base.ref["brg_slope_G1"], df.brg_slope_G6 / base.ref["brg_slope_G6"])
    for c in ["upv_girder_mean", "upv_girder_min", "upv_deck_min", "hcp_end_minus_mid", "chloride_end_max",
              "n_upv_doubtful", "n_hcp_high_risk"]:
        X[c] = df[c]
    for c in ["hcp_min", "hcp_mean", "hcp_soffit_mid_min"]:
        X[c + "_c"] = df[c] - base.wet_shift[c] * df.wet
    for c in ["resistivity_min", "resistivity_median"]:
        X[c + "_c"] = df[c] * np.exp(-base.wet_shift[c] * df.wet)
    for c in ["rebound_girder_mean", "rebound_girder_min"]:
        X[c + "_c"] = df[c] - base.age_slope[c] * np.sqrt(df.age_years)
    for c in GROUPS["visual"]:
        X[c] = df[c]
    return X


# ---------------------------------------------------------------------- anomaly detection
class MahalanobisDetector:
    """Novelty detector: squared Mahalanobis distance to the healthy cloud (shrinkage covariance)."""

    def fit(self, Xh, q=0.99, folds=5, seed=0):
        """Fit on all healthy data; set the threshold on out-of-fold scores.

        With ~60 features and ~240 healthy campaigns, in-sample distances are optimistic and a threshold set
        on them gives several times the intended false-alarm rate on new healthy data.
        """
        self._fit(Xh)
        idx = np.random.default_rng(seed).permutation(len(Xh))
        oof = np.empty(len(Xh))
        for k in range(folds):
            te = idx[k::folds]
            tr = np.setdiff1d(idx, te)
            oof[te] = MahalanobisDetector()._fit(Xh[tr]).score(Xh[te])
        self._fit(Xh)
        self.thr = float(np.quantile(oof, q))
        return self

    def _fit(self, Xh):
        self.mu = Xh.mean(axis=0)
        self.sd = np.maximum(Xh.std(axis=0), 1e-2 * (np.abs(Xh.mean(axis=0)) + 1))   # floor: features constant when healthy
        self.cov = LedoitWolf().fit((Xh - self.mu) / self.sd)
        return self

    def score(self, X):
        return self.cov.mahalanobis((X - self.mu) / self.sd)

    def predict(self, X):
        return self.score(X) > self.thr


def isolation_forest(Xh, q=0.99, seed=0):
    m = IsolationForest(n_estimators=300, random_state=seed).fit(Xh)
    thr = float(np.quantile(-m.score_samples(Xh), q))
    return m, thr


def classifier(seed=0):
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)


def regressor(seed=0):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
