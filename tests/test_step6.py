"""Checks of the ML building blocks on data with known answers."""
import numpy as np
import pandas as pd

from bridge import ml


def test_detector_false_alarm_rate_is_honest_on_new_healthy_data():
    """Threshold from out-of-fold scores: ~1 % false alarms on fresh data even with many features."""
    rng = np.random.default_rng(0)
    p = 60
    A = rng.normal(size=(p, p)) / np.sqrt(p)
    healthy = rng.normal(size=(240, p)) @ A
    fresh = rng.normal(size=(20000, p)) @ A
    det = ml.MahalanobisDetector().fit(healthy)
    fa = det.predict(fresh).mean()
    assert fa < 0.04, fa                      # in-sample thresholding gives ~8-10 % here
    # independent unit-variance features: an 8-sigma change of one feature adds 64 to the squared distance
    # (healthy: chi2 with 60 dof, mean 60, 99th percentile ~ 88), so it must be detected
    ind = ml.MahalanobisDetector().fit(rng.normal(size=(240, p)))
    shifted = rng.normal(size=(2000, p))
    shifted[:, 0] += 8.0
    assert ind.predict(shifted).mean() > 0.9


def test_temperature_compensation_recovers_slope():
    rng = np.random.default_rng(1)
    n = 300
    T = rng.uniform(15, 45, n)
    base_cols = {f"lt_defl_G{j}": 8.0 - 0.02 * (T - 30) + rng.normal(0, 0.05, n) for j in range(1, 7)}
    base_cols.update({f"lt_strain_G{j}": 100 + rng.normal(0, 3, n) for j in range(1, 7)})
    h = pd.DataFrame(dict(T_deck=T, f1=3.44 * (1 - 0.0017 * (T - 30)) + rng.normal(0, 0.005, n),
                          f2=3.51 + rng.normal(0, 0.005, n), f3=7.96 + rng.normal(0, 0.01, n),
                          wet=rng.random(n) < 0.3, hcp_min=-170 + rng.normal(0, 10, n), hcp_mean=-120.0,
                          hcp_soffit_mid_min=-150.0, resistivity_min=20.0, resistivity_median=30.0,
                          age_years=rng.uniform(1, 30, n), rebound_girder_mean=41.0, rebound_girder_min=37.0,
                          brg_slope_G1=0.385, brg_slope_G6=0.385, **base_cols))
    b = ml.Baseline().fit(h)
    a, slope, r2 = b.temp_models["f1"]
    assert abs(a - 3.44) < 0.003 and abs(slope / a + 0.0017) < 0.0002 and r2 > 0.9
    comp, _ = b.compensated(h, "f1")
    assert comp.std() < 0.5 * h.f1.std()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
