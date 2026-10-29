"""Checks of the twin, mass, vibration processing and load-test tools."""
import math

import numpy as np
from scipy import integrate

from bridge import config as C
from bridge import grillage as G
from bridge import loads
from bridge import section as S
from bridge import twin as T
from bridge import vibration as V


def test_total_mass_equals_dead_load():
    bg = T.build()
    m = T.mass_matrix(bg)[0::3].sum()
    W = 0.0
    for j in range(C.N_GIRDERS):
        W += integrate.quad(loads.stage1_line_load(j), 0, C.L, limit=200)[0]
        W += sum(P for _, P in loads.cross_girder_point_loads(j))
    sidl = 2 * (C.SIDL_LINE["crash_barrier"] + C.SIDL_LINE["railing"]) + 2 * 1.5 * C.FOOTPATH_FINISH + 11.0 * C.WEARING_COAT
    W += sidl * C.L
    assert abs(m - W / 9.81) / (W / 9.81) < 0.01


def test_first_frequency_close_to_beam_formula():
    bg = T.build(T.State(bearing_kv={(j, e): 1e4 for j in range(6) for e in (0, 1)}))
    md = T.modal(bg, 2)
    mass = T.mass_matrix(bg)[0::3].sum() / C.L
    EI = T.State().E_factor_dyn * C.GIRDER_CONCRETE["Ecm"] * 1e3 * sum(S.props(S.composite_layers(j)).I for j in range(6))
    f_beam = math.pi / (2 * C.L ** 2) * math.sqrt(EI / mass)
    assert md.labels[0].startswith("bending") and abs(md.freq[0] - f_beam) / f_beam < 0.04


def test_fdd_and_peak_pick_on_known_sdof():
    rng = np.random.default_rng(0)
    fs, f0, z = V.FS, 4.2, 0.02
    Q = rng.normal(size=(1, int(600 * V.FS_SIM)))
    acc = V.modal_acceleration(Q, [f0], z)[0, :: int(V.FS_SIM // fs)]
    a = np.vstack([acc, 0.5 * acc]) + rng.normal(0, 1e-3 * acc.std(), (2, len(acc)))
    f, P = V.welch_psd(V.preprocess(a))
    assert abs(V.peak_pick(f, P, [(3.5, 5.0)])[0] - f0) < 0.02
    fq, s, U = V.fdd(a)
    fr, u = V.fdd_modes(fq, s, U, [(3.5, 5.0)])[0]
    assert abs(fr - f0) < 0.03 and abs(u[1] / u[0] - 0.5) < 0.02


def test_mac_bounds():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(T.mac(v, -2 * v) - 1) < 1e-12 and T.mac(v, np.array([3.0, 0, -1.0])) < 1e-12


def test_bearing_stiffness_hand_values():
    b = T.bearing_stiffness()
    assert abs(b["shape_factor"] - 0.16 / (2 * 0.012 * 0.8)) < 1e-9
    assert abs(b["kh"] - 1.0 * 0.16 / 0.048 * 1e3) < 1e-6


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
