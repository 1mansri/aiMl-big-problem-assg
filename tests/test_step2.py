"""Hand-check tests for the prestress and design tools."""
import math

import numpy as np

from bridge import config as C
from bridge import design as D
from bridge import prestress as PS


def straight_cable():
    return PS.Cable("S", 12, 1, 0.2, 0.2, None, 0.0, None, None)


def test_friction_straight_cable_is_wobble_only():
    x, sf, sa, lset = PS.stress_after_friction_slip(straight_cable())
    s_mid = C.JACK_RATIO * C.STRAND_FPK * math.exp(-C.MU * C.WOBBLE * C.L_GIRDER / 2)
    assert abs(sf[-1] - s_mid) < 1e-6


def test_slip_set_length_matches_closed_form():
    # linear friction loss p (MPa/m): l_set = sqrt(Ep * slip / p)
    x, sf, sa, lset = PS.stress_after_friction_slip(straight_cable())
    p = (sf[0] - sf[-1]) / (C.L_GIRDER / 2)
    expected = math.sqrt(C.STRAND_EP * C.SLIP / p)
    if expected < C.L_GIRDER / 2:
        assert abs(lset - expected) / expected < 0.01
    else:
        assert lset == x[-1] + C.END_OVERHANG


def test_parabola_geometry():
    c = PS.cables()[-1]
    assert abs(c.y(-C.END_OVERHANG) - c.y_anchor) < 1e-12
    assert abs(c.y(C.L / 2) - c.y_mid) < 1e-12
    assert abs(c.y(1.0) - c.y(C.L - 1.0)) < 1e-12           # symmetric


def test_creep_and_shrinkage_are_reasonable():
    phi = PS.creep_coefficient(45, 250, 14, 36500)
    assert 1.2 < phi < 2.5
    eps = PS.shrinkage(45, 250, 36500)
    assert 200e-6 < eps < 500e-6


def test_eigenstress_of_uniform_temperature_is_zero():
    s, _ = D.eigen_stress(1, C.L / 2, lambda y: np.full_like(y, 1e-4))
    assert all(abs(v) < 1e-6 for v in s.values())


def test_eigenstress_of_linear_temperature_is_zero():
    s, _ = D.eigen_stress(1, C.L / 2, lambda y: 1e-4 * y)
    assert all(abs(v) < 1e-6 for v in s.values())


def test_uls_rectangular_block_hand_check():
    """Parabola-rectangle stress block (alpha = 0.8095, beta = 0.416) with the NA inside the deck.

    Two cables only, so that the compression zone stays in the 220 mm deck slab.
    """
    saved = C.CABLES
    C.CABLES = saved[:2]
    try:
        sig = {c.name: 1000.0 for c in PS.cables()}
        M, xu, info = D.uls_moment_capacity(1, C.L / 2, sig)
        Ap = sum(c.area for c in PS.cables())
        T = Ap * 0.87 * C.STRAND_FPK / 1.15                  # MN, all yielded
        fcd = 0.67 * C.DECK_CONCRETE["fck"] / 1.5
        xu_hand = T / (0.8095 * fcd * 2.6)
        yp = sum(c.area * c.y(C.L / 2) for c in PS.cables()) / Ap
        M_hand = T * (C.H_COMPOSITE - 0.416 * xu_hand - yp) * 1e3
    finally:
        C.CABLES = saved
    assert xu_hand < C.T_DECK and info["yielded"]
    assert abs(xu - xu_hand) / xu_hand < 0.01
    assert abs(M - M_hand) / M_hand < 0.005


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
