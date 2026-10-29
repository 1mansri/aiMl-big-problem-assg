"""IRC:6-2017 live loads, impact, footpath load and dead loads of the 33 m bridge."""
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import section as S


# ------------------------------------------------------------------ vehicles
@dataclass
class Vehicle:
    """A vehicle as axle loads along x and wheel lines across y.

    axle_x: positions of the axles from the leading axle (m, increasing backwards)
    axle_P: total axle load (kN), shared equally by the wheel lines
    wheel_dy: transverse offsets of the wheel lines from the vehicle centre
    width: overall width used for the clearance rules
    kerb_clear: minimum clear distance from the kerb (barrier face) to the vehicle edge
    """
    name: str
    axle_x: np.ndarray
    axle_P: np.ndarray
    wheel_dy: tuple
    width: float
    kerb_clear: float
    impact: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def length(self):
        return float(self.axle_x.max())


def impact_class_a(span):
    """IRC:6 cl. 208.2, concrete bridges: 4.5 / (6 + L)."""
    return 4.5 / (6.0 + span)


def class_a_train(n_trains=2):
    """IRC:6 Class A train, repeated with a 20 m clear gap (cl. 204.1, Fig. 1)."""
    P = np.array([27, 27, 114, 114, 68, 68, 68, 68], float)
    gaps = [1.1, 3.2, 1.2, 4.3, 3.0, 3.0, 3.0]
    x1 = np.concatenate([[0.0], np.cumsum(gaps)])
    xs, Ps = [], []
    for k in range(n_trains):
        xs.append(x1 + k * (x1[-1] + 20.0))
        Ps.append(P)
    return Vehicle("Class A", np.concatenate(xs), np.concatenate(Ps), (-0.9, 0.9), 2.3, 0.15,
                   impact_class_a(C.L))


def class_70r_wheeled():
    """IRC:6 Class 70R wheeled vehicle, 1000 kN (cl. 204.1, Fig. 1A)."""
    P = np.array([80, 120, 120, 170, 170, 170, 170], float)
    gaps = [3.96, 1.52, 2.13, 1.37, 3.05, 1.37]
    x = np.concatenate([[0.0], np.cumsum(gaps)])
    # impact for wheeled 70R on spans >= 12 m: as Class A (cl. 208.3)
    return Vehicle("70R wheeled", x, P, (-0.965, 0.965), 2.79, 1.20, impact_class_a(C.L))


def class_70r_tracked(n_pts=24):
    """IRC:6 Class 70R tracked vehicle, 700 kN on two 4.57 m tracks at 2.05 m c/c.

    Each track's uniform pressure is represented by n_pts equal point loads.
    Impact 10 % for spans up to 40 m (cl. 208.3).
    """
    l = 4.57
    x = (np.arange(n_pts) + 0.5) * l / n_pts
    return Vehicle("70R tracked", x, np.full(n_pts, 700.0 / n_pts), (-1.025, 1.025), 2.89, 1.20, 0.10)


VEHICLE_GAP = 1.20      # min clear distance between passing vehicles, carriageway >= 7.5 m (IRC:6 Table 2 note)


@dataclass
class LoadCase:
    """Vehicles placed side by side, left to right, and the multi-lane reduction."""
    name: str
    vehicles: list
    reduction: float = 1.0


def irc_live_load_cases():
    """IRC:6 Table 6A for a 3-lane carriageway, plus every case with fewer vehicles.

    The 10 % reduction (cl. 205) is applied only when three Class A lanes are loaded.
    70R + Class A is treated as a two-vehicle case without reduction (conservative choice).
    """
    A, W, T = class_a_train(), class_70r_wheeled(), class_70r_tracked()
    return [
        LoadCase("1 x Class A", [A]),
        LoadCase("2 x Class A", [A, A]),
        LoadCase("3 x Class A (x0.9)", [A, A, A], 0.9),
        LoadCase("70R wheeled", [W]),
        LoadCase("70R wheeled + Class A", [W, A]),
        LoadCase("Class A + 70R wheeled", [A, W]),
        LoadCase("70R tracked", [T]),
        LoadCase("70R tracked + Class A", [T, A]),
        LoadCase("Class A + 70R tracked", [A, T]),
    ]


# ------------------------------------------------------------------ footpath
def footpath_intensity(span=C.L, width=None):
    """IRC:6 cl. 206.1 footpath live load (kPa) for the loaded span and footway width."""
    if width is None:
        width = dict(C.DECK_STRIPS)["footpath"]
    P = C.FOOTPATH_P
    if span <= 7.5:
        p = P
    elif span <= 30.0:
        p = P - (40 * span - 300) / 9
    else:
        p = (P - 260 + 4800 / span) * (16.5 - width) / 15
    return p * 9.81e-3      # kg/m^2 -> kPa


# ------------------------------------------------------------------ dead loads
def girder_web(x):
    """Web thickness at x (from the left bearing) including the end-block transition."""
    d = min(x + C.END_OVERHANG, C.L + C.END_OVERHANG - x)       # distance from nearest girder end
    if d <= C.END_BLOCK_LEN:
        return C.END_BLOCK_WEB
    if d >= C.END_BLOCK_LEN + C.END_BLOCK_TAPER:
        return C.B_WEB
    f = (d - C.END_BLOCK_LEN) / C.END_BLOCK_TAPER
    return C.END_BLOCK_WEB + f * (C.B_WEB - C.END_BLOCK_WEB)


def girder_self_weight(x):
    return C.GAMMA_CONC * S.props(S.girder_layers(girder_web(x))).A


def cross_girder_point_loads(i_girder):
    """Weight of the cross girders delivered to girder i as (x, P) point loads."""
    out = []
    for x, tw, d in C.CROSS_GIRDERS:
        bay = C.GIRDER_SPACING - (C.END_BLOCK_WEB if x in (0.0, C.L) else C.B_WEB)
        w_bay = C.GAMMA_CONC * tw * d * bay
        n_bays = 1 if i_girder in (0, C.N_GIRDERS - 1) else 2
        out.append((x, 0.5 * w_bay * n_bays))
    return out


def stage1_line_load(i_girder):
    """q(x) carried by the precast girder alone: self-weight + wet deck slab."""
    slab = C.GAMMA_CONC * C.T_DECK * S.deck_width(i_girder)
    return lambda x: girder_self_weight(x) + slab


def ss_beam(q, point_loads=(), xs=None, a=C.END_OVERHANG, span=C.L, n=4001):
    """Moment and shear of a beam on supports at 0 and span with overhangs a at both ends.

    q: callable line load (downward positive) over [-a, span + a]; point_loads: [(x, P)].
    Returns xs, M(xs) (sagging +), V(xs), reactions (RA, RB).  V is taken just right of x
    in the left half and just left of x in the right half, so that it is the support-side
    shear at both bearings.
    """
    xg = np.linspace(-a, span + a, n)
    qg = np.array([q(x) for x in xg])
    Q = np.concatenate([[0.0], np.cumsum(0.5 * (qg[1:] + qg[:-1]) * np.diff(xg))])   # load from left end
    QM = np.concatenate([[0.0], np.cumsum(0.5 * (qg[1:] * xg[1:] + qg[:-1] * xg[:-1]) * np.diff(xg))])
    total = Q[-1] + sum(P for _, P in point_loads)
    moment_about_A = QM[-1] + sum(P * x for x, P in point_loads)
    RB = moment_about_A / span
    RA = total - RB
    xs = np.linspace(0, span, 45) if xs is None else np.asarray(xs, float)
    M, V = [], []
    for x in xs:
        Qx = np.interp(x, xg, Q)
        QMx = np.interp(x, xg, QM)
        m = RA * (x - 0.0) - (Qx * x - QMx)       # moment of the distributed load left of x about x
        v = RA - Qx
        for xp, P in point_loads:
            if xp < x - 1e-9 or (abs(xp - x) <= 1e-9 and x < span / 2):
                m -= P * (x - xp)
                v -= P
        M.append(m)
        V.append(v)
    return xs, np.array(M), np.array(V), (RA, RB)
