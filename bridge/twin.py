"""As-built digital twin: grillage + mass, bearings as springs, damage hooks, modal analysis.

A State describes one condition of the bridge.  The healthy as-built state is State();
deterioration scenarios (step 5) change its fields.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from . import config as C
from . import grillage as G
from . import loads
from . import section as S

G_ACC = 9.81


def dynamic_modulus_factor(Ecm_MPa):
    """BS 8110-2 cl. 7.2: Ec = 1.25 Ed - 19 (GPa)  ->  Ed / Ec."""
    Ec = Ecm_MPa / 1e3
    return (Ec + 19) / 1.25 / Ec


# ---------------------------------------------------------------------- bearings (preliminary)
BEARING = dict(a=0.40, b=0.40, t_layer=0.012, n_layers=4, G=1.0, K_bulk=2000.0)


def bearing_stiffness(br=BEARING):
    """Laminated elastomeric bearing: vertical and shear stiffness (kN/m)."""
    A = br["a"] * br["b"]
    S_ = A / (2 * br["t_layer"] * (br["a"] + br["b"]))
    Ec = 6 * br["G"] * S_ ** 2
    Ec = 1 / (1 / Ec + 1 / br["K_bulk"])                    # MPa, with bulk compressibility
    Te = br["t_layer"] * br["n_layers"]
    return dict(shape_factor=S_, kv=Ec * A / Te * 1e3, kh=br["G"] * A / Te * 1e3, Te=Te)


# ---------------------------------------------------------------------- state
@dataclass
class State:
    E_factor_dyn: float = dynamic_modulus_factor(C.GIRDER_CONCRETE["Ecm"])
    temperature: float = 30.0                 # deck temperature, degC
    T_ref: float = 30.0
    dE_dT: float = -0.0035                    # relative change of E per degC (assumed)
    barrier_composite: float = 0.0            # fraction of barrier EI acting with the outer girders
    # (girder, station-range (i0, i1)) -> EI factor, e.g. {(2, (20, 24)): 0.7}
    girder_EI: dict = field(default_factory=dict)
    # (girder, end) -> vertical bearing stiffness factor
    bearing_kv: dict = field(default_factory=dict)
    # (girder, end) -> horizontal restraint (kN/m) at the bearing (seized bearing); default = elastomer shear
    bearing_kh: dict = field(default_factory=dict)
    added_mass: float = 0.0                   # t/m^2 over the whole deck (e.g. extra surfacing)

    def E_temperature_factor(self):
        return 1 + self.dE_dT * (self.temperature - self.T_ref)


BARRIER_EI = 34.3e6 * 0.025      # kN m^2: New Jersey barrier ~ 0.025 m^4 about its own axis (approximate)


def build(state=None, dynamic=True, nx=44):
    """Grillage of the given state; dynamic=True uses the dynamic modulus."""
    st = State() if state is None else state
    Ef = (st.E_factor_dyn if dynamic else 1.0) * st.E_temperature_factor()
    bg = G.build(nx=nx, E_factor=Ef, barrier_EI=st.barrier_composite * BARRIER_EI * Ef)
    g = bg.grid
    for (j, (i0, i1)), f in st.girder_EI.items():
        for i in range(i0, i1):
            g.members[bg.long[(i, j)]].EI *= f
    bs = bearing_stiffness()
    e_na = S.props(S.composite_layers(1)).yb                 # bearing level to composite centroid
    for j, end, d in bg.supports:
        g.fixed.discard(d)
        g.springs[d] = bs["kv"] * st.bearing_kv.get((j, end), 1.0)
        kh = st.bearing_kh.get((j, end), bs["kh"])
        node = bg.node[0 if end == 0 else len(bg.xs) - 1, j]
        g.springs[3 * node + 2] = g.springs.get(3 * node + 2, 0.0) + kh * e_na ** 2   # eccentric restraint -> end moment
    g._fact = None
    bg.state = st
    return bg


# ---------------------------------------------------------------------- mass
def mass_matrix(bg, state=None):
    """Lumped diagonal mass (t, t m^2) incl. SIDL; rotary inertia from the transverse deck strip."""
    st = getattr(bg, "state", State()) if state is None else state
    g = bg.grid
    m = np.zeros(g.ndof)
    xs, ys = bg.xs, bg.ys
    dx = np.diff(xs)
    trib = np.zeros(len(xs))
    trib[:-1] += dx / 2
    trib[1:] += dx / 2
    # girders + deck (member masses already include the deck slab width of each girder)
    for (i, j), mi in bg.long.items():
        mem = g.members[mi]
        half = mem.mass * dx[i] / 2
        for node in (mem.n1, mem.n2):
            m[3 * node] += half
            m[3 * node + 1] += half * (S.deck_width(j) ** 2) / 12
    # cross girders
    for x, tw, d in C.CROSS_GIRDERS:
        i = int(np.argmin(abs(xs - x)))
        for j in range(len(ys)):
            for x_, P in loads.cross_girder_point_loads(j):
                if abs(x_ - x) < 1e-6:
                    m[3 * bg.node[i, j]] += P / G_ACC
    # superimposed dead loads as line masses at their transverse positions
    lines = []
    for name, y0, y1 in C.strip_extents():
        yc = 0.5 * (y0 + y1)
        if name in C.SIDL_LINE:
            lines.append((yc, C.SIDL_LINE[name] / G_ACC))
        elif name == "footpath":
            lines.append((yc, C.FOOTPATH_FINISH * (y1 - y0) / G_ACC))
        elif name == "carriageway":
            n = 22
            w = (y1 - y0) / n
            lines += [(y0 + (k + 0.5) * w, (C.WEARING_COAT * w) / G_ACC) for k in range(n)]
    lines += [(yy, st.added_mass * C.B_DECK / 6) for yy in np.linspace(0, C.B_DECK, 6)]
    for yc, mu in lines:
        for j, wy, e in G.lever_weights(yc, ys):
            for i in range(len(xs)):
                mm = mu * trib[i] * wy
                m[3 * bg.node[i, j]] += mm
                m[3 * bg.node[i, j] + 1] += mm * e ** 2
    for i in range(len(xs)):
        for j in range(len(ys)):
            n = bg.node[i, j]
            m[3 * n + 2] += m[3 * n] * (trib[i] ** 2) / 12 + 1e-6
            m[3 * n + 1] += 1e-6
    return m


@dataclass
class Modes:
    freq: np.ndarray           # Hz
    shapes: np.ndarray         # (ndof, n) mass-normalised
    w_shapes: np.ndarray       # (n_stations, n_girders, n) vertical components
    labels: list


def classify(ws):
    """Label a vertical mode shape (stations x girders) as bending/torsion with half-wave count."""
    rel = ws[np.argmax(abs(ws).max(axis=1))]               # transverse shape at the peak station
    rel = rel / abs(rel).max()
    anti = 0.5 * (rel - rel[::-1])                           # antisymmetric about the deck centreline
    if abs(anti).sum() > 0.5 * abs(rel).sum():
        kind = "torsion"
    elif rel.min() * rel.max() > -0.05:                      # all girders move together
        kind = "bending"
    else:
        kind = "transverse"                                  # cross-section distortion (deck bending)
    # half-waves along the span from the dominant girder line
    line = ws[:, int(np.argmax(abs(ws).max(axis=0)))]
    sig = line[1:-1][abs(line[1:-1]) > 0.05 * abs(line).max()]
    k_long = int(np.sum(np.diff(np.sign(sig)) != 0)) + 1
    return f"{kind} {k_long}"


def modal(bg, n_modes=12, state=None):
    K = bg.grid.stiffness()
    M = mass_matrix(bg, state)
    f = bg.grid.free()
    lam, V = eigh(K[np.ix_(f, f)], np.diag(M[f]), subset_by_index=[0, n_modes - 1])
    shapes = np.zeros((bg.grid.ndof, n_modes))
    shapes[f] = V
    freq = np.sqrt(np.maximum(lam, 0)) / (2 * math.pi)
    ws = np.array([[[shapes[bg.w_dof(i, j), r] for r in range(n_modes)] for j in range(len(bg.ys))]
                   for i in range(len(bg.xs))])
    labels = [classify(ws[:, :, r]) for r in range(n_modes)]
    return Modes(freq, shapes, ws, labels)


def mac(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(abs(a @ b) ** 2 / ((a @ a) * (b @ b)))


# ---------------------------------------------------------------------- load test
TEST_TRUCK = dict(P=[60.0, 170.0, 170.0], offsets=[0.0, 3.8, 5.15], gauge=1.8)    # 400 kN loaded tipper
LOAD_TESTS = {"LT-1 eccentric (lanes 1-2)": [3.50, 6.50], "LT-2 central": [6.20, 9.20]}
LT_FRONT_AXLE_X = 20.30          # position of maximum mid-span moment found for the healthy bridge


def load_test(bg_static, trucks_y, x_front=None):
    """Two trucks side by side; mid-span deflection and soffit strain of every girder.

    x_front=None searches the front-axle position of maximum total mid-span moment.
    Strain uses each girder's actual (possibly damaged) member stiffness.
    """
    g = bg_static.grid
    i_mid = len(bg_static.xs) // 2
    best = None
    xs_try = np.arange(C.L / 2, C.L / 2 + 8.0, 0.05) if x_front is None else [x_front]
    for xf in xs_try:
        F = np.zeros(g.ndof)
        for yc in trucks_y:
            for P, o in zip(TEST_TRUCK["P"], TEST_TRUCK["offsets"]):
                for dy in (-TEST_TRUCK["gauge"] / 2, TEST_TRUCK["gauge"] / 2):
                    F += G.point_load_vector(bg_static, xf - o, yc + dy, P / 2)
        u = g.solve(F)
        M = [g.moment_row(g.members[bg_static.long[(i_mid - 1, j)]], 1) @ u for j in range(len(bg_static.ys))]
        if best is None or sum(M) > best[0]:
            best = (sum(M), xf, u, M)
    _, xf, u, M = best
    defl = [-u[bg_static.w_dof(i_mid, j)] * 1e3 for j in range(len(bg_static.ys))]
    strain = []
    for j, m in enumerate(M):
        yb = S.props(S.composite_layers(j)).yb
        EI = g.members[bg_static.long[(i_mid - 1, j)]].EI
        strain.append(m * yb / EI * 1e6)                       # microstrain, tension at the soffit
    return dict(front_axle_x=float(xf), deflection_mm=defl, soffit_strain_ue=strain, girder_moment_kNm=M,
                distribution=[d / sum(defl) for d in defl])
