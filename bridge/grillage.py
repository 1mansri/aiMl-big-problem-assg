"""Plane grillage (grid) finite-element model.

Node DOFs (global, z up): w (vertical displacement), rX, rY (rotation vector components).
Members are prismatic grid elements with bending EI about their local y axis and
St Venant torsion GJ about their own axis.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import config as C
from . import section as S
from . import loads


@dataclass
class Member:
    n1: int
    n2: int
    EI: float
    GJ: float
    kind: str = ""
    tag: tuple = ()
    mass: float = 0.0     # t/m, for modal analysis


class Grid:
    def __init__(self):
        self.xy = []
        self.members = []
        self.fixed = set()          # global dof indices with w = 0
        self.springs = {}           # dof -> stiffness (kN/m)
        self.point_mass = {}        # node -> t
        self._fact = None

    # ---------------------------------------------------------------- building
    def add_node(self, x, y):
        self.xy.append((x, y))
        return len(self.xy) - 1

    def add_member(self, *args, **kw):
        self.members.append(Member(*args, **kw))
        self._fact = None

    @property
    def ndof(self):
        return 3 * len(self.xy)

    def member_geometry(self, m):
        (x1, y1), (x2, y2) = self.xy[m.n1], self.xy[m.n2]
        Lm = np.hypot(x2 - x1, y2 - y1)
        return Lm, (x2 - x1) / Lm, (y2 - y1) / Lm

    def local_stiffness(self, m):
        """6x6 in local order [w1, rx1, ry1, w2, rx2, ry2]; ry = -dw/dx."""
        Lm, _, _ = self.member_geometry(m)
        a = m.EI / Lm ** 3
        # bending in (w, phi = dw/dx), then phi -> -ry flips the w-phi coupling terms
        kb = a * np.array([[12, 6 * Lm, -12, 6 * Lm],
                           [6 * Lm, 4 * Lm ** 2, -6 * Lm, 2 * Lm ** 2],
                           [-12, -6 * Lm, 12, -6 * Lm],
                           [6 * Lm, 2 * Lm ** 2, -6 * Lm, 4 * Lm ** 2]])
        sgn = np.array([1, -1, 1, -1])
        kb = kb * np.outer(sgn, sgn)
        k = np.zeros((6, 6))
        ib = [0, 2, 3, 5]
        k[np.ix_(ib, ib)] = kb
        kt = m.GJ / Lm
        k[np.ix_([1, 4], [1, 4])] = kt * np.array([[1, -1], [-1, 1]])
        return k

    def transform(self, m):
        _, c, s = self.member_geometry(m)
        t = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        T = np.zeros((6, 6))
        T[:3, :3] = t
        T[3:, 3:] = t
        return T

    def dofs(self, m):
        return [3 * m.n1, 3 * m.n1 + 1, 3 * m.n1 + 2, 3 * m.n2, 3 * m.n2 + 1, 3 * m.n2 + 2]

    def stiffness(self):
        K = np.zeros((self.ndof, self.ndof))
        for m in self.members:
            T = self.transform(m)
            d = self.dofs(m)
            K[np.ix_(d, d)] += T.T @ self.local_stiffness(m) @ T
        for d, k in self.springs.items():
            K[d, d] += k
        return K

    def free(self):
        return np.array([d for d in range(self.ndof) if d not in self.fixed])

    def factor(self):
        if self._fact is None:
            K = self.stiffness()
            f = self.free()
            self._K = K
            self._f = f
            self._fact = cho_factor(K[np.ix_(f, f)])
        return self._fact

    # ---------------------------------------------------------------- solving
    def solve(self, F):
        """Displacements for nodal load vector(s) F (ndof,) or (ndof, n)."""
        fact = self.factor()
        u = np.zeros_like(F, dtype=float)
        u[self._f] = cho_solve(fact, F[self._f])
        return u

    def sensitivity(self, c):
        """s with response = s . F for any nodal load F, for response = c . u (c over all dofs)."""
        fact = self.factor()
        s = np.zeros(self.ndof)
        s[self._f] = cho_solve(fact, np.asarray(c, float)[self._f])
        return s

    def member_end_forces_row(self, m):
        """Matrix R (6 x ndof) with local end forces f = R u."""
        R = np.zeros((6, self.ndof))
        R[:, self.dofs(m)] = self.local_stiffness(m) @ self.transform(m)
        return R

    def moment_row(self, m, end):
        """Row c with sagging bending moment at member end (0 or 1) = c . u."""
        R = self.member_end_forces_row(m)
        return R[2] if end == 0 else -R[5]

    def shear_row(self, m):
        """Row c with member shear (positive = upward force of node 1 on the member) = c . u."""
        return self.member_end_forces_row(m)[0]

    def reaction_sensitivity(self, dof):
        """s with support reaction (upward) at a fixed w dof = s . F."""
        fact = self.factor()
        K, f = self._K, self._f
        s = np.zeros(self.ndof)
        s[f] = cho_solve(fact, K[f, dof])
        s[dof] = -1.0
        return s


# ====================================================================== bridge model
@dataclass
class BridgeGrillage:
    grid: Grid
    xs: np.ndarray          # station x
    ys: np.ndarray          # girder line y
    node: np.ndarray        # node[i_station, i_girder]
    long: dict              # (i_station, g) -> member index of element starting at station i
    supports: list          # w dofs at bearings, [(g, end, dof)]

    def w_dof(self, i, g):
        return 3 * self.node[i, g]

    def rx_dof(self, i, g):
        return 3 * self.node[i, g] + 1


def _cross_girder_EI_GJ(tw, d, slab_w):
    """In-situ T cross girder (deck concrete) with slab flange slab_w."""
    E = C.DECK_CONCRETE["Ecm"] * 1e3
    G = E / (2 * (1 + C.POISSON))
    p = S.props([(tw, tw, d), (slab_w, slab_w, C.T_DECK)])
    J = (1 - 0.63 * tw / d) * d * tw ** 3 / 3 + slab_w * C.T_DECK ** 3 / 6
    return E * p.I, G * J, p


def build(nx=44, bearing_kv=None, E_factor=1.0, barrier_EI=0.0):
    """Grillage of the 33 m deck: N_GIRDERS longitudinal lines, transverse members at every station.

    nx: number of longitudinal elements (44 -> 0.75 m).
    bearing_kv: vertical bearing spring (kN/m) instead of rigid supports.
    E_factor: multiplier on all concrete moduli (e.g. dynamic modulus for the twin).
    barrier_EI: extra bending stiffness added to the outer girders (kN m^2) for composite barriers.
    """
    g = Grid()
    xs = np.linspace(0, C.L, nx + 1)
    ys = np.array(C.GIRDER_Y)
    ng = len(ys)
    node = np.array([[g.add_node(x, y) for y in ys] for x in xs])
    Eg = C.GIRDER_CONCRETE["Ecm"] * 1e3 * E_factor
    Gg = Eg / (2 * (1 + C.POISSON))
    Ed = C.DECK_CONCRETE["Ecm"] * 1e3 * E_factor
    Gd = Ed / (2 * (1 + C.POISSON))
    long = {}
    for j in range(ng):
        for i in range(nx):
            xm = 0.5 * (xs[i] + xs[i + 1])
            web = loads.girder_web(xm)
            p = S.props(S.composite_layers(j, web))
            J = S.torsion_constant_composite(j)
            EI = Eg * p.I + (barrier_EI if j in (0, ng - 1) else 0.0)
            mass = (S.props(S.girder_layers(web)).A + C.T_DECK * S.deck_width(j)) * C.GAMMA_CONC / 9.81
            long[(i, j)] = len(g.members)
            g.add_member(node[i, j], node[i + 1, j], EI, Gg * J, "girder", (i, j), mass)
    dx = C.L / nx
    cg = {round(x, 6): (tw, d) for x, tw, d in C.CROSS_GIRDERS}
    for i, x in enumerate(xs):
        w = dx if 0 < i < nx else dx / 2 + C.END_OVERHANG
        EI = Ed * w * C.T_DECK ** 3 / 12
        GJ = Gd * w * C.T_DECK ** 3 / 6
        kind = "slab"
        key = round(x, 6)
        if key in cg:
            tw, d = cg[key]
            EIc, GJc, _ = _cross_girder_EI_GJ(tw, d, w)
            EI, GJ, kind = EIc * E_factor, GJc * E_factor, "cross_girder"
        for j in range(ng - 1):
            g.add_member(node[i, j], node[i, j + 1], EI, GJ, kind, (i, j))
    supports = []
    for j in range(ng):
        for end, i in ((0, 0), (1, nx)):
            d = 3 * node[i, j]
            if bearing_kv is None:
                g.fixed.add(d)
            else:
                g.springs[d] = bearing_kv
            supports.append((j, end, d))
    return BridgeGrillage(g, xs, ys, node, long, supports)


# ====================================================================== loading helpers
def lever_weights(y, ys):
    """Nodal force weights and torque for a unit downward load at transverse position y.

    Returns list of (girder index, force weight, torque arm e): load P gives a downward
    force weight*P at that girder line and, if e != 0, a torque from offset e.
    """
    ys = np.asarray(ys)
    if y <= ys[0]:
        return [(0, 1.0, y - ys[0])]
    if y >= ys[-1]:
        return [(len(ys) - 1, 1.0, y - ys[-1])]
    j = int(np.searchsorted(ys, y)) - 1
    f = (y - ys[j]) / (ys[j + 1] - ys[j])
    return [(j, 1 - f, 0.0), (j + 1, f, 0.0)]


def point_load_vector(bg, x, y, P):
    """Nodal load vector for a downward point load P at (x, y) (lever rule in x and y)."""
    F = np.zeros(bg.grid.ndof)
    xs = bg.xs
    if x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        return F
    i = min(int(np.searchsorted(xs, x, side="right")) - 1, len(xs) - 2)
    fx = (x - xs[i]) / (xs[i + 1] - xs[i])
    for ii, wx in ((i, 1 - fx), (i + 1, fx)):
        for j, wy, e in lever_weights(y, bg.ys):
            F[bg.w_dof(ii, j)] -= P * wx * wy
            F[bg.rx_dof(ii, j)] -= P * wx * wy * e      # Mx = -P e for downward P at offset e
    return F


def line_load_vector(bg, y, q, x0=0.0, x1=None):
    """Nodal loads for a longitudinal line load q (kN/m, downward) at transverse position y."""
    x1 = C.L if x1 is None else x1
    F = np.zeros(bg.grid.ndof)
    xs = bg.xs
    for i in range(len(xs) - 1):
        a, b = max(xs[i], x0), min(xs[i + 1], x1)
        if b <= a:
            continue
        # consistent lumping of the uniform segment [a, b] onto the element end nodes
        h = xs[i + 1] - xs[i]
        r1 = q * ((xs[i + 1] - a) ** 2 - (xs[i + 1] - b) ** 2) / (2 * h)
        r2 = q * (b - a) - r1
        for ii, P in ((i, r1), (i + 1, r2)):
            for j, wy, e in lever_weights(y, bg.ys):
                F[bg.w_dof(ii, j)] -= P * wy
                F[bg.rx_dof(ii, j)] -= P * wy * e
    return F


def area_load_vector(bg, y0, y1, p, x0=0.0, x1=None, dy=0.05):
    """Uniform pressure p (kPa) over the strip y0..y1, as line loads every dy."""
    n = max(1, int(round((y1 - y0) / dy)))
    w = (y1 - y0) / n
    F = np.zeros(bg.grid.ndof)
    for k in range(n):
        F += line_load_vector(bg, y0 + (k + 0.5) * w, p * w, x0, x1)
    return F


# ====================================================================== responses
def girder_moment_sensitivities(bg):
    """s[g][i]: sagging moment of girder g at station i (interior stations), = s . F."""
    nx = len(bg.xs) - 1
    out = {}
    for j in range(len(bg.ys)):
        for i in range(1, nx):
            m = bg.grid.members[bg.long[(i - 1, j)]]
            out[(j, i)] = bg.grid.sensitivity(bg.grid.moment_row(m, 1))
    return out


def girder_shear_sensitivities(bg):
    """s[g][i]: shear in girder element i (between stations i and i+1) = s . F."""
    out = {}
    for (i, j), mi in bg.long.items():
        out[(j, i)] = bg.grid.sensitivity(bg.grid.shear_row(bg.grid.members[mi]))
    return out


def girder_deflection_sensitivity(bg, i, j):
    """Downward deflection of girder j at station i = s . F."""
    c = np.zeros(bg.grid.ndof)
    c[bg.w_dof(i, j)] = -1.0
    return bg.grid.sensitivity(c)
