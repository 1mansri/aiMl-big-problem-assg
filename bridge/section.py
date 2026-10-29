"""Section properties of the precast and composite girder sections.

A section is a list of trapezoidal layers stacked from the soffit:
(width at layer bottom, width at layer top, height).
"""
from dataclasses import dataclass

from . import config as C


@dataclass
class Props:
    A: float        # m^2
    yb: float       # centroid above soffit, m
    I: float        # second moment about the centroid, m^4
    h: float        # total depth, m

    @property
    def yt(self):
        return self.h - self.yb

    @property
    def Zb(self):
        return self.I / self.yb

    @property
    def Zt(self):
        return self.I / self.yt


def _trapezoid(b1, b2, h, y0):
    """Area, centroid height (absolute) and own-centroid I of one trapezoid layer."""
    A = 0.5 * (b1 + b2) * h
    yc = y0 + h * (b1 + 2 * b2) / (3 * (b1 + b2))
    I0 = h ** 3 * (b1 ** 2 + 4 * b1 * b2 + b2 ** 2) / (36 * (b1 + b2))
    return A, yc, I0


def props(layers):
    parts, y0 = [], 0.0
    for b1, b2, h in layers:
        parts.append(_trapezoid(b1, b2, h, y0))
        y0 += h
    A = sum(p[0] for p in parts)
    yb = sum(p[0] * p[1] for p in parts) / A
    I = sum(p[2] + p[0] * (p[1] - yb) ** 2 for p in parts)
    return Props(A, yb, I, y0)


def girder_layers(web=C.B_WEB):
    """Precast layers; a thicker web models the end block (tapers flatten out)."""
    if web >= C.GIRDER_LAYERS[0][0]:
        bf_b, _, hb = C.GIRDER_LAYERS[0]
        bf_t = C.GIRDER_LAYERS[-1][0]
        h_top_fl = C.GIRDER_LAYERS[-1][2]
        return [
            (web, web, C.H_GIRDER - h_top_fl - C.GIRDER_LAYERS[-2][2]),
            (web, bf_t, C.GIRDER_LAYERS[-2][2]),
            (bf_t, bf_t, h_top_fl),
        ]
    out = []
    for b1, b2, h in C.GIRDER_LAYERS:
        out.append((b1 if b1 != C.B_WEB else web, b2 if b2 != C.B_WEB else web, h))
    return out


def deck_width(i_girder):
    """Deck width attributed to girder i (full width effective, see IRC:112 cl. 7.6.1.2 check)."""
    if i_girder in (0, C.N_GIRDERS - 1):
        return C.CANTILEVER + C.GIRDER_SPACING / 2
    return C.GIRDER_SPACING


def modular_ratio():
    """Short-term ratio E_deck / E_girder used to transform the deck slab."""
    return C.DECK_CONCRETE["Ecm"] / C.GIRDER_CONCRETE["Ecm"]


def composite_layers(i_girder=1, web=C.B_WEB, m=None):
    m = modular_ratio() if m is None else m
    b = deck_width(i_girder) * m
    return girder_layers(web) + [(b, b, C.T_DECK)]


def effective_flange_check():
    """IRC:112 cl. 7.6.1.2: b_eff,i = 0.2 b_i + 0.1 l0 <= 0.2 l0 and <= b_i."""
    l0 = C.L
    rows = []
    for name, bi in (("interior half-bay", (C.GIRDER_SPACING - C.B_WEB) / 2),
                     ("cantilever", C.CANTILEVER - C.B_WEB / 2)):
        beff = min(0.2 * bi + 0.1 * l0, 0.2 * l0, bi)
        rows.append((name, bi, beff, beff >= bi - 1e-9))
    return rows


def torsion_constant_precast(web=C.B_WEB):
    """St Venant J of the I-girder as a sum of thick rectangles, k = (1 - 0.63 t/b)/3."""
    def rect(b, t):
        b, t = max(b, t), min(b, t)
        return (1 - 0.63 * t / b) * b * t ** 3 / 3

    (bb, _, hb), (_, _, htb), (_, _, hw), (_, _, htt), (bt, _, ht) = C.GIRDER_LAYERS
    t_bot = hb + htb / 2              # taper averaged into the flange
    t_top = ht + htt / 2
    return rect(bb, t_bot) + rect(web, C.H_GIRDER - t_bot - t_top) + rect(bt, t_top)


def torsion_constant_composite(i_girder=1):
    """Grillage longitudinal member: girder J + half of the slab's b t^3/3 (Hambly), in girder concrete."""
    return torsion_constant_precast() + modular_ratio() * deck_width(i_girder) * C.T_DECK ** 3 / 6
