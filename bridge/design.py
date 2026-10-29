"""Prestress losses, stage stresses, SLS / ULS checks of the PSC girder.

Stresses in MPa, compression positive.  Forces in kN, moments in kNm, lengths in m.
Composite-section stresses are in girder-concrete units; deck fibres are multiplied by the
modular ratio m = E_deck / E_girder.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import config as C
from . import loads
from . import prestress as PS
from . import section as S

FIBRES_PRECAST = ("girder_bottom", "girder_top")
FIBRES_COMPOSITE = ("girder_bottom", "girder_top", "deck_top")


# ====================================================================== sections
def net_props(p, holes):
    """Remove holes [(y, area)] from section p."""
    A = p.A - sum(a for _, a in holes)
    yb = (p.A * p.yb - sum(a * y for y, a in holes)) / A
    I = p.I + p.A * (p.yb - yb) ** 2 - sum(a * (y - yb) ** 2 for y, a in holes)
    return S.Props(A, yb, I, p.h)


def precast(x, holes=()):
    p = S.props(S.girder_layers(loads.girder_web(x)))
    return net_props(p, holes) if holes else p


def composite(x, j):
    return S.props(S.composite_layers(j, loads.girder_web(x)))


def fibre_y(fibre):
    return {"girder_bottom": 0.0, "girder_top": C.H_GIRDER, "deck_top": C.H_COMPOSITE}[fibre]


def stress(p, N, M, fibre, m=1.0):
    """Fibre stress for axial N (kN, compression +) and sagging moment M (kNm) about p's centroid."""
    y = fibre_y(fibre)
    s = (N / p.A + M * (y - p.yb) / p.I) * 1e-3
    return s * (m if fibre == "deck_top" else 1.0)


def prestress_stress(p, P, y_p, fibre, m=1.0):
    """Stress from a compressive tendon force P (kN) acting at height y_p."""
    return stress(p, P, -P * (p.yb - y_p), fibre, m)


# ====================================================================== prestress forces
@dataclass
class ForceHistory:
    x: np.ndarray
    sigma_slip: dict          # cable -> stress after friction + slip
    es1: np.ndarray           # elastic shortening of stage-1 cables at stage-1 stressing
    es2: np.ndarray           # elastic shortening of stage-1 cables due to stage-2 cable
    lt: dict                  # cable stage -> total long-term loss (MPa)
    relax: dict
    phi: dict
    eps_cs: dict
    g: dict                   # (stage, t) -> fraction of long-term loss developed at age t

    def area(self, stage):
        return sum(c.area for c in PS.stage_cables(stage))

    def y(self, stage, x=None):
        x = self.x if x is None else x
        cs = PS.stage_cables(stage)
        return sum(c.area * c.y(x) for c in cs) / sum(c.area for c in cs)

    def sigma0(self, stage):
        cs = PS.stage_cables(stage)
        return sum(c.area * self.sigma_slip[c.name] for c in cs) / self.area(stage)

    def sigma(self, stage, t):
        """Mean stress of a cable group at girder age t (after its stressing)."""
        if stage == 1:
            s = self.sigma0(1) - self.es1
            if t >= C.T_STAGE2:
                s = s - self.es2
            return s - self.lt[1] * self.g[(1, t)]
        s = self.sigma0(2)
        return s - self.lt[2] * self.g[(2, t)]

    def force(self, stage, t):
        return self.area(stage) * self.sigma(stage, t) * 1e3        # kN


def girder_h0_mm():
    """Notional size 2 Ac / u of the precast girder (perimeter of the layers)."""
    per = 0.0
    lay = S.girder_layers()
    per += lay[0][0] + lay[-1][1]                         # soffit and top of flange
    for b1, b2, h in lay:
        per += 2 * math.hypot(h, (b1 - b2) / 2)
    return 2 * S.props(lay).A / per * 1e3


def deck_h0_mm():
    exposed = C.GIRDER_SPACING + (C.GIRDER_SPACING - C.GIRDER_LAYERS[-1][0])
    return 2 * C.T_DECK * C.GIRDER_SPACING / exposed * 1e3


def force_history(xs, M_dl1, M_perm_comp, j=0):
    """Prestress stresses at stations xs.

    M_dl1: stage-1 dead-load moment on the precast girder at xs (for elastic shortening and the
    quasi-permanent concrete stress); M_perm_comp: SIDL + wearing coat moment on the composite.
    """
    fg = C.GIRDER_CONCRETE["fck"]
    sig = {}
    for c in PS.cables():
        xh, _, sa, _ = PS.stress_after_friction_slip(c)
        sig[c.name] = PS.half_to_full(xh, sa, xs)
    holes = lambda x: [(c.y(x), math.pi * C.DUCT_OD ** 2 / 4) for c in PS.cables()]
    A1 = sum(c.area for c in PS.stage_cables(1))
    A2 = sum(c.area for c in PS.stage_cables(2))
    y1 = sum(c.area * c.y(xs) for c in PS.stage_cables(1)) / A1
    y2 = sum(c.area * c.y(xs) for c in PS.stage_cables(2)) / A2
    s1 = sum(c.area * sig[c.name] for c in PS.stage_cables(1)) / A1
    s2 = sum(c.area * sig[c.name] for c in PS.stage_cables(2)) / A2
    n1 = len(PS.stage_cables(1))
    Ep = C.STRAND_EP
    es1, es2, sc1, sc2 = [], [], [], []
    for k, x in enumerate(xs):
        pn = precast(x, holes(x))
        P1 = A1 * s1[k] * 1e3
        # concrete stress at the stage-1 tendon centroid: prestress + girder self-weight
        Msw = loads.ss_beam(loads.girder_self_weight, xs=[x])[1][0]
        sc = (P1 / pn.A + (Msw - P1 * (pn.yb - y1[k])) * (y1[k] - pn.yb) / pn.I) * 1e-3
        es1.append((n1 - 1) / (2 * n1) * Ep / PS.ecm_t(fg, C.T_STAGE1) * sc)
        pc = composite(x, j)
        P2 = A2 * s2[k] * 1e3
        sc_2at1 = (P2 / pc.A - P2 * (pc.yb - y2[k]) * (y1[k] - pc.yb) / pc.I) * 1e-3
        es2.append(Ep / PS.ecm_t(fg, C.T_STAGE2) * sc_2at1)
        # quasi-permanent concrete stress at each group centroid (final, before long-term loss)
        pp = precast(x)
        P1f = A1 * (s1[k] - es1[-1] - es2[-1]) * 1e3
        def at(y):
            v = (P1f / pp.A + (M_dl1[k] - P1f * (pp.yb - y1[k])) * (y - pp.yb) / pp.I) * 1e-3
            v += (P2 / pc.A + (M_perm_comp[k] - P2 * (pc.yb - y2[k])) * (y - pc.yb) / pc.I) * 1e-3
            return v
        sc1.append(at(y1[k]))
        sc2.append(at(y2[k]))
    es1, es2, sc1, sc2 = map(np.array, (es1, es2, sc1, sc2))

    h0g = girder_h0_mm()
    Ecm = C.GIRDER_CONCRETE["Ecm"]
    lt, relax, phi, eps = {}, {}, {}, {}
    for stage, t0, sp, scq, yg, A_p, sec in ((1, C.T_STAGE1, s1 - es1, sc1, y1, A1, precast),
                                              (2, C.T_STAGE2, s2, sc2, y2, A2, lambda x: composite(x, j))):
        phi[stage] = PS.creep_coefficient(fg, h0g, t0, C.T_FINAL)
        eps[stage] = PS.shrinkage(fg, h0g, C.T_FINAL) - PS.shrinkage(fg, h0g, t0)
        vals, rel = [], []
        for k, x in enumerate(xs):
            p = sec(x)
            v, r = PS.time_dependent_loss(sp[k], scq[k], phi[stage], eps[stage], p.A, p.I,
                                          p.yb - yg[k], A_p, Ecm)
            vals.append(v)
            rel.append(r)
        lt[stage] = np.array(vals)
        relax[stage] = np.array(rel)
    g = {}
    for t in (C.T_STAGE1, C.T_DECK_CAST, C.T_STAGE2, C.T_SIDL, C.T_FINAL):
        g[(1, t)] = PS.creep_coefficient(fg, h0g, C.T_STAGE1, t) / phi[1] if t > C.T_STAGE1 else 0.0
        g[(2, t)] = PS.creep_coefficient(fg, h0g, C.T_STAGE2, t) / phi[2] if t > C.T_STAGE2 else 0.0
    return ForceHistory(np.asarray(xs), sig, es1, es2, lt, relax, phi, eps, g)


# ====================================================================== secondary actions
def temperature_profiles(h=C.H_COMPOSITE):
    """IRC:6 cl. 215.3 (Fig. 10) temperature differences: T(depth from top) for heating/cooling."""
    def heating(d):
        top = np.interp(d, [0, 0.15, 0.40], [17.8, 4.0, 0.0], right=0.0)
        bot = np.interp(h - d, [0, 0.15], [2.1, 0.0], right=0.0)
        return top + bot

    h1 = min(0.20 * h, 0.25)
    h2 = min(0.25 * h, 0.20)

    def cooling(d):
        top = np.interp(d, [0, h1, h1 + h2], [-10.6, -0.7, 0.0], right=0.0)
        bot = np.interp(h - d, [0, h1, h1 + h2], [-6.6, -0.8, 0.0], right=0.0)
        return top + bot
    return {"heating": heating, "cooling": cooling}


def _strips(j, x, n=2020):
    """Fibre discretisation of the composite section: heights, widths, moduli (MPa)."""
    y = (np.arange(n) + 0.5) * C.H_COMPOSITE / n
    dy = C.H_COMPOSITE / n
    b = np.zeros(n)
    E = np.zeros(n)
    z = 0.0
    for b1, b2, hh in S.girder_layers(loads.girder_web(x)):
        m = (y >= z) & (y < z + hh)
        b[m] = b1 + (b2 - b1) * (y[m] - z) / hh
        E[m] = C.GIRDER_CONCRETE["Ecm"]
        z += hh
    m = y >= C.H_GIRDER
    b[m] = S.deck_width(j)
    E[m] = C.DECK_CONCRETE["Ecm"]
    return y, dy, b, E


def eigen_stress(j, x, free_strain):
    """Self-equilibrating stresses (compression +) for an imposed free strain profile eps(y)
    (positive = expansion) on a simply supported (statically determinate) composite section."""
    y, dy, b, E = _strips(j, x)
    ef = free_strain(y)
    # plane strain e0 + k (y - yr) that restores N = 0 and M = 0
    w = E * b * dy
    A = np.array([[w.sum(), (w * y).sum()], [(w * y).sum(), (w * y * y).sum()]])
    rhs = np.array([(w * ef).sum(), (w * ef * y).sum()])
    e0, k = np.linalg.solve(A, rhs)
    sig = E * (ef - e0 - k * y)                  # restrained part: compression where expansion is blocked
    pick = lambda yy: float(np.interp(yy, y, sig))
    return {"girder_bottom": pick(0.0), "girder_top": float(sig[np.searchsorted(y, C.H_GIRDER) - 1]),
            "deck_top": pick(C.H_COMPOSITE)}, (e0, k)


def temperature_stresses(j, x):
    out = {}
    for name, T in temperature_profiles().items():
        out[name], _ = eigen_stress(j, x, lambda y, T=T: C.ALPHA_T * T(C.H_COMPOSITE - y))
    return out


def differential_shrinkage(j, x):
    """Stresses from deck shrinkage relative to the girder, age-adjusted modulus (chi = 0.8)."""
    fd, fg = C.DECK_CONCRETE["fck"], C.GIRDER_CONCRETE["fck"]
    t_deck_end = C.T_FINAL - C.T_DECK_CAST
    eps_deck = PS.shrinkage(fd, deck_h0_mm(), t_deck_end)
    h0g = girder_h0_mm()
    eps_girder = PS.shrinkage(fg, h0g, C.T_FINAL) - PS.shrinkage(fg, h0g, C.T_DECK_CAST)
    d_eps = eps_deck - eps_girder
    phi_d = PS.creep_coefficient(fd, deck_h0_mm(), 7, t_deck_end)
    red = 1 / (1 + 0.8 * phi_d)
    # free strain: deck shortens by d_eps relative to the girder; creep relief via the reduced deck strain
    sig, _ = eigen_stress(j, x, lambda y: np.where(y >= C.H_GIRDER, -d_eps * red, 0.0))
    return sig, dict(eps_deck=eps_deck, eps_girder_after_deck=eps_girder, diff=d_eps, phi_deck=phi_d, factor=red)


# ====================================================================== ULS flexure
def parabola_rectangle(eps, fck):
    fcd = 0.67 * fck / 1.5
    e = np.clip(eps, 0, None)
    return np.where(e < 0.002, fcd * (1 - (1 - e / 0.002) ** 2), fcd)


def uls_moment_capacity(j, x, sigma_pe):
    """M_Rd (kNm) by strain compatibility; tendons with their effective prestrain.

    sigma_pe: {cable name: effective stress (MPa)}.  Untensioned bars are ignored.
    Returns M_Rd, neutral-axis depth from the deck top, tendon strains/stresses.
    """
    y, dy, b, E = _strips(j, x)
    fck = np.where(y >= C.H_GIRDER, C.DECK_CONCRETE["fck"], C.GIRDER_CONCRETE["fck"])
    cs = PS.cables()
    yp = np.array([c.y(x) for c in cs])
    Ap = np.array([c.area for c in cs])
    pre = np.array([sigma_pe[c.name] for c in cs]) / C.STRAND_EP
    fpd = 0.87 * C.STRAND_FPK / 1.15
    top = C.H_COMPOSITE

    def forces(xu):
        eps_c = 0.0035 * (y - (top - xu)) / xu                 # compression +
        Fc = (parabola_rectangle(eps_c, fck) * b * dy).sum()
        eps_p = pre + 0.0035 * ((top - xu) - yp) / xu          # added tensile strain below the NA
        sp = np.minimum(C.STRAND_EP * eps_p, fpd)
        Fp = (sp * Ap).sum()
        return Fc, Fp, eps_c, sp, eps_p

    lo, hi = 0.01, top
    for _ in range(80):
        xu = 0.5 * (lo + hi)
        Fc, Fp, *_ = forces(xu)
        if Fc > Fp:
            hi = xu
        else:
            lo = xu
    Fc, Fp, eps_c, sp, eps_p = forces(xu)
    sc = parabola_rectangle(eps_c, fck) * b * dy
    yc = (sc * y).sum() / sc.sum()
    ypc = (sp * Ap * yp).sum() / (sp * Ap).sum()
    M = Fp * (yc - ypc) * 1e3                                   # MN m -> kNm
    return M, xu, dict(eps_p=eps_p.tolist(), sigma_p=sp.tolist(), lever=yc - ypc, yielded=bool((sp >= fpd - 1e-6).all()))


# ====================================================================== shear (IRC:112 cl. 10.3)
def shear_check(x, V_Ed, P_m, slope_resultant, j=0, fywk=500.0, cot_theta=2.0):
    """Shear design at section x for V_Ed (kN, before the prestress component).

    P_m: total effective prestress (kN); slope_resultant: dy/dx of the resultant cable.
    Returns dict with V_Rd,c (uncracked), required link area per m, V_Rd,max.
    """
    fck = C.GIRDER_CONCRETE["fck"]
    bw = loads.girder_web(x)
    pc = composite(x, j)
    pp = precast(x)
    # ducts crossing the web at this section reduce the effective width (grouted: 0.5 phi)
    in_web = [c for c in PS.cables() if c.y(x) > C.GIRDER_LAYERS[0][2] + C.GIRDER_LAYERS[1][2]]
    bw_nom = bw - 0.5 * C.DUCT_OD * (1 if in_web else 0)
    Vp = 0.9 * P_m * abs(math.sin(math.atan(slope_resultant)))
    V = max(V_Ed - Vp, 0.0)
    fctd = 0.7 * C.GIRDER_CONCRETE["fctm"] / 1.5
    sig_cp = 0.9 * P_m / pp.A * 1e-3
    # first moment of area above the centroid of the precast section
    yb = pp.yb
    Sx = 0.0
    z = 0.0
    for b1, b2, hh in S.girder_layers(bw):
        for k in range(50):
            yy = z + (k + 0.5) * hh / 50
            if yy > yb:
                Sx += (b1 + (b2 - b1) * (k + 0.5) / 50) * hh / 50 * (yy - yb)
        z += hh
    V_rdc = pp.I * bw_nom / Sx * math.sqrt(fctd ** 2 + sig_cp * fctd) * 1e3
    d = C.H_COMPOSITE - min(c.y(x) for c in PS.cables())
    z_lev = 0.9 * d
    fcd = 0.67 * fck / 1.5
    nu1 = 0.6 * (1 - fck / 310)
    a_cw = 1 + sig_cp / fcd if sig_cp <= 0.25 * fcd else (1.25 if sig_cp <= 0.5 * fcd else 2.5 * (1 - sig_cp / fcd))
    V_rdmax = a_cw * bw_nom * z_lev * nu1 * fcd / (cot_theta + 1 / cot_theta) * 1e3
    fywd = 0.87 * fywk
    asw_req = V / (z_lev * fywd * cot_theta) * 1e-3 * 1e6       # mm^2 per m
    rho_min = 0.072 * math.sqrt(fck) / fywk
    asw_min = rho_min * bw * 1e6                                # mm^2 per m
    return dict(x=x, bw=bw, bw_nom=bw_nom, V_Ed=V_Ed, V_p=Vp, V_Ed_net=V, V_Rd_c=V_rdc, z=z_lev,
                V_Rd_max=V_rdmax, asw_req_mm2_per_m=asw_req, asw_min_mm2_per_m=asw_min,
                links_needed=V > V_rdc, asw_design_mm2_per_m=max(asw_min, asw_req))   # truss only, concrete term ignored


def interface_shear(V_Ed, x, j=0, c=0.45, mu=0.7, fyk=500.0, beta=1.0):
    """Deck-girder interface shear (IRC:112 cl. 10.3.4), rough surface; returns required steel."""
    fd = C.DECK_CONCRETE["fck"]
    bi = C.GIRDER_LAYERS[-1][0]
    d = C.H_COMPOSITE - min(cb.y(x) for cb in PS.cables())
    z = 0.9 * d
    v_edi = beta * V_Ed / (z * bi) * 1e-3                      # MPa
    fctd = 0.7 * C.DECK_CONCRETE["fctm"] / 1.5
    nu = 0.6 * (1 - fd / 310)
    fcd = 0.67 * fd / 1.5
    rho = max(0.0, (v_edi - c * fctd) / (0.87 * fyk * mu))     # alpha = 90 deg links
    return dict(v_Edi=v_edi, v_Rdi_concrete=c * fctd, v_Rdi_max=0.5 * nu * fcd,
                As_req_mm2_per_m=rho * bi * 1e6)
