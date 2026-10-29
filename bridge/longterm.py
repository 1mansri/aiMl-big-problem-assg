"""Long-term reference curves: embedded vibrating-wire strain and mid-span camber vs. time.

Linear creep superposition: every stress (or curvature) increment applied at age t_k contributes
increment / E(t_k) * (1 + phi(t, t_k)); shrinkage adds directly to the strain.  Prestress losses are
applied as a sequence of small increments over time.  Compression / shortening / upward camber are
positive in the strain curve; deflection is positive downward.
"""
import numpy as np

from . import config as C
from . import design as D
from . import prestress as PS

GAUGE_Y = 0.20          # VW gauge height above the soffit (bottom flange, between the ducts)


def _sec_stress(p, N, M, y):
    return (N / p.A + M * (y - p.yb) / p.I) * 1e-3


def history(eff, xs, times, j=2, extra_loss=None, gauge_y=GAUGE_Y):
    """Strain at the gauge (mid-span) and mid-span deflection at the given ages (days).

    eff: load effects of girder j at stations xs (as in run_step2_design.load_effects).
    extra_loss: optional callable t -> additional fractional prestress loss (deterioration).
    """
    fg = C.GIRDER_CONCRETE["fck"]
    h0 = D.girder_h0_mm()
    fh = D.force_history(xs, eff["M_dl1"], eff["M_sidl"] + eff["M_wc"], j)
    k = len(xs) // 2
    pp = [D.precast(x) for x in xs]
    pc = [D.composite(x, j) for x in xs]
    yb_p = np.array([p.yb for p in pp])
    yb_c = np.array([p.yb for p in pc])
    I_p = np.array([p.I for p in pp])
    I_c = np.array([p.I for p in pc])
    y1, y2 = fh.y(1), fh.y(2)
    mbar = np.where(xs <= C.L / 2, xs / 2, (C.L - xs) / 2)

    events = []          # (age, stress at gauge (MPa), sagging curvature*E (kN m^2 -> moment/I))

    def add(t, N_p, M_p, N_c, M_c):
        """Precast (N_p, M_p) and composite (N_c, M_c) actions as arrays along x."""
        s = _sec_stress(pp[k], N_p[k], M_p[k], gauge_y) + _sec_stress(pc[k], N_c[k], M_c[k], gauge_y)
        kap = (M_p / I_p + M_c / I_c) * 1e-3           # MPa per m  ->  divide by E (MPa) for curvature
        events.append((t, s, kap))

    z = np.zeros_like(xs)
    P1 = lambda t: fh.force(1, t)
    # stage-1 stressing with girder self-weight
    add(C.T_STAGE1, P1(C.T_STAGE1), eff["M_sw"] - P1(C.T_STAGE1) * (yb_p - y1), z, z)
    add(C.T_DECK_CAST, z, eff["M_dl1"] - eff["M_sw"], z, z)
    P2 = fh.force(2, C.T_STAGE2)
    dP1 = -fh.area(1) * fh.es2 * 1e3
    add(C.T_STAGE2, z, z, P2 + dP1, -P2 * (yb_c - y2) - dP1 * (yb_c - y1))
    add(C.T_SIDL, z, z, z, eff["M_sidl"] + eff["M_wc"])
    # long-term losses in small increments on a log time grid
    grid = np.unique(np.concatenate([np.geomspace(C.T_STAGE1 + 1, C.T_FINAL, 60), [C.T_DECK_CAST, C.T_STAGE2]]))
    prev1, prev2 = 0.0, 0.0
    for t in grid:
        g1 = PS.creep_coefficient(fg, h0, C.T_STAGE1, t) / fh.phi[1]
        g2 = PS.creep_coefficient(fg, h0, C.T_STAGE2, t) / fh.phi[2] if t > C.T_STAGE2 else 0.0
        L1 = -fh.area(1) * fh.lt[1] * 1e3 * g1
        L2 = -fh.area(2) * fh.lt[2] * 1e3 * g2
        d1, d2 = L1 - prev1, L2 - prev2
        prev1, prev2 = L1, L2
        if t <= C.T_DECK_CAST:
            add(t, d1, -d1 * (yb_p - y1), z, z)
        else:
            add(t, z, z, d1 + d2, -d1 * (yb_c - y1) - d2 * (yb_c - y2))
    if extra_loss is not None:
        # additional loss (e.g. corrosion of strands) as fraction of the current force of both groups
        prev = 0.0
        for t in grid:
            f = extra_loss(t)
            df = f - prev
            prev = f
            if abs(df) > 0:
                dP1 = -df * fh.force(1, C.T_FINAL)
                dP2 = -df * fh.force(2, C.T_FINAL)
                add(t, z, z, dP1 + dP2, -dP1 * (yb_c - y1) - dP2 * (yb_c - y2))

    strain, defl = [], []
    for t in times:
        e = PS.shrinkage(fg, h0, t)
        d = 0.0
        for tk, s, kap in events:
            if t < tk:
                continue
            E = PS.ecm_t(fg, tk)
            cf = 1 + PS.creep_coefficient(fg, h0, tk, t)
            e += s / E * cf
            d += np.trapezoid(kap / E * mbar, xs) * cf * 1e3          # mm, + down
        strain.append(e)
        defl.append(d)
    return np.array(strain), np.array(defl)
