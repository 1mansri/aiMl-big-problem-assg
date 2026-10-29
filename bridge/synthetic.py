"""Deterioration scenarios and synthetic inspection / NDT / monitoring campaigns.

A campaign is one visit of the bridge (age, season, deck temperature, traffic) in one condition.
Every measurement is produced by the same digital twin plus the measurement chain's errors, so the
synthetic data keep the physics of the 33 m bridge (and its blind spots).

Severity s in [0, 1] maps to a physical parameter of each mechanism (DAMAGE_RANGES); the ranges are
assumptions chosen to span "just detectable" to "severe but standing".
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import grillage as G
from . import inspection as I
from . import prestress as PS
from . import section as S
from . import twin as T
from . import vibration as V

KINDS = ["healthy", "prestress_loss", "girder_cracking", "concrete_deterioration",
         "bearing_seized", "bearing_degraded", "chloride_corrosion"]

DAMAGE_RANGES = {
    "prestress_loss": "loss of 3-30 % of the effective force of 1 girder (30 % of cases: 2 adjacent girders); "
                      "strand corrosion from grout voids / chloride. Cracks only when the soffit decompresses "
                      "under frequent load (margin ~3 MPa), then EI of the mid zone drops (to 0.5 at most).",
    "girder_cracking": "flexural cracking after an overload: EI of a 9-15 m zone around mid-span of 1 girder "
                       "x(1 - 0.45 s), i.e. down to 0.55 (partly cracked PSC); residual crack width 0.05-0.55 mm.",
    "concrete_deterioration": "distributed loss of concrete quality (e.g. poor compaction, ASR-type damage) in "
                              "1-3 adjacent girders: E x(1 - 0.30 s); UPV and rebound fall locally.",
    "bearing_seized": "1-6 free-end (P1) bearings restrained by debris / corrosion of the guides: horizontal "
                      "stiffness 3e4-1e6 kN/m (log scale) instead of 3.3e3 (elastomer shear).",
    "bearing_degraded": "1-2 elastomeric bearings split / aged: vertical stiffness x(1 - 0.85 s).",
    "chloride_corrosion": "expansion joint at A1 or P1 leaking chloride-laden water: corrosion activity up to s "
                          "in the end zones of the girders and deck below it; for s > 0.6 spalling and an end-zone "
                          "EI loss of up to 20 %.",
}

CRACK_MARGIN = 3.0        # MPa: frequent-load soffit stress margin to cracking (step 2: ~ -0.8 vs -3.3 fctm, + 0.1 Pm)


@dataclass
class Scenario:
    kind: str
    severity: float = 0.0
    girders: tuple = ()           # affected girders (0-based)
    end: int = 0                  # 0: A1, 1: P1 (joint / bearings)
    params: dict = field(default_factory=dict)


def sample_scenario(kind, rng, severity=None):
    s = float(rng.uniform(0.05, 1.0)) if severity is None else float(severity)
    if kind == "healthy":
        return Scenario(kind, 0.0)
    if kind == "prestress_loss":
        g = int(rng.integers(C.N_GIRDERS))
        gs = (g,) if rng.random() > 0.3 else tuple(sorted({g, min(g + 1, C.N_GIRDERS - 1) if g < C.N_GIRDERS - 1 else g - 1}))
        # cause: chloride reaching the strands (40 %) or water in grout voids (60 %)
        cause = "chloride" if rng.random() < 0.4 else "grout_void"
        return Scenario(kind, s, gs, params=dict(loss=0.03 + 0.27 * s, cause=cause))
    if kind == "girder_cracking":
        g = int(rng.integers(C.N_GIRDERS))
        half = float(rng.uniform(4.5, 7.5))
        return Scenario(kind, s, (g,), params=dict(EI=1 - 0.45 * s, half_zone=half, crack_mm=0.05 + 0.5 * s))
    if kind == "concrete_deterioration":
        g = int(rng.integers(C.N_GIRDERS))
        n = int(rng.integers(1, 4))
        gs = tuple(sorted({min(max(g + k, 0), C.N_GIRDERS - 1) for k in range(n)}))
        return Scenario(kind, s, gs, params=dict(E=1 - 0.30 * s))
    if kind == "bearing_seized":
        n = int(rng.integers(1, C.N_GIRDERS + 1))
        gs = tuple(sorted(rng.choice(C.N_GIRDERS, n, replace=False).tolist()))
        return Scenario(kind, s, gs, end=1, params=dict(kh=10 ** (4.5 + 1.5 * s)))
    if kind == "bearing_degraded":
        n = int(rng.integers(1, 3))
        gs = tuple(sorted(rng.choice(C.N_GIRDERS, n, replace=False).tolist()))
        return Scenario(kind, s, gs, end=int(rng.integers(2)), params=dict(kv=1 - 0.85 * s))
    if kind == "chloride_corrosion":
        return Scenario(kind, s, tuple(range(C.N_GIRDERS)), end=int(rng.integers(2)),
                        params=dict(c=s, EI_end=1 - 0.2 * max(0.0, (s - 0.6) / 0.4)))
    raise ValueError(kind)


# ====================================================================== prestress-loss response (static, linear)
def _tendon_eccentricity(x, j):
    cs = PS.cables()
    A = sum(c.area for c in cs)
    yp = sum(c.area * c.y(x) for c in cs) / A
    return S.props(S.composite_layers(j)).yb - yp


_PL_CACHE = {}


def prestress_loss_unit(bg_static, g):
    """Response to a 1000 kN loss of tendon force in girder g (composite, healthy, static).

    The loss acts like the tendon's equivalent loads with reversed sign: nodal couples
    m_i = dP * e_i on every element of girder g (internal sagging moment dP*e in a determinate beam);
    the grillage then shares it with the other girders.  Returns mid-span deflection (mm, + down) of
    every girder and the stress change (MPa, compression +) at the VW gauge height of every girder.
    """
    if g in _PL_CACHE:
        return _PL_CACHE[g]
    dP = 1000.0
    grid = bg_static.grid
    F = np.zeros(grid.ndof)
    xs = bg_static.xs
    for i in range(len(xs) - 1):
        m = dP * _tendon_eccentricity(0.5 * (xs[i] + xs[i + 1]), g)
        a, b = bg_static.node[i, g], bg_static.node[i + 1, g]
        F[3 * a + 2] += m            # ry is minus the slope dof: see Grid.local_stiffness
        F[3 * b + 2] -= m
    u = grid.solve(F)
    i_mid = len(xs) // 2
    defl, sig, sig_soffit = [], [], []
    for j in range(len(bg_static.ys)):
        defl.append(-u[bg_static.w_dof(i_mid, j)] * 1e3)
        M = grid.moment_row(grid.members[bg_static.long[(i_mid - 1, j)]], 1) @ u
        p = S.props(S.composite_layers(j))
        N = -dP if j == g else 0.0
        sig.append((N / p.A + M * (0.20 - p.yb) / p.I) * 1e-3)
        sig_soffit.append((N / p.A + M * (0.0 - p.yb) / p.I) * 1e-3)
    _PL_CACHE[g] = (np.array(defl), np.array(sig), np.array(sig_soffit))
    return _PL_CACHE[g]


def effective_prestress():
    """Final effective force per girder (kN), from the step-2 loss model at mid-span."""
    import json
    from pathlib import Path
    d = json.loads(Path("outputs/step2_design.json").read_text())
    return float(np.array(d["force_profile"]["P_final"])[len(d["force_profile"]["P_final"]) // 2])


# ====================================================================== state of the twin
def twin_state(scn, temperature, pl_crack=None):
    """twin.State for a scenario.  pl_crack: {girder: EI factor} from prestress-loss cracking."""
    st = T.State(temperature=temperature)
    nx = 44
    mid = nx // 2
    k = scn.kind
    p = scn.params
    if k == "girder_cracking":
        n = int(round(p["half_zone"] / (C.L / nx)))
        st.girder_EI[(scn.girders[0], (mid - n, mid + n))] = p["EI"]
    elif k == "concrete_deterioration":
        for g in scn.girders:
            st.girder_EI[(g, (0, nx))] = p["E"]
    elif k == "bearing_seized":
        for g in scn.girders:
            st.bearing_kh[(g, 1)] = p["kh"]
    elif k == "bearing_degraded":
        for g in scn.girders:
            st.bearing_kv[(g, scn.end)] = p["kv"]
    elif k == "chloride_corrosion" and p["EI_end"] < 1:
        rng_ = (0, 4) if scn.end == 0 else (nx - 4, nx)
        for g in scn.girders:
            st.girder_EI[(g, rng_)] = p["EI_end"]
    for g, f in (pl_crack or {}).items():
        st.girder_EI[(g, (mid - 8, mid + 8))] = f
    return st


# ====================================================================== environment
def environment(rng, age_years=None):
    age = float(rng.uniform(1, 30)) if age_years is None else age_years
    doy = float(rng.uniform(0, 365))
    hour = float(rng.uniform(0, 24))
    T_deck = 31 + 7 * math.sin(2 * math.pi * (doy - 105) / 365) + 5 * math.sin(2 * math.pi * (hour - 9) / 24) \
        + rng.normal(0, 1.5)
    wet = 152 <= doy <= 273                     # June - September monsoon
    traffic = float(np.interp(hour, [0, 5, 8, 12, 18, 21, 24], [3, 2, 15, 12, 16, 8, 3])) * rng.uniform(0.7, 1.3)
    return dict(age_years=age, doy=doy, hour=hour, T_deck=float(T_deck), wet=bool(wet), traffic_per_min=traffic)


# ====================================================================== NDT forward model
class AsBuilt:
    """Fixed as-built variability of the test points (same for every campaign of this bridge)."""

    def __init__(self, seed=2026):
        r = np.random.default_rng(seed)
        self.points = I.ndt_locations()
        n = len(self.points)
        girder = np.array([p["girder"] is not None for p in self.points])
        self.upv0 = np.where(girder, 4.60, 4.45) + r.normal(0, 0.08, n)
        self.reb0 = np.where(girder, 41.0, 37.0) + r.normal(0, 1.5, n)
        self.rho0 = np.where(girder, 35.0, 25.0) * np.exp(r.normal(0, 0.15, n))
        self.cover = np.where(girder, 55.0, 54.0) + r.normal(0, 4, n)
        self.girder = girder


def point_condition(scn, p):
    """(concrete damage d, corrosion activity c) at an NDT point for a scenario."""
    k, s = scn.kind, scn.severity
    g = (p["girder"] - 1) if p["girder"] else None
    z = p["zone"]
    end_zone = (z == "End-A1" and scn.end == 0) or (z == "End-P1" and scn.end == 1)
    near_end = (z == "L/4" and scn.end == 0) or (z == "3L/4" and scn.end == 1)
    d = c = 0.0
    if k == "concrete_deterioration" and g in scn.girders:
        d = s
    elif k == "girder_cracking" and g == scn.girders[0] and z in ("L/2", "L/4", "3L/4"):
        d = (0.25 if z == "L/2" else 0.08) * s
    elif (k == "prestress_loss" and scn.params.get("cause") == "chloride" and g in scn.girders
          and p["face"] == "soffit" and z in ("L/4", "L/2", "3L/4")):
        # HDPE ducts insulate the strands: the half-cell sees only the links / bottom bars that the same
        # chloride reaches.  Strand corrosion from water in grout voids leaves no surface signal at all.
        c = 0.10 + 0.30 * s
    elif k == "chloride_corrosion":
        if p["girder"] and end_zone:
            c = s
        elif p["girder"] and near_end:
            c = 0.3 * s
        elif not p["girder"] and ((z == "L/4" and scn.end == 0) or (z == "3L/4" and scn.end == 1)):
            c = 0.5 * s
        if end_zone and s > 0.6:
            d = 0.4 * (s - 0.6) / 0.4
    return d, c


def ndt_readings(scn, env, asb, rng):
    rows = []
    age, wet = env["age_years"], env["wet"]
    # campaign-level calibration of each survey: reference-electrode offset, moisture state, operator,
    # and the uncertain link between corrosion activity / damage and the reading
    hcp_off, hcp_k = rng.normal(0, 20), rng.uniform(0.75, 1.25)
    upv_off, upv_k = rng.normal(0, 0.03), rng.uniform(0.8, 1.2)
    reb_off = rng.normal(0, 1.0)
    rho_k = math.exp(rng.normal(0, 0.1))
    for k, p in enumerate(asb.points):
        d, c = point_condition(scn, p)
        upv = asb.upv0[k] * (1 - 0.30 * upv_k * d) + upv_off + rng.normal(0, 0.05)
        reb = asb.reb0[k] * (1 - 0.25 * d) + 0.4 * math.sqrt(age) + reb_off + rng.normal(0, 1.5)
        hcp = -110 - 380 * hcp_k * c - (40 if wet else 0) + hcp_off + rng.normal(0, 25)
        rho = asb.rho0[k] * (1 - 0.55 * c) * (0.6 if wet else 1.0) * rho_k * math.exp(rng.normal(0, 0.1))
        cover = asb.cover[k] + rng.normal(0, 2)
        carb = (1.5 if asb.girder[k] else 2.0) * math.sqrt(age) * math.exp(rng.normal(0, 0.15))
        chl = (0.04 + 0.5 * c ** 1.2 + rng.normal(0, 0.02)) if p["zone"].startswith("End") else np.nan
        rows.append(dict(point=p["id"], component=p["component"], girder=p["girder"], zone=p["zone"], face=p["face"],
                         upv_km_s=upv, rebound=reb, half_cell_mV=hcp, resistivity_kOhm_cm=rho, cover_mm=cover,
                         carbonation_mm=carb, chloride_pct=chl, true_d=d, true_c=c))
    return rows


def ndt_summary(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    gd = df[df.component == "PSC girders"]
    dk = df[df.component == "Deck slab"]
    end = gd[gd.zone.str.startswith("End")]
    mid = gd[~gd.zone.str.startswith("End")]
    return dict(
        upv_girder_mean=gd.upv_km_s.mean(), upv_girder_min=gd.upv_km_s.min(), upv_deck_min=dk.upv_km_s.min(),
        rebound_girder_mean=gd.rebound.mean(), rebound_girder_min=gd.rebound.min(),
        hcp_min=df.half_cell_mV.min(), hcp_mean=df.half_cell_mV.mean(),
        hcp_end_minus_mid=end.half_cell_mV.mean() - mid.half_cell_mV.mean(),
        hcp_soffit_mid_min=mid[mid.face == "soffit"].half_cell_mV.min(),
        resistivity_min=df.resistivity_kOhm_cm.min(), resistivity_median=df.resistivity_kOhm_cm.median(),
        chloride_end_max=end.chloride_pct.max(), cover_min=df.cover_mm.min(),
        carbonation_mean=df.carbonation_mm.mean(),
        n_upv_doubtful=int((df.upv_km_s < 3.75).sum()), n_hcp_high_risk=int((df.half_cell_mV < -350).sum()),
    )


# ====================================================================== visual inspection model
def _p(s, s50, w=0.08):
    return 1 / (1 + math.exp(-(s - s50) / w))


def visual_observations(scn, env, rng):
    """Observations an inspector records at a principal inspection (imperfect detection + benign defects)."""
    obs = []
    comps = I.components()
    k, s = scn.kind, scn.severity
    endn = "A1" if scn.end == 0 else "P1"
    zend = "End-" + endn

    def add(comp, loc, defect, inputs):
        obs.append(dict(component=comp, location=loc, defect=defect, inputs=inputs,
                        cs=I.condition_state(defect, comps[comp]["material"], inputs)))

    if k == "prestress_loss":
        if scn.params.get("cracked") and rng.random() < _p(scn.params.get("crack_mm", 0), 0.06, 0.03):
            for g in scn.girders:
                add("PSC girders", f"G{g+1}/L/2/bottom flange", "crack",
                    {"width_mm": round(scn.params["crack_mm"] * rng.uniform(0.8, 1.2), 2), "length_m": 0.5,
                     "orientation": "flexural (transverse, soffit)", "rust stain": False})
        # longitudinal cracking along a duct needs expansive corrosion near the surface (chloride cause)
        if rng.random() < _p(s, 0.55 if scn.params.get("cause") == "chloride" else 0.9):
            g = scn.girders[0]
            add("PSC girders", f"G{g+1}/L/2/bottom flange", "crack",
                {"width_mm": round(0.1 + 0.4 * s * rng.uniform(0.7, 1.3), 2), "length_m": round(1 + 3 * s, 1),
                 "orientation": "longitudinal along duct",
                 "rust stain": bool(s > 0.6 and scn.params.get("cause") == "chloride")})
    elif k == "girder_cracking":
        if rng.random() < _p(scn.params["crack_mm"], 0.08, 0.03):
            g = scn.girders[0]
            add("PSC girders", f"G{g+1}/L/2/bottom flange", "crack",
                {"width_mm": round(scn.params["crack_mm"] * rng.uniform(0.8, 1.2), 2),
                 "length_m": 0.6, "orientation": "flexural (transverse, soffit)", "rust stain": False})
            if s > 0.5 and rng.random() < 0.7:
                add("PSC girders", f"G{g+1}/L/2/web", "crack",
                    {"width_mm": round(0.6 * scn.params["crack_mm"], 2), "length_m": 0.8,
                     "orientation": "flexural (transverse, soffit)", "rust stain": False})
    elif k == "concrete_deterioration":
        for g in scn.girders:
            if rng.random() < _p(s, 0.45):
                add("PSC girders", f"G{g+1}/{rng.choice(['L/4', 'L/2', '3L/4'])}/web",
                    "surface deterioration", {"area_m2": round(0.5 + 3 * s, 1), "depth_mm": round(2 + 18 * s, 0)})
            if rng.random() < _p(s, 0.65):
                add("PSC girders", f"G{g+1}/{rng.choice(['L/4', 'L/2', '3L/4'])}/bottom flange",
                    "crack", {"width_mm": round(0.05 + 0.2 * s, 2), "length_m": 1.0, "orientation": "map / random",
                              "rust stain": False})
    elif k == "bearing_seized":
        for g in scn.girders:
            if rng.random() < _p(s, 0.5, 0.15):
                add("Bearings", f"BRG-P1-G{g+1}", "bearing seized / debris",
                    {"state": "restrained / seized" if s > 0.6 else "debris around"})
            if rng.random() < 0.5 * s:
                add("Bearings", f"BRG-P1-G{g+1}", "bearing shear deformation",
                    {"shear_mm": round(48 * (0.3 + 0.6 * s), 0), "elastomer_mm": 48})
    elif k == "bearing_degraded":
        for g in scn.girders:
            if rng.random() < _p(s, 0.3):
                st = "uneven bulging" if s < 0.45 else "surface cracks" if s < 0.75 else "splits / exposed plates"
                add("Bearings", f"BRG-{endn}-G{g+1}", "bearing bulging / splitting", {"state": st})
    elif k == "chloride_corrosion":
        if rng.random() < _p(s, 0.15):
            add("Expansion joints", f"EJ-{endn}/carriageway", "joint seal torn / leaking", {"length_m": round(1 + 8 * s, 1)})
        for g in range(C.N_GIRDERS):
            if rng.random() < _p(s, 0.35):
                add("PSC girders", f"G{g+1}/{zend}/web", "leakage / efflorescence",
                    {"area_m2": round(0.2 + s, 1), "state": "efflorescence" if s < 0.6 else "leakage with rust stain"})
            if s > 0.6 and rng.random() < _p(s, 0.75):
                add("PSC girders", f"G{g+1}/{zend}/bottom flange", "spalling",
                    {"area_m2": round(0.05 + 0.4 * (s - 0.6), 2), "depth_mm": 20, "steel exposed": "rebar" if s > 0.8 else "none"})
        if rng.random() < _p(s, 0.4):
            add("Pier P1" if scn.end == 1 else "Abutment A1", "pier cap top" if scn.end == 1 else "bearing shelf",
                "leakage / efflorescence", {"area_m2": 1.0, "state": "efflorescence"})

    # benign / unrelated findings that appear on any bridge
    if rng.random() < 0.35:
        add("Deck slab", f"B{rng.integers(1, 6)}/{rng.choice(I.ZONES)}/soffit", "crack",
            {"width_mm": round(float(rng.uniform(0.05, 0.25)), 2), "length_m": 1.0, "orientation": "map / random",
             "rust stain": False})
    if rng.random() < 0.3:
        add("Wearing coat", f"Lane{rng.integers(1, 4)}/{rng.choice(I.ZONES)}", "potholes / ravelling",
            {"area_m2": 0.5, "depth_mm": round(float(rng.uniform(10, 40)), 0)})
    if rng.random() < 0.25:
        add("Drainage", f"D{rng.integers(1, 5)}-{rng.choice(['L', 'R'])}", "drain blocked",
            {"state": str(rng.choice(["partly", "fully"]))})
    if rng.random() < 0.2:
        add("Expansion joints", f"EJ-{rng.choice(['A1', 'P1'])}/carriageway", "joint debris", {"extent": "partial"})
    if rng.random() < 0.15:
        add("PSC girders", f"G{rng.integers(1, 7)}/{rng.choice(I.ZONES)}/web", "crack",
            {"width_mm": 0.03, "length_m": 0.3, "orientation": "map / random", "rust stain": False})
    return obs


def visual_summary(obs):
    worst = {}
    for o in obs:
        worst[o["component"]] = max(worst.get(o["component"], 1), o["cs"])
    used = {c: worst.get(c, 1) for c in I.components()}
    has = lambda pred: int(any(pred(o) for o in obs))
    return dict(
        vis_hi=I.health_index(used), vis_cs_girders=worst.get("PSC girders", 1), vis_cs_bearings=worst.get("Bearings", 1),
        vis_cs_joints=worst.get("Expansion joints", 1), vis_cs_deck=worst.get("Deck slab", 1),
        vis_n_obs=len(obs),
        vis_flexural_crack=has(lambda o: o["defect"] == "crack" and str(o["inputs"].get("orientation", "")).startswith("flexural")),
        vis_duct_crack=has(lambda o: str(o["inputs"].get("orientation", "")).startswith("longitudinal")),
        vis_leakage_girder_end=has(lambda o: o["defect"] == "leakage / efflorescence" and "End" in o["location"]),
        vis_bearing_defect=has(lambda o: o["component"] == "Bearings"),
        vis_surface_deterioration=has(lambda o: o["defect"] == "surface deterioration"),
        vis_spalling=has(lambda o: o["defect"] == "spalling"),
    )


# ====================================================================== monitoring residuals
GAUGE_GIRDERS = (0, 2, 5)              # VW gauges on G1, G3, G6 (mid-span, bottom flange)
LEVEL_GIRDERS = (0, 2, 3, 5)           # hydrostatic levelling on G1, G3, G4, G6


def monitoring_residuals(scn, env, bg_static, Pe, rng, ref_change):
    """Long-term residuals vs. the reference curves (measured - predicted).

    Prestress loss: unit grillage response x loss, amplified by creep of the changed stress state.
    Stiffness damage: change of the deflection under the composite permanent loads.
    Errors: a creep-model error common to all gauges of the bridge (fraction of the reference change since
    handing over), per-sensor noise and imperfect temperature compensation.
    """
    fg = C.GIRDER_CONCRETE["fck"]
    age_d = env["age_years"] * 365
    phi_rem = PS.creep_coefficient(fg, 244, age_d, C.T_FINAL)
    amp = 1 + 0.4 * phi_rem
    defl = np.zeros(C.N_GIRDERS)
    dsig = np.zeros(C.N_GIRDERS)
    soffit = np.zeros(C.N_GIRDERS)
    if scn.kind == "prestress_loss":
        for g in scn.girders:
            d1, s1, so1 = prestress_loss_unit(bg_static, g)
            f = scn.params["loss"] * Pe / 1000.0
            defl += f * d1 * amp
            dsig += f * s1
            soffit += f * so1
    E = C.GIRDER_CONCRETE["Ecm"]
    strain = -dsig / E * 1e6 * amp                     # microstrain, elongation of the gauge is negative
    sys_err = rng.normal(0, 0.08)
    eps_res = strain + sys_err * ref_change["strain_ue"] + rng.normal(0, 6, C.N_GIRDERS)
    cam_res = defl + sys_err * ref_change["defl_mm"] + rng.normal(0, 0.4, C.N_GIRDERS) + rng.normal(0, 0.3)
    return eps_res, cam_res, soffit


# ====================================================================== one campaign
class Generator:
    """Holds the healthy references; call campaign() for each scenario/environment."""

    def __init__(self):
        self.asb = AsBuilt()
        self.bg_h = T.build(T.State())
        self.md_h = T.modal(self.bg_h, 10)
        self.sensors = V.sensor_layout()
        Smat = V.sensor_rows(self.bg_h, self.sensors)
        self.refs = [Smat @ self.md_h.shapes[:, r] for r in range(3)]
        self.bands = [(2.6, 4.6), (2.6, 4.6), (6.0, 10.0)]
        self.bg_static_h = T.build(T.State(), dynamic=False)
        self.Pe = effective_prestress()
        self.sidl_F = self._sidl_vector(self.bg_static_h)
        self.lt_h = T.load_test(self.bg_static_h, T.LOAD_TESTS["LT-1 eccentric (lanes 1-2)"], T.LT_FRONT_AXLE_X)
        self.w_sidl_h = self._mid_defl(self.bg_static_h, self.sidl_F)
        import json
        from pathlib import Path
        lt = json.loads(Path("outputs/baseline.json").read_text())["long_term"]
        self._ref_age = np.array(lt["age_days"]) / 365
        self._ref_eps = np.array(lt["gauge_strain"]) * 1e6
        self._ref_dfl = np.array(lt["midspan_deflection_mm"])

    def reference_change(self, age_years):
        """Predicted change since the reference reading at 1 year (what the creep-model error scales with)."""
        e = np.interp(age_years, self._ref_age, self._ref_eps) - np.interp(1.0, self._ref_age, self._ref_eps)
        d = np.interp(age_years, self._ref_age, self._ref_dfl) - np.interp(1.0, self._ref_age, self._ref_dfl)
        return dict(strain_ue=abs(e) + 20.0, defl_mm=abs(d) + 0.5)

    @staticmethod
    def _sidl_vector(bg):
        F = np.zeros(bg.grid.ndof)
        for name, y0, y1 in C.strip_extents():
            yc = 0.5 * (y0 + y1)
            if name in C.SIDL_LINE:
                F += G.line_load_vector(bg, yc, C.SIDL_LINE[name])
            elif name == "footpath":
                F += G.area_load_vector(bg, y0, y1, C.FOOTPATH_FINISH)
            elif name == "carriageway":
                F += G.area_load_vector(bg, y0, y1, C.WEARING_COAT)
        return F

    @staticmethod
    def _mid_defl(bg, F):
        u = bg.grid.solve(F)
        i = len(bg.xs) // 2
        return np.array([-u[bg.w_dof(i, j)] * 1e3 for j in range(len(bg.ys))])

    def campaign(self, scn, env, seed, keep_record=False, record_s=300.0):
        rng = np.random.default_rng(seed)
        ref_change = self.reference_change(env["age_years"])
        # prestress loss: decompression check decides whether the girder cracks under traffic
        pl_crack = {}
        if scn.kind == "prestress_loss":
            for g in scn.girders:
                _, _, so = prestress_loss_unit(self.bg_static_h, g)
                dsig = -so[g] * scn.params["loss"] * self.Pe / 1000.0          # loss of soffit compression
                if dsig > CRACK_MARGIN:
                    ex = dsig - CRACK_MARGIN
                    pl_crack[g] = 1 - 0.5 * min(1.0, ex / 6.0)
                    scn.params["cracked"] = True
                    scn.params["crack_mm"] = round(0.05 + 0.04 * ex, 3)
        st = twin_state(scn, env["T_deck"], pl_crack)
        # vibration
        bg = T.build(st)
        md = T.modal(bg, 10)
        t, a, _ = V.simulate(bg, md, duration=record_s, rate_per_min=env["traffic_per_min"], seed=seed)
        ap = V.preprocess(a)
        f, Gm = V.csd_matrix(ap)
        fid = V.modal_filter_peaks(f, Gm, self.refs, self.bands)
        feats = V.features(a)
        # load test (static, same temperature)
        st_s = twin_state(scn, env["T_deck"], pl_crack)
        bs = T.build(st_s, dynamic=False)
        lt = T.load_test(bs, T.LOAD_TESTS["LT-1 eccentric (lanes 1-2)"], T.LT_FRONT_AXLE_X)
        truck = rng.normal(1, 0.015)                      # weighed trucks: +-1.5 %
        lt_defl = np.array(lt["deflection_mm"]) * truck + rng.normal(0, 0.05, 6)
        lt_strain = np.array(lt["soffit_strain_ue"]) * truck + rng.normal(0, 3, 6)
        # long-term monitoring
        eps_res, cam_res, _ = monitoring_residuals(scn, env, self.bg_static_h, self.Pe, rng, ref_change)
        if scn.kind in ("girder_cracking", "concrete_deterioration", "chloride_corrosion") or pl_crack:
            st_r = twin_state(scn, 30.0, pl_crack)                 # temperature-compensated: reference 30 degC
            cam_res = cam_res + (self._mid_defl(T.build(st_r, dynamic=False), self.sidl_F) - self.w_sidl_h)
        # bearing movement / temperature slope at the free end (P1) of G1 and G6
        bs_k = T.bearing_stiffness()["kh"]
        k_ax = C.GIRDER_CONCRETE["Ecm"] * 1e3 * S.props(S.composite_layers(0)).A / C.L
        slope = []
        for g in (0, 5):
            kh = st.bearing_kh.get((g, 1), bs_k)
            slope.append(C.ALPHA_T * C.L * 1e3 * k_ax / (k_ax + kh) * rng.normal(1, 0.02))
        # NDT and visual
        nd = ndt_readings(scn, env, self.asb, rng)
        obs = visual_observations(scn, env, rng)
        row = dict(kind=scn.kind, severity=scn.severity, girders=" ".join(str(g + 1) for g in scn.girders),
                   end=["A1", "P1"][scn.end], **{f"p_{k}": v for k, v in scn.params.items() if not isinstance(v, bool)},
                   pl_cracked=int(bool(pl_crack)), **env,
                   f1=fid[0], f2=fid[1], f3=fid[2], f1_true=md.freq[0], **feats,
                   **{f"lt_defl_G{j+1}": lt_defl[j] for j in range(6)},
                   **{f"lt_strain_G{j+1}": lt_strain[j] for j in range(6)},
                   **{f"vw_res_G{g+1}": eps_res[g] for g in GAUGE_GIRDERS},
                   **{f"camber_res_G{g+1}": cam_res[g] for g in LEVEL_GIRDERS},
                   brg_slope_G1=slope[0], brg_slope_G6=slope[1],
                   **ndt_summary(nd), **visual_summary(obs))
        rec = (t, a) if keep_record else None
        return row, nd, obs, rec
