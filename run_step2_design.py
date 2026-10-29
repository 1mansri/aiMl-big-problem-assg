"""Step 2: cable layout, prestress losses, stage stresses, SLS and ULS checks.

Reads outputs/step1_analysis.json; writes outputs/step2_design.json and figures.
"""
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bridge import config as C
from bridge import design as D
from bridge import loads
from bridge import prestress as PS
from bridge import section as S

OUT = Path("outputs")
FIG = OUT / "figures"
NG_UNIQUE = (C.N_GIRDERS + 1) // 2
XS = np.linspace(0, C.L, 45)
R_SUP, R_INF = 1.1, 0.9          # characteristic prestress, IRC:112 cl. 7.9.5 (post-tensioned)
PSI = {"rare": (1.0, 0.6), "frequent": (0.75, 0.5), "quasi_permanent": (0.0, 0.5)}   # (live, temperature)


def load_effects(step1, j):
    """Characteristic moments at XS and shears at element mid-points for girder j."""
    j = str(j)
    dl = step1["stage1_dead_load"][j]
    cp = step1["composite_loads"][j]
    ll = step1["live_load"][j]
    full = lambda v: np.concatenate([[0.0], v, [0.0]])
    Mll = np.concatenate([[0.0], ll["M"], ll["M"][-2::-1], [0.0]])
    xe = 0.5 * (XS[:-1] + XS[1:])
    half = len(ll["V"])
    return dict(
        M_sw=np.array(dl["M_girder_only"]), M_dl1=np.array(dl["M"]),
        M_sidl=full(cp["sidl_M"]), M_wc=full(cp["wearing_coat_M"]), M_fp=full(cp["footpath_M"]), M_ll=Mll,
        xe=xe[:half], V_dl1=np.interp(xe[:half], XS, dl["V"]), V_sidl=np.array(cp["sidl_V"][:half]),
        V_wc=np.array(cp["wearing_coat_V"][:half]), V_fp=np.array(cp["footpath_V"][:half]),
        V_ll=np.array(ll["V"]),
    )


# ---------------------------------------------------------------------- cable layout checks
def width_at(y, x):
    z = 0.0
    for b1, b2, h in S.girder_layers(loads.girder_web(x)):
        if y <= z + h:
            return b1 + (b2 - b1) * (y - z) / h
        z += h
    return S.girder_layers()[-1][1]


def layout_checks():
    cs = PS.cables()
    xg = np.linspace(-C.END_OVERHANG + 0.3, C.L / 2, 600)       # from just inside the anchor
    min_cc, worst_pair = 9.0, None
    for a, b in combinations(cs, 2):
        d = np.hypot(a.y(xg) - b.y(xg), a.z(xg) - b.z(xg))
        k = int(d.argmin())
        if d[k] < min_cc:
            min_cc, worst_pair = d[k], (a.name, b.name, float(xg[k]))
    min_cover = 9.0
    where = None
    for c in cs:
        for x in xg:
            y, z = float(c.y(x)), float(c.z(x))
            side = min(width_at(y - C.DUCT_OD / 2, x), width_at(y + C.DUCT_OD / 2, x)) / 2 - abs(z) - C.DUCT_OD / 2
            cov = min(side, y - C.DUCT_OD / 2)
            if cov < min_cover:
                min_cover, where = cov, (c.name, float(x))
    radii = {c.name: PS.min_radius(c) for c in cs}
    ends = sorted(c.y_anchor for c in cs)
    return dict(min_centre_distance=min_cc, required_centre_distance=2 * C.DUCT_OD, worst_pair=worst_pair,
                min_cover=min_cover, required_cover=C.DUCT_COVER, cover_at=where,
                min_radius_vertical_horizontal=radii,
                anchor_angles_deg={c.name: math.degrees(PS.anchor_angle(c)) for c in cs},
                anchor_spacing=float(np.diff(ends).min()), anchor_top_edge=C.H_GIRDER - ends[-1],
                anchor_bottom_edge=ends[0])


# ---------------------------------------------------------------------- stresses
def stage_stresses(fh, eff, j, k):
    """Stress components at station k: dict stage -> fibre -> (prestress part, load part)."""
    x = XS[k]
    m = S.modular_ratio()
    holes_all = [(c.y(x), math.pi * C.DUCT_OD ** 2 / 4) for c in PS.cables()]
    holes_2 = [(c.y(x), math.pi * C.DUCT_OD ** 2 / 4) for c in PS.stage_cables(2)]
    pn_all = D.precast(x, holes_all)
    pn_2 = D.precast(x, holes_2)
    pp = D.precast(x)
    pc = D.composite(x, j)
    y1, y2 = fh.y(1)[k], fh.y(2)[k]
    F = lambda st, t: fh.force(st, t)[k]
    out = {}
    out["transfer"] = {f: (D.prestress_stress(pn_all, F(1, C.T_STAGE1), y1, f), D.stress(pn_all, 0, eff["M_sw"][k], f))
                       for f in D.FIBRES_PRECAST}
    out["deck_cast"] = {f: (D.prestress_stress(pn_2, F(1, C.T_DECK_CAST), y1, f), D.stress(pn_2, 0, eff["M_dl1"][k], f))
                        for f in D.FIBRES_PRECAST}

    def comp_state(t):
        res = {}
        for f in D.FIBRES_COMPOSITE:
            sp = 0.0
            if f != "deck_top":
                sp += D.prestress_stress(pp, F(1, C.T_DECK_CAST), y1, f)
            sp += D.prestress_stress(pc, F(1, t) - F(1, C.T_DECK_CAST), y1, f, m)
            sp += D.prestress_stress(pc, F(2, t), y2, f, m)
            sl = D.stress(pp, 0, eff["M_dl1"][k], f) if f != "deck_top" else 0.0
            res[f] = (sp, sl)
        return res
    out["stage2"] = comp_state(C.T_STAGE2)
    out["open"] = comp_state(C.T_SIDL)
    out["final"] = comp_state(C.T_FINAL)
    comp_loads = {
        "perm": {f: D.stress(pc, 0, eff["M_sidl"][k] + 1.2 * eff["M_wc"][k], f, m) for f in D.FIBRES_COMPOSITE},
        "live": {f: D.stress(pc, 0, eff["M_ll"][k] + eff["M_fp"][k], f, m) for f in D.FIBRES_COMPOSITE},
    }
    return out, comp_loads


def run_checks(fh, eff, j, temp, dshr):
    fg, fd = C.GIRDER_CONCRETE["fck"], C.DECK_CONCRETE["fck"]
    rows = []

    def add(check, stage, fibre, k, val, limit, kind):
        ok = val <= limit + 1e-9 if kind == "max" else val >= limit - 1e-9
        rows.append(dict(girder=j + 1, check=check, stage=stage, fibre=fibre, x=float(XS[k]),
                         value=float(val), limit=float(limit), ok=bool(ok)))

    for k in range(len(XS) // 2 + 1):
        st, cl = stage_stresses(fh, eff, j, k)
        for stage, t in (("transfer", C.T_STAGE1), ("deck_cast", C.T_DECK_CAST), ("stage2", C.T_STAGE2)):
            fct, fckt = PS.fctm_t(fg, t), PS.fck_t(fg, t)
            for f, (sp, sl) in st[stage].items():
                if f == "deck_top":
                    continue
                vmax = max(r * sp for r in (R_SUP, R_INF)) + sl
                vmin = min(r * sp for r in (R_SUP, R_INF)) + sl
                add("compression <= 0.48 fck(t)", stage, f, k, vmax, 0.48 * fckt, "max")
                add("tension >= -fctm(t)", stage, f, k, vmin, -fct, "min")
        for stage in ("open", "final"):
            for combo, (psi_l, psi_t) in PSI.items():
                for f in D.FIBRES_COMPOSITE:
                    sp, sl = st[stage][f]
                    base = sl + cl["perm"][f] + psi_l * cl["live"][f]
                    if stage == "final":
                        base += dshr[k][f]
                    tmax = psi_t * max(temp[k]["heating"][f], temp[k]["cooling"][f])
                    tmin = psi_t * min(temp[k]["heating"][f], temp[k]["cooling"][f])
                    vmax = max(r * sp for r in (R_SUP, R_INF)) + base + max(tmax, 0) * 1 + min(tmax, 0) * 0
                    vmin = min(r * sp for r in (R_SUP, R_INF)) + base + min(tmin, 0)
                    fck = fd if f == "deck_top" else fg
                    fctm = C.DECK_CONCRETE["fctm"] if f == "deck_top" else C.GIRDER_CONCRETE["fctm"]
                    if combo == "rare":
                        add("compression <= 0.48 fck", f"{stage}/rare", f, k, vmax, 0.48 * fck, "max")
                        if f == "girder_bottom":
                            add("[legacy] no tension under rare", f"{stage}/rare", f, k, vmin, 0.0, "min")
                    elif combo == "frequent":
                        add("uncracked: tension >= -fctm", f"{stage}/frequent", f, k, vmin, -fctm, "min")
                    else:
                        add("compression <= 0.36 fck", f"{stage}/QP", f, k, vmax, 0.36 * fck, "max")
                        if f == "girder_bottom":
                            add("decompression at soffit", f"{stage}/QP", f, k, vmin, 0.0, "min")
    return rows


def deflections(fh, eff, j):
    """Mid-span deflections (mm, downward +) by integrating curvature x unit-load moment."""
    fg = C.GIRDER_CONCRETE["fck"]
    xs = XS
    mbar = np.where(xs <= C.L / 2, xs / 2, (C.L - xs) / 2)
    pp = [D.precast(x) for x in xs]
    pc = [D.composite(x, j) for x in xs]
    y1, y2 = fh.y(1), fh.y(2)

    def defl(M, props, E):
        kappa = M / (E * 1e3 * np.array([p.I for p in props]))
        return np.trapezoid(kappa * mbar, xs) * 1e3

    E = lambda t: PS.ecm_t(fg, t)
    phi = lambda t0: PS.creep_coefficient(fg, D.girder_h0_mm(), t0, C.T_FINAL)
    yb_p = np.array([p.yb for p in pp])
    yb_c = np.array([p.yb for p in pc])
    P1_0, P1_d, P1_f = fh.force(1, C.T_STAGE1), fh.force(1, C.T_DECK_CAST), fh.force(1, C.T_FINAL)
    P2_0, P2_f = fh.force(2, C.T_STAGE2), fh.force(2, C.T_FINAL)
    d_sw = defl(eff["M_sw"], pp, E(C.T_STAGE1))
    d_p1 = defl(-P1_0 * (yb_p - y1), pp, E(C.T_STAGE1))
    d_p1_mean = defl(-0.5 * (P1_0 + P1_f) * (yb_p - y1), pp, E(C.T_STAGE1))
    d_slab = defl(eff["M_dl1"] - eff["M_sw"], pp, E(C.T_DECK_CAST))
    d_p2 = defl(-P2_0 * (yb_c - y2), pc, E(C.T_STAGE2))
    d_p2_mean = defl(-0.5 * (P2_0 + P2_f) * (yb_c - y2), pc, E(C.T_STAGE2))
    d_sidl = defl(eff["M_sidl"] + eff["M_wc"], pc, E(C.T_SIDL))
    d_ll = defl(eff["M_ll"] + eff["M_fp"], pc, C.GIRDER_CONCRETE["Ecm"])
    transfer = d_sw + d_p1
    open_ = transfer + d_slab + d_p2 + d_sidl
    final = (d_sw * (1 + phi(C.T_STAGE1)) + d_p1 * (P1_f / P1_0).mean() + d_p1_mean * phi(C.T_STAGE1)
             + d_slab * (1 + phi(C.T_DECK_CAST)) + d_p2 * (P2_f / P2_0).mean() + d_p2_mean * phi(C.T_STAGE2)
             + d_sidl * (1 + phi(C.T_SIDL)))
    return dict(at_transfer=transfer, at_opening=open_, long_term=final, live_load_beam_line=d_ll,
                limit_live=C.L / 800 * 1e3, parts=dict(sw=d_sw, p1=d_p1, slab=d_slab, p2=d_p2, sidl=d_sidl))


def main():
    step1 = json.loads((OUT / "step1_analysis.json").read_text())
    res = {}

    lay = layout_checks()
    res["layout"] = lay
    cs = PS.cables()
    print("Cable layout: " + ", ".join(f"{c.name} {c.n}T15 (stage {c.stage})" for c in cs))
    print(f"  min duct c/c {lay['min_centre_distance']*1e3:.0f} mm (need >= {lay['required_centre_distance']*1e3:.0f}), "
          f"pair {lay['worst_pair'][0]}-{lay['worst_pair'][1]} at x={lay['worst_pair'][2]:.2f} m")
    print(f"  min cover to duct {lay['min_cover']*1e3:.0f} mm (need >= {C.DUCT_COVER*1e3:.0f}) at {lay['cover_at']}")
    for n, (rv, rh) in lay["min_radius_vertical_horizontal"].items():
        print(f"  {n}: R_vert={rv:.1f} m" + (f", R_plan={rh:.1f} m" if rh else "") +
              f", anchor angle {lay['anchor_angles_deg'][n]:.1f} deg")
    print(f"  anchorages at the end face: min spacing {lay['anchor_spacing']:.2f} m, "
          f"top edge {lay['anchor_top_edge']:.2f} m, bottom edge {lay['anchor_bottom_edge']:.2f} m")

    all_checks, summary = [], {}
    for j in range(NG_UNIQUE):
        eff = load_effects(step1, j)
        fh = D.force_history(XS, eff["M_dl1"], eff["M_sidl"] + eff["M_wc"], j)
        temp = [D.temperature_stresses(j, x) for x in XS]
        dshr = [D.differential_shrinkage(j, x)[0] for x in XS]
        all_checks += run_checks(fh, eff, j, temp, dshr)
        k = len(XS) // 2
        sj = C.JACK_RATIO * C.STRAND_FPK
        loss = {
            "jack": sj,
            "after friction+slip (stage 1 mean)": fh.sigma0(1)[k],
            "after friction+slip (stage 2)": fh.sigma0(2)[k],
            "elastic shortening, stage 1 sequential": fh.es1[k],
            "elastic shortening of stage 1 by stage 2": fh.es2[k],
            "long-term stage 1 (creep+shrinkage+relaxation)": fh.lt[1][k],
            "long-term stage 2": fh.lt[2][k],
            "relaxation part stage 1/2": [fh.relax[1][k], fh.relax[2][k]],
            "final stage 1": fh.sigma(1, C.T_FINAL)[k],
            "final stage 2": fh.sigma(2, C.T_FINAL)[k],
        }
        P0 = fh.force(1, C.T_STAGE1)[k] + fh.area(2) * fh.sigma0(2)[k] * 1e3
        Pe = fh.force(1, C.T_FINAL)[k] + fh.force(2, C.T_FINAL)[k]
        Pj = (fh.area(1) + fh.area(2)) * sj * 1e3
        loss["P_jack_kN"], loss["P0_kN"], loss["Pe_kN"] = Pj, P0, Pe
        loss["total_loss_pct"] = 100 * (1 - Pe / Pj)
        loss["phi"] = fh.phi
        loss["eps_cs"] = fh.eps_cs
        _, ds_info = D.differential_shrinkage(j, C.L / 2)

        # ULS flexure along the span
        uls = []
        for kk in range(1, len(XS) // 2 + 1):
            x = XS[kk]
            spe = {}
            for c in PS.cables():
                s0 = fh.sigma_slip[c.name][kk]
                spe[c.name] = (s0 - fh.es1[kk] - fh.es2[kk] - fh.lt[1][kk]) if c.stage == 1 else (s0 - fh.lt[2][kk])
            MRd, xu, info = D.uls_moment_capacity(j, x, spe)
            MEd = 1.35 * (eff["M_dl1"][kk] + eff["M_sidl"][kk]) + 1.75 * eff["M_wc"][kk] + 1.5 * (eff["M_fp"][kk] + eff["M_ll"][kk])
            uls.append(dict(x=float(x), M_Ed=float(MEd), M_Rd=float(MRd), ratio=float(MEd / MRd), xu=float(xu),
                            tendons_yielded=info["yielded"]))
        # shear
        shear = []
        for kk, x in enumerate(eff["xe"]):
            VEd = 1.35 * (eff["V_dl1"][kk] + eff["V_sidl"][kk]) + 1.75 * eff["V_wc"][kk] + 1.5 * (eff["V_fp"][kk] + eff["V_ll"][kk])
            Pm = float(np.interp(x, XS, fh.force(1, C.T_FINAL) + fh.force(2, C.T_FINAL)))
            cs = PS.cables()
            A = sum(c.area for c in cs)
            slope = sum(c.area * (c.y(x + 0.01) - c.y(x - 0.01)) / 0.02 for c in cs) / A
            r = D.shear_check(float(x), VEd, Pm, slope, j)
            r["interface"] = D.interface_shear(VEd, float(x), j)
            shear.append(r)
        dfl = deflections(fh, eff, j)
        summary[j] = dict(losses=loss, diff_shrinkage=ds_info, uls=uls, shear=shear, deflection=dfl,
                          temperature_mid={k2: v for k2, v in temp[len(XS) // 2].items()},
                          diff_shrinkage_mid=dshr[len(XS) // 2])
        if j == 0:
            res["force_profile"] = dict(x=XS.tolist(),
                                        P_transfer=fh.force(1, C.T_STAGE1).tolist(),
                                        P_stage2=(fh.force(1, C.T_STAGE2) + fh.force(2, C.T_STAGE2)).tolist(),
                                        P_final=(fh.force(1, C.T_FINAL) + fh.force(2, C.T_FINAL)).tolist(),
                                        y1=fh.y(1).tolist(), y2=fh.y(2).tolist())

    res["girders"] = summary
    res["checks"] = all_checks

    # ------------------------------------------------------------------ print
    L0 = summary[0]["losses"]
    print("\nPrestress at mid-span (MPa unless noted), girder G1")
    for kk, v in L0.items():
        if isinstance(v, (float, np.floating)):
            print(f"  {kk:48s} {v:9.1f}")
    print(f"  creep coefficients phi(inf,t0): stage 1 {L0['phi'][1]:.2f}, stage 2 {L0['phi'][2]:.2f};  "
          f"shrinkage after stressing: {L0['eps_cs'][1]*1e6:.0f} / {L0['eps_cs'][2]*1e6:.0f} microstrain")
    ds = summary[0]["diff_shrinkage"]
    print(f"  differential shrinkage: deck {ds['eps_deck']*1e6:.0f}, girder after deck {ds['eps_girder_after_deck']*1e6:.0f}, "
          f"difference {ds['diff']*1e6:.0f} microstrain, creep factor {ds['factor']:.2f}")
    print("  mid-span secondary stresses (MPa): temperature", {k: {f: round(v, 2) for f, v in d.items()} for k, d in summary[0]['temperature_mid'].items()})
    print("                                    diff. shrinkage", {f: round(v, 2) for f, v in summary[0]['diff_shrinkage_mid'].items()})

    print("\nStress checks (worst over stations, girders G1-G3; characteristic prestress 0.9/1.1 P_m)")
    groups = {}
    for r in all_checks:
        key = (r["check"], r["stage"], r["fibre"])
        margin = (r["limit"] - r["value"]) if r["value"] <= r["limit"] + 1e9 and "compression" in r["check"] else (r["value"] - r["limit"])
        if key not in groups or margin < groups[key][0]:
            groups[key] = (margin, r)
    for key, (margin, r) in groups.items():
        flag = "OK " if r["ok"] else "NOT OK"
        print(f"  {flag:6s} {r['stage']:16s} {r['fibre']:13s} {r['check']:32s} {r['value']:7.2f} vs {r['limit']:6.2f} "
              f"(G{r['girder']}, x={r['x']:.2f})")

    print("\nULS flexure (worst M_Ed/M_Rd along the span)")
    for j, sj in summary.items():
        w = max(sj["uls"], key=lambda r: r["ratio"])
        print(f"  G{j+1}: M_Ed={w['M_Ed']:.0f}  M_Rd={w['M_Rd']:.0f} kNm  ratio {w['ratio']:.2f} at x={w['x']:.2f} m, "
              f"NA depth {w['xu']*1e3:.0f} mm, all tendons yielded: {w['tendons_yielded']}")
    print("\nShear (ULS, per girder: worst sections)")
    for j, sj in summary.items():
        for r in sj["shear"][::4]:
            print(f"  G{j+1} x={r['x']:5.2f} bw={r['bw']:.2f} V_Ed={r['V_Ed']:5.0f} V_p={r['V_p']:4.0f} "
                  f"V_Rd,c={r['V_Rd_c']:5.0f} V_Rd,max={r['V_Rd_max']:5.0f} links {'needed' if r['links_needed'] else 'minimum'}: "
                  f"Asw/s={r['asw_design_mm2_per_m']:4.0f} mm2/m | interface v={r['interface']['v_Edi']:.2f} MPa, "
                  f"As={r['interface']['As_req_mm2_per_m']:.0f} mm2/m")
    print("\nMid-span deflection (mm, + down)")
    for j, sj in summary.items():
        d = sj["deflection"]
        print(f"  G{j+1}: transfer {d['at_transfer']:.1f}, opening {d['at_opening']:.1f}, long-term {d['long_term']:.1f}, "
              f"live (beam line) {d['live_load_beam_line']:.1f} (limit {d['limit_live']:.1f})")

    (OUT / "step2_design.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    plots(res)
    n_bad = sum(not r["ok"] for r in all_checks if not r["check"].startswith("[legacy]"))
    print(f"\nFailed code checks: {n_bad}")


def plots(res):
    cs = PS.cables()
    xg = np.linspace(-C.END_OVERHANG, C.L + C.END_OVERHANG, 800)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for c in cs:
        axs[0].plot(xg, c.y(xg), label=f"{c.name} ({c.n}T15, stage {c.stage})")
        axs[1].plot(xg, c.z(xg), label=c.name)
    axs[0].axhline(C.H_GIRDER, color="k", lw=0.8)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set(ylabel="height above soffit (m)", title="Cable profiles: elevation")
    axs[1].set(ylabel="lateral offset (m)", xlabel="x from left bearing (m)", title="Plan")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "step2_cable_profiles.png", dpi=150)
    plt.close(fig)

    # mid-span section with ducts
    fig, ax = plt.subplots(figsize=(4, 5))
    z = 0.0
    for b1, b2, h in S.girder_layers():
        ax.add_patch(plt.Polygon([(-b1 / 2, z), (b1 / 2, z), (b2 / 2, z + h), (-b2 / 2, z + h)], color="#9bb"))
        z += h
    bd = S.deck_width(1)
    ax.add_patch(plt.Rectangle((-bd / 2, C.H_GIRDER), bd, C.T_DECK, color="#bcd"))
    for c in cs:
        ax.add_patch(plt.Circle((float(c.z(C.L / 2)), float(c.y(C.L / 2))), C.DUCT_OD / 2, color="k" if c.stage == 1 else "r"))
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.1, 2.1)
    ax.set_title("Mid-span: black stage 1, red stage 2")
    fig.tight_layout()
    fig.savefig(FIG / "step2_midspan_section.png", dpi=150)
    plt.close(fig)

    fp = res["force_profile"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(fp["x"], np.array(fp["P_transfer"]) / 1e3, label="stage 1 at transfer")
    ax.plot(fp["x"], np.array(fp["P_stage2"]) / 1e3, label="all cables after stage-2 stressing")
    ax.plot(fp["x"], np.array(fp["P_final"]) / 1e3, label="all cables, final (100 y)")
    ax.set(xlabel="x (m)", ylabel="prestress force (MN)", title="Prestress force along the girder")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "step2_prestress_force.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
