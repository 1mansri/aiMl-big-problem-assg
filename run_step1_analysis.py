"""Step 1: section properties, loads and grillage load effects per girder.

Writes outputs/step1_analysis.json and outputs/figures/step1_*.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bridge import config as C
from bridge import grillage as G
from bridge import loads, moving
from bridge import section as S

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NG_UNIQUE = (C.N_GIRDERS + 1) // 2          # symmetric deck: girders 1..3 cover all


def section_table():
    rows = {}
    for name, layers in (
        ("precast mid", S.girder_layers()),
        ("precast end block", S.girder_layers(C.END_BLOCK_WEB)),
        ("composite mid, interior", S.composite_layers(1)),
        ("composite mid, outer", S.composite_layers(0)),
    ):
        p = S.props(layers)
        rows[name] = dict(A=p.A, yb=p.yb, yt=p.yt, I=p.I, Zb=p.Zb, Zt=p.Zt)
    return rows


def dead_load_stage1(i_mid):
    out = {}
    for j in range(NG_UNIQUE):
        xs, M, V, R = loads.ss_beam(loads.stage1_line_load(j), loads.cross_girder_point_loads(j),
                                    xs=np.linspace(0, C.L, 45))
        _, Mg, Vg, _ = loads.ss_beam(loads.girder_self_weight, xs=np.linspace(0, C.L, 45))
        out[j] = dict(M=M.tolist(), V=V.tolist(), M_girder_only=Mg.tolist(), V_girder_only=Vg.tolist(),
                      reaction=R[0], M_mid=M[i_mid], V_sup=V[0])
    return out


def sidl_vectors(bg):
    """Nodal load vectors of the superimposed dead loads, grouped by load factor family."""
    ext = C.strip_extents()
    F_sidl = np.zeros(bg.grid.ndof)
    F_wc = np.zeros(bg.grid.ndof)
    F_fp = np.zeros(bg.grid.ndof)
    p_fp = loads.footpath_intensity()
    F_fp_one = np.zeros(bg.grid.ndof)
    for name, y0, y1 in ext:
        yc = 0.5 * (y0 + y1)
        if name in C.SIDL_LINE:
            F_sidl += G.line_load_vector(bg, yc, C.SIDL_LINE[name])
        elif name == "footpath":
            F_sidl += G.area_load_vector(bg, y0, y1, C.FOOTPATH_FINISH)
            Ff = G.area_load_vector(bg, y0, y1, p_fp)
            F_fp += Ff
            if y0 < C.B_DECK / 2:
                F_fp_one += Ff
        elif name == "carriageway":
            F_wc += G.area_load_vector(bg, y0, y1, C.WEARING_COAT)
    return F_sidl, F_wc, F_fp, F_fp_one, p_fp


def main():
    bg = G.build()
    nx = len(bg.xs) - 1
    i_mid = nx // 2
    res = {"config": {
        "L": C.L, "B_deck": C.B_DECK, "carriageway": [C.CW_LEFT, C.CW_RIGHT], "girder_y": C.GIRDER_Y,
        "H_girder": C.H_GIRDER, "t_deck": C.T_DECK, "girder_concrete": C.GIRDER_CONCRETE,
        "deck_concrete": C.DECK_CONCRETE, "modular_ratio": S.modular_ratio(),
    }}

    # ---------------------------------------------------------------- sections
    res["sections"] = section_table()
    res["J_precast"] = S.torsion_constant_precast()
    res["effective_flange"] = [dict(part=a, b_i=b, b_eff=c, fully_effective=d) for a, b, c, d in S.effective_flange_check()]
    print("Section properties (m, m^2, m^4, m^3)")
    for k, r in res["sections"].items():
        print(f"  {k:26s} A={r['A']:.4f} yb={r['yb']:.3f} I={r['I']:.4f} Zb={r['Zb']:.4f} Zt={r['Zt']:.4f}")
    print(f"  modular ratio Ed/Eg = {S.modular_ratio():.3f};  J precast = {res['J_precast']:.5f} m^4")
    for r in res["effective_flange"]:
        print(f"  b_eff {r['part']}: b_i={r['b_i']:.3f} -> {r['b_eff']:.3f} (fully effective: {r['fully_effective']})")

    # ---------------------------------------------------------------- dead load, stage 1
    dl = dead_load_stage1(i_mid)
    res["stage1_dead_load"] = dl
    print("\nStage 1 (precast girder alone): girder + wet slab + cross girders")
    for j, r in dl.items():
        print(f"  G{j+1}: M_mid={r['M_mid']:.0f} kNm  V_sup={r['V_sup']:.0f} kN  "
              f"(girder self-weight M_mid={r['M_girder_only'][i_mid]:.0f})")

    # ---------------------------------------------------------------- composite loads via grillage
    sM = G.girder_moment_sensitivities(bg)
    sV = G.girder_shear_sensitivities(bg)
    F_sidl, F_wc, F_fp, F_fp_one, p_fp = sidl_vectors(bg)
    res["footpath_intensity_kPa"] = p_fp
    comp = {}
    for j in range(NG_UNIQUE):
        r = {}
        for name, F in (("sidl", F_sidl), ("wearing_coat", F_wc)):
            r[name + "_M"] = [sM[(j, i)] @ F for i in range(1, nx)]
            r[name + "_V"] = [sV[(j, i)] @ F for i in range(nx)]
        r["footpath_M"] = [max(sM[(j, i)] @ F_fp, sM[(j, i)] @ F_fp_one) for i in range(1, nx)]
        r["footpath_V"] = [max(sV[(j, i)] @ F_fp, sV[(j, i)] @ F_fp_one) for i in range(nx)]
        comp[j] = r
    res["composite_loads"] = comp
    print(f"\nFootpath live load (IRC:6 cl. 206.1) = {p_fp:.2f} kPa")
    print("Composite stage (grillage): SIDL = barriers + railing + footpath finish; WC = wearing coat")
    for j, r in comp.items():
        print(f"  G{j+1}: M_mid SIDL={r['sidl_M'][i_mid-1]:.0f}  WC={r['wearing_coat_M'][i_mid-1]:.0f}  "
              f"footpath LL={r['footpath_M'][i_mid-1]:.0f} kNm")

    # ---------------------------------------------------------------- vehicle live load
    print("\nVehicle live load envelopes (impact included) ...")
    ll = {}
    for j in range(NG_UNIQUE):
        Mmax, Mcase, Vmax, Vmin, Vcase = [], [], [], [], []
        for i in range(1, i_mid + 1):
            p = moving.envelope(bg, sM[(j, i)])
            Mmax.append(p.value)
            Mcase.append(p.case)
        for i in range(i_mid):
            p = moving.envelope(bg, sV[(j, i)])
            q = moving.envelope(bg, sV[(j, i)], sign=-1.0)
            Vmax.append(p.value)
            Vmin.append(q.value)
            Vcase.append(p.case)
        pm = moving.envelope(bg, sM[(j, i_mid)])
        sd = G.girder_deflection_sensitivity(bg, i_mid, j)
        pd = moving.envelope(bg, sd)
        ll[j] = dict(M=Mmax, M_case=Mcase, V=Vmax, V_neg=Vmin, V_case=Vcase,
                     mid_placement=pm.__dict__, deflection_mid_mm=pd.value * 1e3, deflection_case=pd.case)
        print(f"  G{j+1}: M_mid={Mmax[-1]:.0f} kNm ({Mcase[-1]}); V near support={Vmax[0]:.0f} kN ({Vcase[0]}); "
              f"defl={pd.value*1e3:.1f} mm ({pd.case})")
        for name, yc, x0, d in pm.vehicles:
            print(f"        {name:12s} centre y={yc:.2f} m, lead axle x={x0:.2f} m, dir {'+' if d > 0 else '-'}")
    res["live_load"] = ll

    # ---------------------------------------------------------------- Courbon cross-check
    # Same placement as the governing outer-girder moment; Courbon share of the total static moment
    pm = ll[0]["mid_placement"]
    pl = moving.Placement(pm["case"], pm["value"], pm["vehicles"])
    F = moving.placement_load_vector(bg, pl)
    total = sum(sM[(j, i_mid)] @ F for j in range(C.N_GIRDERS))
    shares = [sM[(j, i_mid)] @ F / total for j in range(C.N_GIRDERS)]
    # resultant eccentricity of the placed wheel loads
    w = -F[0::3]
    ybar = np.sum(w * np.repeat(bg.ys[None, :], len(bg.xs), 0).ravel()) / w.sum()
    yc = np.array(bg.ys) - C.B_DECK / 2
    e = ybar - C.B_DECK / 2
    courbon = [(1 + C.N_GIRDERS * e * yi / np.sum(yc ** 2)) / C.N_GIRDERS for yi in yc]
    res["courbon_check"] = dict(case=pm["case"], eccentricity=e, grillage_share=shares, courbon_share=courbon)
    print(f"\nCourbon check ({pm['case']}, resultant eccentricity {e:+.2f} m)")
    print("  girder   grillage   Courbon")
    for j in range(C.N_GIRDERS):
        print(f"  G{j+1}      {shares[j]:.3f}      {courbon[j]:.3f}")

    # ---------------------------------------------------------------- summary of characteristic effects
    summ = {}
    print("\nCharacteristic mid-span moments (kNm) and support shears (kN)")
    print("  girder  DL1   SIDL   WC   FP-LL  Veh-LL  | V: DL1  SIDL  WC  FP  Veh")
    for j in range(NG_UNIQUE):
        d = dict(M_DL1=dl[j]["M_mid"], M_SIDL=comp[j]["sidl_M"][i_mid - 1], M_WC=comp[j]["wearing_coat_M"][i_mid - 1],
                 M_FP=comp[j]["footpath_M"][i_mid - 1], M_LL=ll[j]["M"][-1],
                 V_DL1=dl[j]["V_sup"], V_SIDL=comp[j]["sidl_V"][0], V_WC=comp[j]["wearing_coat_V"][0],
                 V_FP=comp[j]["footpath_V"][0], V_LL=ll[j]["V"][0])
        d["M_service_rare"] = d["M_DL1"] + d["M_SIDL"] + 1.2 * d["M_WC"] + d["M_FP"] + d["M_LL"]
        d["M_ULS"] = 1.35 * (d["M_DL1"] + d["M_SIDL"]) + 1.75 * d["M_WC"] + 1.5 * (d["M_FP"] + d["M_LL"])
        summ[j] = d
        print(f"  G{j+1}   {d['M_DL1']:5.0f} {d['M_SIDL']:5.0f} {d['M_WC']:5.0f} {d['M_FP']:5.0f} {d['M_LL']:6.0f}"
              f"  | {d['V_DL1']:5.0f} {d['V_SIDL']:5.0f} {d['V_WC']:4.0f} {d['V_FP']:3.0f} {d['V_LL']:4.0f}"
              f"   rare M={d['M_service_rare']:.0f}, ULS M={d['M_ULS']:.0f}")
    gov = max(summ, key=lambda j: summ[j]["M_ULS"])
    print(f"  Governing girder for design: G{gov+1}")
    res["summary"] = summ
    res["governing_girder"] = gov

    (OUT / "step1_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    plots(bg, res, i_mid)


def plots(bg, res, i_mid):
    # cross-section
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for name, y0, y1 in C.strip_extents():
        col = {"carriageway": "#bbb", "footpath": "#e8d9a8", "crash_barrier": "#777", "railing": "#555"}[name]
        hgt = {"carriageway": 0.07, "footpath": 0.25, "crash_barrier": 0.9, "railing": 1.1}[name]
        ax.add_patch(plt.Rectangle((y0, C.H_COMPOSITE), y1 - y0, hgt, color=col))
    ax.add_patch(plt.Rectangle((0, C.H_GIRDER), C.B_DECK, C.T_DECK, color="#9ab"))
    for yg in C.GIRDER_Y:
        z = 0.0
        for b1, b2, h in S.girder_layers():
            ax.add_patch(plt.Polygon([(yg - b1 / 2, z), (yg + b1 / 2, z), (yg + b2 / 2, z + h), (yg - b2 / 2, z + h)],
                                     color="#6a8caf"))
            z += h
    ax.set_xlim(-0.3, C.B_DECK + 0.3)
    ax.set_ylim(-0.2, C.H_COMPOSITE + 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)")
    ax.set_title("33 m PSC I-girder bridge: cross-section (6 girders at 2.6 m, 1.80 m + 0.22 m deck)")
    fig.tight_layout()
    fig.savefig(FIG / "step1_cross_section.png", dpi=150)
    plt.close(fig)

    # envelopes
    xs = bg.xs
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for j in range(NG_UNIQUE):
        M = res["live_load"][j]["M"]
        Mfull = np.concatenate([[0], M, M[-2::-1], [0]])
        axs[0].plot(xs, Mfull, label=f"G{j+1}")
        V = res["live_load"][j]["V"]
        xe = 0.5 * (xs[:-1] + xs[1:])[: len(V)]
        axs[1].plot(xe, V, label=f"G{j+1}")
    axs[0].set(xlabel="x (m)", ylabel="M (kNm)", title="Vehicle LL moment envelope (incl. impact)")
    axs[1].set(xlabel="x (m)", ylabel="V (kN)", title="Vehicle LL shear envelope, left half")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "step1_ll_envelopes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
