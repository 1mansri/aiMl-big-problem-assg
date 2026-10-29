"""Step 3: healthy baseline of the as-built bridge.

Modal properties, load-test predictions, long-term reference curves, vibration identification
on simulated records, thermal sensitivities, monitoring scheme and NDT reference values.
Writes outputs/baseline.json, outputs/data/baseline_accel.csv.gz and figures.
"""
import gzip
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bridge import config as C
from bridge import grillage as G
from bridge import longterm, monitoring
from bridge import section as S
from bridge import twin as T
from bridge import vibration as V
from run_step2_design import XS, load_effects
from bridge.twin import LOAD_TESTS, load_test

OUT = Path("outputs")
FIG = OUT / "figures"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

def bands_from(freqs, n=4):
    f = list(freqs[:n])
    edges = [f[0] - 0.4] + [0.5 * (a + b) for a, b in zip(f[:-1], f[1:])] + [f[-1] + 0.4]
    return list(zip(edges[:-1], edges[1:]))


def main():
    step1 = json.loads((OUT / "step1_analysis.json").read_text())
    res = {}

    # ------------------------------------------------------------------ modal
    st = T.State()
    bg = T.build(st)
    md = T.modal(bg, 12)
    mass = T.mass_matrix(bg)
    m_per_m = mass[0::3].sum() / C.L
    EI = st.E_factor_dyn * C.GIRDER_CONCRETE["Ecm"] * 1e3 * sum(S.props(S.composite_layers(j)).I for j in range(6))
    f_beam = math.pi / (2 * C.L ** 2) * math.sqrt(EI / m_per_m)
    res["modal"] = dict(freq=md.freq.tolist(), labels=md.labels, mass_t_per_m=m_per_m, f1_beam_hand=f_beam,
                        E_dyn_factor=st.E_factor_dyn, bearing=T.bearing_stiffness(), damping=V.DAMPING)
    print(f"As-built twin (E_dyn = {st.E_factor_dyn:.3f} Ecm, elastomeric bearings, 30 degC), mass {m_per_m:.2f} t/m")
    for k, (f, l) in enumerate(zip(md.freq, md.labels)):
        print(f"  mode {k+1:2d}: {f:6.2f} Hz  {l}")
    print(f"  hand check, simply supported beam f1 = {f_beam:.2f} Hz")

    sens = {}
    for name, s2, dyn in (("design model (static Ecm)", T.State(), False),
                          ("rigid-ish bearings (kv x 100)", T.State(bearing_kv={(j, e): 100.0 for j in range(6) for e in (0, 1)}), True),
                          ("barriers fully composite", T.State(barrier_composite=1.0), True),
                          ("deck 10 degC", T.State(temperature=10.0), True),
                          ("deck 45 degC", T.State(temperature=45.0), True),
                          ("all bearings seized (kh = 2e5 kN/m)", T.State(bearing_kh={(j, e): 2e5 for j in range(6) for e in (0, 1)}), True)):
        m2 = T.modal(T.build(s2, dynamic=dyn), 4)
        sens[name] = m2.freq.tolist()
        print(f"  {name:38s} f1..f4 = {np.round(m2.freq, 3)}")
    res["modal_sensitivity"] = sens
    df_dT = (sens["deck 45 degC"][0] - sens["deck 10 degC"][0]) / 35 / md.freq[0] * 100
    res["df1_dT_pct_per_degC"] = df_dT

    # ------------------------------------------------------------------ vibration identification
    bands = bands_from(md.freq, 4)
    Smat = V.sensor_rows(bg, V.sensor_layout())
    ident = []
    for seed in (1, 2, 3):
        t, a, sensors = V.simulate(bg, md, duration=600, seed=seed)
        ap = V.preprocess(a)
        f, P = V.welch_psd(ap)
        fq, s, U = V.fdd(ap)
        fm = V.fdd_modes(fq, s, U, bands)
        rec = dict(seed=seed, peak_pick=V.peak_pick(f, P, bands), fdd=[x[0] for x in fm],
                   mac=[T.mac(u, Smat @ md.shapes[:, r]) for r, (_, u) in enumerate(fm)], **V.features(a))
        ident.append(rec)
        if seed == 1:
            keep = (t, a, sensors, f, P, fq, s)
    res["identification"] = ident
    print("\nIdentification from simulated 10-min records (8 accelerometers, 100 Hz)")
    for r in range(4):
        pk = [d["peak_pick"][r] for d in ident]
        mc = [d["mac"][r] for d in ident]
        print(f"  mode {r+1} ({md.labels[r]:12s}) FE {md.freq[r]:.3f} Hz | identified {np.mean(pk):.3f} +- {np.std(pk):.3f} Hz,"
              f" MAC {np.mean(mc):.2f}")
    print(f"  a_rms {np.mean([d['a_rms'] for d in ident])*1e3/9.81:.1f} mg, a_max {np.mean([d['a_max'] for d in ident])*1e3/9.81:.1f} mg")
    t, a, sensors, f, P, fq, s = keep
    with gzip.open(DATA / "baseline_accel.csv.gz", "wt") as fh:
        fh.write("t," + ",".join(x.name for x in sensors) + "\n")
        for k in range(len(t)):
            fh.write(f"{t[k]:.2f}," + ",".join(f"{v:.6f}" for v in a[:, k]) + "\n")

    # ------------------------------------------------------------------ load test
    bs = T.build(T.State(), dynamic=False)
    lt = {name: load_test(bs, ys) for name, ys in LOAD_TESTS.items()}
    res["load_test"] = lt
    print("\nLoad-test predictions (2 trucks of 400 kN, static Ecm)")
    for name, r in lt.items():
        print(f"  {name}: front axle x = {r['front_axle_x']:.2f} m")
        print("    deflection (mm):  " + "  ".join(f"G{j+1} {d:5.2f}" for j, d in enumerate(r["deflection_mm"])))
        print("    soffit strain (ue): " + "  ".join(f"G{j+1} {e:5.1f}" for j, e in enumerate(r["soffit_strain_ue"])))

    # ------------------------------------------------------------------ long-term reference curves
    eff = load_effects(step1, 2)
    times = np.unique(np.concatenate([np.geomspace(C.T_STAGE1, C.T_FINAL, 120), [365 * y for y in (1, 5, 10, 30, 100)]]))
    eps, dfl = longterm.history(eff, XS, times, j=2)
    res["long_term"] = dict(age_days=times.tolist(), gauge_strain=eps.tolist(), midspan_deflection_mm=dfl.tolist(),
                            gauge_y=longterm.GAUGE_Y, girder=3)
    print("\nLong-term reference (G3 mid-span): VW gauge strain (shortening +) and deflection (+ down)")
    for yrs in (1, 5, 10, 30, 100):
        k = int(np.argmin(abs(times - 365 * yrs)))
        print(f"  {yrs:3d} y: strain {eps[k]*1e6:6.0f} ue, deflection {dfl[k]:6.1f} mm")

    # ------------------------------------------------------------------ thermal sensitivities
    res["thermal"] = dict(df1_dT_pct=df_dT, bearing_movement_mm_per_degC=C.ALPHA_T * C.L * 1e3,
                          note="free-end movement alpha*L; fixed end at the other abutment")
    print(f"\nThermal: df1/dT = {df_dT:+.3f} %/degC; free bearing movement {C.ALPHA_T*C.L*1e3:.3f} mm/degC")

    res["monitoring"] = monitoring.monitoring_scheme()
    res["ndt_baseline"] = monitoring.NDT_BASELINE
    print(f"Monitoring scheme: {len(res['monitoring'])} sensors/items; NDT reference values for {len(monitoring.NDT_BASELINE)} methods")
    (OUT / "baseline.json").write_text(json.dumps(res, indent=1, default=float))
    plots(bg, md, keep, res)


def plots(bg, md, keep, res):
    t, a, sensors, f, P, fq, s = keep
    xs = bg.xs
    fig, axs = plt.subplots(2, 3, figsize=(13, 6))
    for r, ax in enumerate(axs.ravel()):
        ws = md.w_shapes[:, :, r]
        ws = ws / abs(ws).max()
        for j in range(ws.shape[1]):
            ax.plot(xs, ws[:, j], label=f"G{j+1}")
        ax.set_title(f"mode {r+1}: {md.freq[r]:.2f} Hz ({md.labels[r]})", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Vertical mode shapes along each girder line (healthy twin)")
    fig.tight_layout()
    fig.savefig(FIG / "step3_mode_shapes.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(11, 9))
    axs[0].plot(t, a[1] / 9.81e-3, lw=0.4)
    axs[0].set(xlabel="t (s)", ylabel="a (mg)", title=f"{sensors[1].name}: simulated 10-min record (traffic + ambient)")
    for k in range(a.shape[0]):
        axs[1].semilogy(f, P[k], lw=0.6, label=sensors[k].name)
    for fr in md.freq[:6]:
        axs[1].axvline(fr, color="k", ls=":", lw=0.8)
    band = (f > 0.5) & (f < 25)
    axs[1].set(xlim=(0, 25), ylim=(P[:, band].max() * 1e-5, P[:, band].max() * 3),
               xlabel="f (Hz)", ylabel="PSD ((m/s2)2/Hz)", title="Welch PSD (dotted: FE frequencies)")
    axs[1].legend(fontsize=6, ncol=4)
    for k in range(3):
        axs[2].semilogy(fq, s[:, k], lw=0.8, label=f"SV{k+1}")
    for fr in md.freq[:6]:
        axs[2].axvline(fr, color="k", ls=":", lw=0.8)
    band = (fq > 2) & (fq < 16)
    axs[2].set(xlim=(2, 16), ylim=(s[band, 0].max() * 1e-5, s[band, 0].max() * 3),
               xlabel="f (Hz)", title="FDD singular values of the cross-spectral matrix")
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step3_vibration.png", dpi=150)
    plt.close(fig)

    lt = res["long_term"]
    yrs = np.array(lt["age_days"]) / 365
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].semilogx(yrs, np.array(lt["gauge_strain"]) * 1e6)
    axs[0].set(xlabel="girder age (years)", ylabel="strain (microstrain, shortening +)",
               title="VW gauge, G3 mid-span bottom flange")
    axs[1].semilogx(yrs, lt["midspan_deflection_mm"])
    axs[1].set(xlabel="girder age (years)", ylabel="deflection (mm, + down)", title="G3 mid-span camber / deflection")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
        for tt, lab in ((C.T_DECK_CAST, "deck"), (C.T_STAGE2, "stage 2"), (C.T_SIDL, "SIDL")):
            ax.axvline(tt / 365, color="grey", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "step3_long_term.png", dpi=150)
    plt.close(fig)

    # sensor layout (plan)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.add_patch(plt.Rectangle((0, 0), C.L, C.B_DECK, fill=False))
    for j, y in enumerate(C.GIRDER_Y):
        ax.plot([0, C.L], [y, y], color="#999", lw=1)
        ax.text(-1.2, y, f"G{j+1}", va="center", fontsize=8)
    mk = {"A": ("^", "tab:red", "accelerometer"), "VW": ("s", "tab:blue", "VW strain gauge pair"),
          "HL": ("o", "tab:green", "hydrostatic levelling"), "BD": ("D", "tab:purple", "bearing displacement"),
          "T": ("*", "tab:orange", "temperature (through depth)")}
    seen = set()
    for r in res["monitoring"]:
        key = r["id"].split("-")[0]
        if key not in mk or r["x"] is None or not isinstance(r["girder"], int):
            continue
        m, c, lab = mk[key]
        dy = {"A": 0.35, "VW": -0.35, "HL": 0.0, "BD": 0.0, "T": 0.6}[key]
        ax.plot(r["x"], C.GIRDER_Y[r["girder"] - 1] + dy, m, color=c, ms=8, label=None if key in seen else lab)
        seen.add(key)
    ax.set(xlim=(-2, C.L + 1), ylim=(-0.5, C.B_DECK + 0.5), xlabel="x (m)", ylabel="y (m)",
           title="Monitoring layout (plan); load-test gauges at L/4, L/2, 3L/4 of every girder not shown")
    ax.legend(loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.15))
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "step3_sensor_layout.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
