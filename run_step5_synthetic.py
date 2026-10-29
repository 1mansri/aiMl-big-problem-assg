"""Step 5: synthetic inspection, NDT and monitoring data for healthy and deteriorated conditions.

Writes
  outputs/data/campaigns.csv           one row per campaign: environment, measurements, features, truth
  outputs/data/ndt_readings.csv.gz     81 test points per campaign
  outputs/data/visual_observations.csv inspector's observations per campaign
  outputs/data/accel/*.csv.gz          example 300 s acceleration records (healthy + each mechanism)
  outputs/condition_table.json         noise-free effect of each mechanism at three severities
  outputs/figures/step5_*.png
Run: python run_step5_synthetic.py [campaigns per class] [workers]
"""
import gzip
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bridge import config as C
from bridge import synthetic as SY
from bridge import twin as T
from bridge import vibration as V

OUT = Path("outputs")
DATA = OUT / "data"
FIG = OUT / "figures"
(DATA / "accel").mkdir(parents=True, exist_ok=True)

_GEN = None


def _init():
    global _GEN
    _GEN = SY.Generator()


def _work(job):
    cid, kind, seed = job
    rng = np.random.default_rng(seed)
    scn = SY.sample_scenario(kind, rng)
    env = SY.environment(rng)
    row, nd, obs, _ = _GEN.campaign(scn, env, seed=seed + 1)
    row = {"campaign": cid, **row}
    for r in nd:
        r["campaign"] = cid
    for o in obs:
        o["campaign"] = cid
        o["inputs"] = json.dumps(o["inputs"])
    return row, nd, obs


def condition_table(gen):
    """Noise-free effect of each mechanism (fixed scenario geometry, 30 degC, age 10 y)."""
    fixed = {
        "healthy": lambda s: SY.Scenario("healthy"),
        "prestress_loss": lambda s: SY.Scenario("prestress_loss", s, (0,), params=dict(loss=0.03 + 0.27 * s, cause="chloride")),
        "girder_cracking": lambda s: SY.Scenario("girder_cracking", s, (0,), params=dict(EI=1 - 0.45 * s, half_zone=6.0, crack_mm=0.05 + 0.5 * s)),
        "concrete_deterioration": lambda s: SY.Scenario("concrete_deterioration", s, (0, 1), params=dict(E=1 - 0.3 * s)),
        "bearing_seized": lambda s: SY.Scenario("bearing_seized", s, tuple(range(6)), end=1, params=dict(kh=10 ** (4.5 + 1.5 * s))),
        "bearing_degraded": lambda s: SY.Scenario("bearing_degraded", s, (0,), end=1, params=dict(kv=1 - 0.85 * s)),
        "chloride_corrosion": lambda s: SY.Scenario("chloride_corrosion", s, tuple(range(6)), end=0, params=dict(c=s, EI_end=1 - 0.2 * max(0, (s - 0.6) / 0.4))),
    }
    rows = []
    Smat = V.sensor_rows(gen.bg_h, gen.sensors)
    for kind, make in fixed.items():
        for s in ([0.0] if kind == "healthy" else [0.25, 0.5, 0.75]):
            scn = make(s)
            pl_crack = {}
            if kind == "prestress_loss":
                _, _, so = SY.prestress_loss_unit(gen.bg_static_h, 0)
                dsig = -so[0] * scn.params["loss"] * gen.Pe / 1000
                if dsig > SY.CRACK_MARGIN:
                    pl_crack[0] = 1 - 0.5 * min(1, (dsig - SY.CRACK_MARGIN) / 6)
            st = SY.twin_state(scn, 30.0, pl_crack)
            md = T.modal(T.build(st), 8)
            # assign modes to the baseline shapes by MAC
            fr = []
            for ref in gen.refs:
                macs = [T.mac(Smat @ md.shapes[:, r], ref) for r in range(8)]
                fr.append(md.freq[int(np.argmax(macs))])
            lt = T.load_test(T.build(st, dynamic=False), T.LOAD_TESTS["LT-1 eccentric (lanes 1-2)"], T.LT_FRONT_AXLE_X)
            d_pl = np.zeros(6)
            s_pl = np.zeros(6)
            if kind == "prestress_loss":
                d1, s1, _ = SY.prestress_loss_unit(gen.bg_static_h, 0)
                d_pl = d1 * scn.params["loss"] * gen.Pe / 1000
                s_pl = -s1 * scn.params["loss"] * gen.Pe / 1000 / C.GIRDER_CONCRETE["Ecm"] * 1e6
            pts = [p for p in gen.asb.points if p["girder"] == 1]
            dc = [SY.point_condition(scn, p) for p in pts]
            k_ax = C.GIRDER_CONCRETE["Ecm"] * 1e3 * 1.23 / C.L
            kh = st.bearing_kh.get((0, 1), T.bearing_stiffness()["kh"])
            rows.append(dict(kind=kind, severity=s, f_bending=fr[0], f_torsion=fr[1], f_transverse=fr[2],
                             lt_defl_G1=lt["deflection_mm"][0], lt_strain_G1=lt["soffit_strain_ue"][0],
                             camber_change_G1_mm=d_pl[0], vw_strain_change_G1_ue=s_pl[0],
                             prestress_cracking=bool(pl_crack),
                             upv_G1_min=min(4.6 * (1 - 0.3 * d) for d, _ in dc),
                             hcp_G1_min=min(-110 - 380 * c for _, c in dc),
                             brg_slope_G1=C.ALPHA_T * C.L * 1e3 * k_ax / (k_ax + kh)))
    return rows


def plots(df):
    kinds = SY.KINDS
    cols = plt.cm.tab10(np.arange(len(kinds)))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for k, c in zip(kinds, cols):
        d = df[df.kind == k]
        axs[0, 0].scatter(d.T_deck, d.f1, s=5, color=c, label=k, alpha=0.6)
        axs[0, 1].scatter(d.severity, d.lt_defl_G1, s=5, color=c, alpha=0.6)
        axs[0, 2].scatter(d.severity, d[["camber_res_G1", "camber_res_G3", "camber_res_G4", "camber_res_G6"]].max(axis=1), s=5, color=c, alpha=0.6)
        axs[1, 0].scatter(d.severity, d.hcp_min, s=5, color=c, alpha=0.6)
        axs[1, 1].scatter(d.severity, d.upv_girder_min, s=5, color=c, alpha=0.6)
        axs[1, 2].scatter(d.severity, d.vis_hi, s=5, color=c, alpha=0.6)
    axs[0, 0].set(xlabel="deck temperature (degC)", ylabel="identified f_bending (Hz)", title="f1 vs temperature: confounding")
    axs[0, 1].set(xlabel="severity", ylabel="load-test deflection G1 (mm)", title="LT-1 deflection")
    axs[0, 2].set(xlabel="severity", ylabel="max camber residual (mm)", title="levelling residual vs reference")
    axs[1, 0].set(xlabel="severity", ylabel="min half-cell potential (mV CSE)", title="half-cell (wet season shifts)")
    axs[1, 1].set(xlabel="severity", ylabel="min UPV on girders (km/s)", title="UPV")
    axs[1, 2].set(xlabel="severity", ylabel="visual health index", title="visual inspection")
    axs[0, 0].legend(fontsize=7, markerscale=2)
    for a in axs.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step5_synthetic_overview.png", dpi=140)
    plt.close(fig)


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 340
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(i * len(SY.KINDS) + k, kind, 10_000 + 7 * (i * len(SY.KINDS) + k))
            for i in range(n_per) for k, kind in enumerate(SY.KINDS)]
    t0 = time.time()
    rows, nds, obs = [], [], []
    with Pool(workers, initializer=_init) as pool:
        for k, (r, nd, ob) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            rows.append(r)
            nds += nd
            obs += ob
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{len(jobs)} campaigns, {time.time()-t0:.0f} s", flush=True)
    df = pd.DataFrame(rows).sort_values("campaign")
    df.to_csv(DATA / "campaigns.csv", index=False)
    pd.DataFrame(nds).sort_values(["campaign", "point"]).to_csv(DATA / "ndt_readings.csv.gz", index=False, compression="gzip")
    pd.DataFrame(obs).sort_values("campaign").to_csv(DATA / "visual_observations.csv", index=False)
    print(f"{len(df)} campaigns, {len(nds)} NDT readings, {len(obs)} visual observations in {time.time()-t0:.0f} s")

    gen = SY.Generator()
    ct = condition_table(gen)
    (OUT / "condition_table.json").write_text(json.dumps(ct, indent=1, default=float))
    print("\nNoise-free condition table (30 degC; G1 affected where applicable)")
    print(pd.DataFrame(ct).round(3).to_string(index=False))

    # example acceleration records
    rng = np.random.default_rng(5)
    env = dict(age_years=10.0, doy=60.0, hour=10.0, T_deck=30.0, wet=False, traffic_per_min=12.0)
    for kind in SY.KINDS:
        scn = SY.sample_scenario(kind, rng, severity=0.0 if kind == "healthy" else 0.8)
        _, _, _, (t, a) = gen.campaign(scn, env, seed=99, keep_record=True)
        with gzip.open(DATA / "accel" / f"{kind}_s0.8.csv.gz", "wt") as fh:
            fh.write("t," + ",".join(s.name for s in gen.sensors) + "\n")
            for k in range(len(t)):
                fh.write(f"{t[k]:.2f}," + ",".join(f"{v:.6f}" for v in a[:, k]) + "\n")
    print("example acceleration records written to outputs/data/accel/")

    print("\nMean measured features per class")
    feats = ["f1", "f2", "f3", "a_rms", "lt_defl_G1", "vw_res_G1", "camber_res_G1", "brg_slope_G1", "upv_girder_min",
             "hcp_min", "vis_hi"]
    print(df.groupby("kind")[feats].mean().round(3).to_string())
    plots(df)


if __name__ == "__main__":
    main()
