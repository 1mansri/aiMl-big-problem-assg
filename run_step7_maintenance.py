"""Step 7: maintenance scheme.

  a) decisions for every test campaign: is the fused decision logic safe (severe damage not left on routine
     monitoring) and economical (healthy bridges not sent to repair)?
  b) a 30-year life of this bridge, proactive owner (monitoring + NDT + ML, act on level >= 3) versus reactive
     owner (visual inspection only, act on CS4 or a visual health index below 60).
Writes outputs/maintenance_summary.json and outputs/figures/step7_*.png.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bridge import maintenance as MT
from bridge import synthetic as SY

OUT = Path("outputs")
FIG = OUT / "figures"
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# life story of the bridge: (mechanism, start year, severity growth per year or a step, scenario details)
STORY = [
    dict(kind="chloride_corrosion", start=8, rate=0.09, girders=tuple(range(6)), end=0),
    dict(kind="girder_cracking", start=18, step=0.45, girders=(1,), end=0),
    dict(kind="prestress_loss", start=23, rate=0.12, girders=(3,), end=0),
]
# relative costs (assumed, for comparison only): fixed part + severity-dependent part of the repair
COST = {"chloride_corrosion": (1.0, 12.0), "girder_cracking": (2.0, 15.0), "prestress_loss": (3.0, 30.0)}
C_INVESTIGATION, C_MONITORING, C_RESTRICTION = 0.5, 0.3, 5.0


def scenario_for(m, s):
    p = {"chloride_corrosion": dict(c=s, EI_end=1 - 0.2 * max(0.0, (s - 0.6) / 0.4)),
         "girder_cracking": dict(EI=1 - 0.45 * s, half_zone=6.0, crack_mm=0.05 + 0.5 * s),
         "prestress_loss": dict(loss=0.03 + 0.27 * s, cause="grout_void")}[m["kind"]]
    return SY.Scenario(m["kind"], s, m["girders"], end=m["end"], params=p)


def true_severity(m, yr, repaired):
    """Untreated growth of a mechanism; a repaired mechanism stays at 0 (its cause is removed)."""
    if m["kind"] in repaired or yr < m["start"]:
        return 0.0
    return m["step"] if "step" in m else min(1.0, m["rate"] * (yr - m["start"] + 1))


def life(gen, models, policy, years=30, seed=7):
    rng = np.random.default_rng(seed)
    repaired, hist, cost, pending = set(), [], 0.0, 0
    for yr in range(1, years + 1):
        sev = {m["kind"]: true_severity(m, yr, repaired) for m in STORY}
        active = max(STORY, key=lambda m: sev[m["kind"]])      # one mechanism per campaign: the dominant one
        s = sev[active["kind"]]
        scn = scenario_for(active, s) if s > 0 else SY.Scenario("healthy")
        env = SY.environment(rng, age_years=float(yr))
        row, _, _, _ = gen.campaign(scn, env, seed=1000 * seed + yr, record_s=300.0)
        a = MT.assess(models, pd.DataFrame([row]))[0]
        d = MT.decide(a, row)
        if policy == "proactive":
            cost += C_MONITORING + (C_INVESTIGATION if d["level"] >= 2 else 0.0)
            act = d["level"] >= 3 or (d["level"] == 2 and pending >= 1)      # level 2 twice: investigation confirms
            pending = pending + 1 if d["level"] == 2 else 0
        else:
            act = row["vis_cs_girders"] >= 4 or row["vis_cs_bearings"] >= 4 or row["vis_hi"] < 60
        if s > 0.8:
            cost += C_RESTRICTION                                         # severe damage forces load restriction
        rep = None
        if act and s > 0:
            c0, c1 = COST[active["kind"]]
            cost += c0 + c1 * s ** 2
            rep = active["kind"]
            repaired.add(rep)
            pending = 0
        hist.append(dict(year=yr, active=active["kind"] if s > 0 else "healthy", true_severity=s,
                         anomaly_score=a["anomaly_score"], threshold=a["anomaly_threshold"], ml_kind=a["kind"],
                         p_kind=a["p_kind"], level=d["level"], decision=d["decision"], repaired=rep,
                         vis_hi=row["vis_hi"], cost=cost, reasons="; ".join(d["reasons"])))
    h = pd.DataFrame(hist)
    # damage still present at the end of the horizon is a deferred cost, not a saving: charge its repair
    h.attrs["outstanding"] = sum(COST[m["kind"]][0] + COST[m["kind"]][1] * sev[m["kind"]] ** 2
                                 for m in STORY if sev[m["kind"]] > 0)
    return h


def main():
    models = joblib.load(OUT / "models.joblib")
    df = pd.read_csv(OUT / "data" / "campaigns.csv")
    _, te = train_test_split(df.index, test_size=0.3, stratify=df.kind, random_state=42)
    test = df.loc[te].reset_index(drop=True)
    A = MT.assess(models, test)
    D = [MT.decide(a, test.iloc[i]) for i, a in enumerate(A)]
    test["level"] = [d["level"] for d in D]
    test["sev_bin"] = pd.cut(test.severity, [-0.01, 0.0, 0.3, 0.6, 1.0], labels=["healthy", "low", "medium", "high"])
    tab = pd.crosstab([test.kind, test.sev_bin], test.level, normalize="index").round(2)
    print("Decision level distribution (rows: true condition; columns: 0 routine .. 4 urgent)")
    print(tab.to_string())
    healthy = test[test.kind == "healthy"]
    severe = test[(test.severity >= 0.6)]
    summary = dict(
        healthy_routine=float((healthy.level <= 1).mean()), healthy_to_repair=float((healthy.level >= 3).mean()),
        severe_left_routine=float((severe.level == 0).mean()), severe_investigated=float((severe.level >= 2).mean()),
        severe_structural_repair_or_urgent=float((severe[severe.kind.isin(MT.STRUCTURAL)].level >= 3).mean()),
        table={f"{k[0]}|{k[1]}": v for k, v in tab.to_dict(orient="index").items()})
    print(f"\nhealthy bridges kept on routine/enhanced monitoring: {100*summary['healthy_routine']:.0f} %, sent to repair "
          f"{100*summary['healthy_to_repair']:.1f} %")
    print(f"severe damage (s >= 0.6) left on routine monitoring: {100*summary['severe_left_routine']:.1f} %; "
          f"investigated or repaired: {100*summary['severe_investigated']:.0f} %")
    ex = D[int(np.argmax([(t == "prestress_loss") * s for t, s in zip(test.kind, test.severity)]))]
    summary["example_decision"] = ex
    print("\nExample decision (most severe prestress-loss test campaign):")
    for k, v in ex.items():
        print(f"  {k}: {v}")

    gen = SY.Generator()
    lives = {p: life(gen, models, p) for p in ("proactive", "reactive")}
    for p, h in lives.items():
        reps = h[h.repaired.notna()][["year", "repaired"]].values.tolist()
        print(f"\n{p} owner: repairs {reps}; max true severity {h.true_severity.max():.2f}; "
              f"spent {h.cost.iloc[-1]:.1f} + outstanding repair at year 30 {h.attrs['outstanding']:.1f} "
              f"= {h.cost.iloc[-1] + h.attrs['outstanding']:.1f} (relative units)")
    summary["life"] = {p: h.to_dict(orient="records") for p, h in lives.items()}
    summary["life_cost"] = {p: dict(spent=float(h.cost.iloc[-1]), outstanding=float(h.attrs["outstanding"]),
                                    total=float(h.cost.iloc[-1] + h.attrs["outstanding"]),
                                    max_severity=float(h.true_severity.max()))
                            for p, h in lives.items()}
    summary["story"] = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()} for m in STORY]
    summary["costs_assumed"] = dict(repair=COST, investigation=C_INVESTIGATION, monitoring_per_year=C_MONITORING,
                                    load_restriction=C_RESTRICTION)
    (OUT / "maintenance_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    plot_life(lives)


def plot_life(lives):
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    col = {"chloride_corrosion": "tab:pink", "girder_cracking": "tab:green", "prestress_loss": "tab:orange",
           "healthy": "tab:blue"}
    for p, ls in (("proactive", "-"), ("reactive", "--")):
        h = lives[p]
        axs[0].plot(h.year, h.true_severity, ls, color="k", lw=1.2, label=f"{p}: true severity of the active mechanism")
        for _, r in h[h.repaired.notna()].iterrows():
            axs[0].plot(r.year, 0.02, "v" if p == "proactive" else "x", color=col[r.repaired], ms=10)
        axs[2].plot(h.year, h.cost, ls, color="k", label=f"{p}: spent")
        if h.attrs["outstanding"] > 0:
            tot = h.cost.iloc[-1] + h.attrs["outstanding"]
            axs[2].plot([30, 30], [h.cost.iloc[-1], tot], color="tab:red", lw=4,
                        label=f"{p}: unrepaired damage at year 30 (total {tot:.0f})")
    hp = lives["proactive"]
    axs[1].semilogy(hp.year, hp.anomaly_score, "o-", ms=4, label="anomaly score (proactive owner's monitoring)")
    axs[1].axhline(hp.threshold.iloc[0], color="r", ls=":", label="threshold (99 % healthy)")
    for _, r in hp.iterrows():
        axs[1].annotate(str(r.level), (r.year, r.anomaly_score), textcoords="offset points", xytext=(0, 6), fontsize=7)
    for m in STORY:
        axs[0].axvline(m["start"], color=col[m["kind"]], ls=":", lw=1)
        axs[0].text(m["start"] + 0.2, 0.92, m["kind"].replace("_", " "), color=col[m["kind"]], fontsize=8)
    axs[0].set(ylabel="severity", ylim=(0, 1.05), title="30-year life: v = proactive repair, x = reactive repair")
    axs[0].legend(fontsize=8, loc="center left")
    axs[1].set(ylabel="Mahalanobis distance", title="AI/ML anomaly score each year (numbers: decision level)")
    axs[1].legend(fontsize=8)
    axs[2].set(xlabel="year", ylabel="cumulative relative cost", title="Cost (assumed relative units)")
    axs[2].legend()
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step7_life.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
