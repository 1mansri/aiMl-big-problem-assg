"""Step 6: AI/ML-assisted condition assessment on the synthetic campaigns.

  1. temperature / season compensation (healthy training data only)
  2. anomaly detection trained on healthy campaigns only (vibration-only vs. fused)
  3. damage-type classification per data source and fused
  4. regression: prestress loss from monitoring; severity per mechanism
  5. localisation of the damaged girder
  6. clustering without labels
Writes outputs/ml_summary.json, outputs/models.joblib and outputs/figures/step6_*.png.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, f1_score, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bridge import ml
from bridge.synthetic import KINDS

OUT = Path("outputs")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
FIG = OUT / "figures"
SEED = 42


def main():
    df = pd.read_csv(OUT / "data" / "campaigns.csv")
    tr, te = train_test_split(df.index, test_size=0.3, stratify=df.kind, random_state=SEED)
    train, test = df.loc[tr], df.loc[te]
    h_train = train[train.kind == "healthy"]
    res = {"n_train": len(train), "n_test": len(test)}

    # ------------------------------------------------------------------ 1. compensation
    base = ml.Baseline().fit(h_train)
    res["temperature_models"] = {k: dict(at_30C=v[0], per_degC=v[1], r2=v[2]) for k, v in base.temp_models.items()}
    res["wet_shift"] = base.wet_shift
    a, b, r2 = base.temp_models["f1"]
    print(f"Temperature compensation (healthy training data): f1 = {a:.4f} {b:+.5f} (T-30) Hz, "
          f"{100*b/a:+.3f} %/degC, R2 = {r2:.2f}")
    print(f"  wet-season shift of the minimum half-cell potential {base.wet_shift['hcp_min']:+.0f} mV; "
          f"resistivity x{np.exp(base.wet_shift['resistivity_min']):.2f}")
    X = ml.engineer(df, base)
    Xtr, Xte = X.loc[tr], X.loc[te]
    sd_raw = h_train.f1.std()
    sd_comp = X.loc[h_train.index, "df1_pct"].std() * a / 100
    print(f"  scatter of healthy f1: {sd_raw*1e3:.1f} mHz raw -> {sd_comp*1e3:.1f} mHz compensated")
    res["f1_healthy_sd_mHz"] = dict(raw=sd_raw * 1e3, compensated=sd_comp * 1e3)

    # signature table: mean engineered features per class (physical reasonableness)
    sig_cols = ["df1_pct", "df2_pct", "f2_over_f1", "lt_ratio_max", "lt_ratio_min", "camber_res_max", "vw_res_spread",
                "brg_slope_ratio_min", "upv_girder_min", "hcp_min_c", "hcp_end_minus_mid", "vis_hi"]
    sig = X.groupby(df.kind)[sig_cols].mean().loc[KINDS]
    res["signatures"] = sig.round(4).to_dict(orient="index")
    print("\nClass signatures (mean engineered features, all campaigns)")
    print(sig.round(3).to_string())

    # ------------------------------------------------------------------ 2. anomaly detection
    print("\nAnomaly detection, trained on healthy training campaigns only (threshold: 99th percentile)")
    anom = {}
    det_curves = {}
    for name, cols in (("vibration", ml.GROUPS["vibration"]), ("load_test", ml.GROUPS["load_test"]),
                       ("monitoring", ml.GROUPS["monitoring"]), ("ndt", ml.GROUPS["ndt"]),
                       ("all", ml.GROUPS["all"])):
        det = ml.MahalanobisDetector().fit(Xtr.loc[h_train.index, cols].values)
        s = det.score(Xte[cols].values)
        flag = s > det.thr
        y = (test.kind != "healthy").values
        fa = flag[~y].mean()
        per = {k: float(flag[(test.kind == k).values].mean()) for k in KINDS if k != "healthy"}
        anom[name] = dict(false_alarm=float(fa), detection=float(flag[y].mean()), auc=float(roc_auc_score(y, s)), per_class=per)
        det_curves[name] = (test.severity.values, flag, test.kind.values)
        print(f"  {name:10s} false alarms {100*fa:4.1f} %  detection {100*flag[y].mean():5.1f} %  AUC {anom[name]['auc']:.3f}  "
              + "  ".join(f"{k[:10]} {100*v:3.0f}%" for k, v in per.items()))
    ifm, ithr = ml.isolation_forest(Xtr.loc[h_train.index, ml.GROUPS["all"]].values)
    s_if = -ifm.score_samples(Xte[ml.GROUPS["all"]].values)
    y = (test.kind != "healthy").values
    anom["isolation_forest_all"] = dict(false_alarm=float((s_if > ithr)[~y].mean()), detection=float((s_if > ithr)[y].mean()),
                                        auc=float(roc_auc_score(y, s_if)))
    print(f"  isolation forest (all): false alarms {100*anom['isolation_forest_all']['false_alarm']:.1f} %, "
          f"detection {100*anom['isolation_forest_all']['detection']:.1f} %, AUC {anom['isolation_forest_all']['auc']:.3f}")
    res["anomaly"] = anom

    # ------------------------------------------------------------------ 3. classification per source
    print("\nDamage-type classification (random forest, test set)")
    cls = {}
    models = {}
    for name, cols in ml.GROUPS.items():
        m = ml.classifier(SEED).fit(Xtr[cols], train.kind)
        p = m.predict(Xte[cols])
        cls[name] = dict(macro_f1=float(f1_score(test.kind, p, average="macro")), accuracy=float((p == test.kind).mean()),
                         recall={k: float((p[(test.kind == k).values] == k).mean()) for k in KINDS})
        models[name] = m
        print(f"  {name:10s} macro-F1 {cls[name]['macro_f1']:.3f}  accuracy {cls[name]['accuracy']:.3f}  | recall "
              + " ".join(f"{k[:8]}={v:.2f}" for k, v in cls[name]["recall"].items()))
    res["classification"] = cls
    # severity dependence of the fused and vibration-only classifiers
    bins = [0.0, 0.01, 0.25, 0.5, 0.75, 1.0]
    sev_acc = {}
    for name in ("vibration", "all"):
        p = models[name].predict(Xte[ml.GROUPS[name]])
        ok = p == test.kind.values
        b = pd.cut(test.severity, bins, include_lowest=True)
        sev_acc[name] = {str(k): float(v) for k, v in pd.Series(ok, index=test.index).groupby(b, observed=True).mean().items()}
    res["accuracy_vs_severity"] = sev_acc
    print("  accuracy vs severity bin (vibration / all): " +
          "  ".join(f"{k}: {sev_acc['vibration'][k]:.2f}/{sev_acc['all'][k]:.2f}" for k in sev_acc["all"]))
    pi = permutation_importance(models["all"], Xte[ml.GROUPS["all"]], test.kind, n_repeats=5, random_state=SEED,
                                scoring="f1_macro", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=ml.GROUPS["all"]).sort_values(ascending=False)
    res["permutation_importance_all"] = imp.head(20).round(4).to_dict()
    print("  most important fused features (permutation, drop in macro-F1):")
    for f, v in imp.head(10).items():
        print(f"    {f:24s} {v:.3f}  {ml.PHYSICAL_MEANING.get(f, '')}")
    cm_all = confusion_matrix(test.kind, models["all"].predict(Xte[ml.GROUPS["all"]]), labels=KINDS)
    cm_vib = confusion_matrix(test.kind, models["vibration"].predict(Xte[ml.GROUPS["vibration"]]), labels=KINDS)
    res["confusion_all"] = cm_all.tolist()
    res["confusion_vibration"] = cm_vib.tolist()

    # ------------------------------------------------------------------ 4. regression
    print("\nRegression")
    reg = {}
    pl_tr, pl_te = train[train.kind == "prestress_loss"], test[test.kind == "prestress_loss"]
    for name, cols in (("monitoring", ml.GROUPS["monitoring"]), ("vibration", ml.GROUPS["vibration"]),
                       ("all", ml.GROUPS["all"])):
        r = ml.regressor(SEED).fit(Xtr.loc[pl_tr.index, cols], 100 * pl_tr.p_loss)
        pr = r.predict(Xte.loc[pl_te.index, cols])
        reg[f"prestress_loss_from_{name}"] = dict(mae_pct=float(mean_absolute_error(100 * pl_te.p_loss, pr)),
                                                  r2=float(r2_score(100 * pl_te.p_loss, pr)))
        if name == "monitoring":
            pl_pred = (100 * pl_te.p_loss.values, pr)
            models["pl_regressor"] = r
        print(f"  prestress loss (% of Pe) from {name:10s}: MAE {reg[f'prestress_loss_from_{name}']['mae_pct']:.2f} %, "
              f"R2 {reg[f'prestress_loss_from_{name}']['r2']:.2f}")
    sev_models = {}
    for k in KINDS[1:]:
        a_tr, a_te = train[train.kind == k], test[test.kind == k]
        r = ml.regressor(SEED).fit(Xtr.loc[a_tr.index, ml.GROUPS["all"]], a_tr.severity)
        pr = r.predict(Xte.loc[a_te.index, ml.GROUPS["all"]])
        reg[f"severity_{k}"] = dict(mae=float(mean_absolute_error(a_te.severity, pr)), r2=float(r2_score(a_te.severity, pr)))
        sev_models[k] = r
        print(f"  severity of {k:22s} (all sources): MAE {reg[f'severity_{k}']['mae']:.3f}, R2 {reg[f'severity_{k}']['r2']:.2f}")
    res["regression"] = reg
    models["severity"] = sev_models

    # ------------------------------------------------------------------ 5. localisation
    print("\nLocalisation of the affected girder")
    loc = {}
    for k, cols in (("prestress_loss", ml.GROUPS["monitoring"]),
                    ("girder_cracking", ml.GROUPS["load_test"]),
                    ("concrete_deterioration", ml.GROUPS["load_test"] + ml.GROUPS["ndt"])):
        a_tr, a_te = train[train.kind == k], test[test.kind == k]
        ytr = a_tr.girders.astype(str).str.split().str[0].astype(int)
        yte = a_te.girders.astype(str).str.split().str[0].astype(int)
        m = ml.classifier(SEED).fit(Xtr.loc[a_tr.index, cols], ytr)
        p = m.predict(Xte.loc[a_te.index, cols])
        within1 = float((abs(p - yte.values) <= 1).mean())
        loc[k] = dict(exact=float((p == yte.values).mean()), within_one_girder=within1)
        print(f"  {k:22s}: exact girder {100*loc[k]['exact']:.0f} %, within one girder {100*within1:.0f} %  (6 girders, chance 17 %)")
    res["localisation"] = loc

    # ------------------------------------------------------------------ 6. clustering
    sc = StandardScaler().fit(X[ml.GROUPS["all"]])
    Z = sc.transform(X[ml.GROUPS["all"]])
    km = KMeans(n_clusters=len(KINDS), n_init=20, random_state=SEED).fit(Z)
    ari = adjusted_rand_score(df.kind, km.labels_)
    ct = pd.crosstab(df.kind, km.labels_).loc[KINDS]
    res["clustering"] = dict(ari=float(ari), crosstab=ct.to_dict(orient="index"))
    print(f"\nK-means (k = 7) on all standardised features: adjusted Rand index vs. true classes {ari:.2f}")
    print(ct.to_string())

    (OUT / "ml_summary.json").write_text(json.dumps(res, indent=1, default=float))
    joblib.dump(dict(baseline=base, detector=ml.MahalanobisDetector().fit(X.loc[h_train.index, ml.GROUPS["all"]].values),
                     classifier=models["all"], pl_regressor=models["pl_regressor"], severity=sev_models,
                     features=ml.GROUPS), OUT / "models.joblib")
    plots(df, X, base, h_train, det_curves, cm_vib, cm_all, cls, imp, pl_pred, Z, km)


def plots(df, X, base, h_train, det_curves, cm_vib, cm_all, cls, imp, pl_pred, Z, km):
    # temperature compensation
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    h = df[df.kind == "healthy"]
    a, b, _ = base.temp_models["f1"]
    axs[0].scatter(h.T_deck, h.f1, s=6, alpha=0.6, label="healthy campaigns")
    tt = np.linspace(15, 46, 10)
    axs[0].plot(tt, a + b * (tt - 30), "k", label=f"fit: {100*b/a:+.2f} %/degC")
    axs[0].set(xlabel="deck temperature (degC)", ylabel="identified f1 (Hz)", title="Temperature dependence of f1")
    axs[0].legend()
    for k in ("healthy", "girder_cracking", "concrete_deterioration", "bearing_seized"):
        axs[1].hist(X.loc[df.kind == k, "df1_pct"], bins=40, alpha=0.5, label=k)
    axs[1].set(xlabel="compensated change of f1 (%)", ylabel="campaigns", title="After compensation")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step6_temperature_compensation.png", dpi=140)
    plt.close(fig)

    # detection probability vs severity
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(0, 1, 6)
    for ax, name in zip(axs, ("vibration", "all")):
        sev, flag, kind = det_curves[name]
        for k in KINDS[1:]:
            m = kind == k
            idx = np.digitize(sev[m], bins) - 1
            rate = [flag[m][idx == i].mean() if (idx == i).any() else np.nan for i in range(5)]
            ax.plot(0.5 * (bins[:-1] + bins[1:]), rate, "o-", label=k)
        ax.set(xlabel="severity", title=f"Anomaly detection: {name} features", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("probability of detection")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "step6_detection_vs_severity.png", dpi=140)
    plt.close(fig)

    # confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    short = [k.replace("_", "\n") for k in KINDS]
    for ax, cm, title in ((axs[0], cm_vib, f"vibration only (macro-F1 {cls['vibration']['macro_f1']:.2f})"),
                          (axs[1], cm_all, f"all sources fused (macro-F1 {cls['all']['macro_f1']:.2f})")):
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(KINDS)):
            for j in range(len(KINDS)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.6 * cm[i].sum() else "black")
        ax.set_xticks(range(len(KINDS)), short, fontsize=7)
        ax.set_yticks(range(len(KINDS)), short, fontsize=7)
        ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(FIG / "step6_confusion.png", dpi=140)
    plt.close(fig)

    # source comparison + importance
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    names = list(cls)
    axs[0].bar(names, [cls[n]["macro_f1"] for n in names], color=["#888"] * (len(names) - 1) + ["#1f5fa8"])
    axs[0].set(ylabel="macro-F1 (test)", title="Damage-type classification by data source", ylim=(0, 1))
    axs[0].grid(alpha=0.3, axis="y")
    top = imp.head(15)[::-1]
    axs[1].barh(top.index, top.values)
    axs[1].set(xlabel="drop in macro-F1 when permuted", title="Most informative fused features")
    axs[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "step6_sources_importance.png", dpi=140)
    plt.close(fig)

    # prestress loss regression + clustering
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    yt, yp = pl_pred
    axs[0].scatter(yt, yp, s=10)
    axs[0].plot([0, 32], [0, 32], "k--")
    axs[0].set(xlabel="true loss (% of Pe)", ylabel="estimated loss (%)", title="Prestress loss from levelling + VW gauges")
    axs[0].grid(alpha=0.3)
    P = PCA(2, random_state=SEED).fit_transform(Z)
    for k in KINDS:
        m = (df.kind == k).values
        axs[1].scatter(P[m, 0], P[m, 1], s=5, alpha=0.6, label=k)
    axs[1].set(xlabel="PC1", ylabel="PC2", title="All campaigns in the first two principal components")
    axs[1].legend(fontsize=7, markerscale=2)
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step6_regression_pca.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
