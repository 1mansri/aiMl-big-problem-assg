"""Maintenance decision support: fuse visual inspection, NDT, structural response, vibration and the
AI/ML assessment into one decision with mechanism-specific actions and a re-inspection interval.

Decision levels (problem statement 1.14):
  0 routine monitoring        1 enhanced monitoring      2 detailed investigation
  3 maintenance / repair      4 urgent: restrict loading and assess now
"""
import numpy as np
import pandas as pd

from . import ml

LEVELS = ["routine monitoring", "enhanced monitoring", "detailed investigation", "maintenance / repair",
          "urgent: restrict loading"]
STRUCTURAL = ("prestress_loss", "girder_cracking", "concrete_deterioration")

ACTIONS = {
    "prestress_loss": dict(
        investigate="drill-and-endoscope the ducts of the suspected girder at L/4, L/2, 3L/4; GPR / ultrasonic "
                    "tomography for grout voids; load test with crack-opening gauges (decompression load)",
        repair="vacuum re-grouting of voids; if the estimated loss exceeds ~10 % of Pe: external post-tensioning "
               "(deviators at the cross girders) designed for the lost force; reinstate monitoring targets",
        reinspect="levelling and VW gauges monthly for 2 years, load test after strengthening"),
    "girder_cracking": dict(
        investigate="crack mapping and crack gauges, UPV across cracks, load test; identify the overload "
                    "(weigh-in-motion records) and check ULS with the reduced stiffness",
        repair="epoxy injection of cracks wider than 0.2 mm; bonded CFRP on the soffit if the capacity check "
               "with the cracked section fails; enforce load limits",
        reinspect="crack gauges continuous; routine inspection every 6 months"),
    "concrete_deterioration": dict(
        investigate="cores (IS 516 Part 4) to calibrate rebound + UPV, petrography, extent mapping",
        repair="remove and patch deteriorated concrete, surface protection; strength assessment with in-situ "
               "strength before deciding on strengthening",
        reinspect="UPV / rebound grid at the next principal inspection"),
    "bearing_seized": dict(
        investigate="close visual of the free bearings, measure movement vs. temperature over a day",
        repair="clean debris, free or replace the bearing (jacks under the end cross girder, 4 per girder end)",
        reinspect="bearing-movement transducers continuous; check after the next hot / cold season"),
    "bearing_degraded": dict(
        investigate="measure bulging, splits and shear strain; Shore A hardness",
        repair="replace the elastomeric bearing (jacking under the end cross girder at the design jacking points)",
        reinspect="next routine inspection, then normal cycle"),
    "chloride_corrosion": dict(
        investigate="chloride profiles and half-cell grid on the end blocks under the joint; open one anchorage pocket",
        repair="replace the expansion-joint seal first (stop the source); patch repair of spalls; silane on end "
               "blocks; galvanic / impressed-current cathodic protection if potentials stay below -350 mV",
        reinspect="half-cell survey of end blocks 1 year after repair"),
    "healthy": dict(investigate="", repair="", reinspect="routine cycle (6-monthly routine, 3-yearly principal)"),
}
REINSPECT = {0: "routine cycle", 1: "targeted NDT within 3 months; monitoring review monthly",
             2: "investigation within 1 month", 3: "repair within 6 months; re-inspect after repair and at 1 year",
             4: "immediately"}


def assess(models, rows):
    """AI/ML assessment of campaigns (DataFrame with the raw campaign columns)."""
    X = ml.engineer(rows, models["baseline"])
    cols = models["features"]["all"]
    det = models["detector"]
    score = det.score(X[cols].values)
    proba = models["classifier"].predict_proba(X[cols])
    classes = list(models["classifier"].classes_)
    out = []
    for i in range(len(rows)):
        k = int(np.argmax(proba[i]))
        kind = classes[k]
        sev = float(models["severity"][kind].predict(X[cols].iloc[[i]])[0]) if kind in models["severity"] else 0.0
        pl = float(models["pl_regressor"].predict(X[models["features"]["monitoring"]].iloc[[i]])[0])
        out.append(dict(anomaly_score=float(score[i]), anomaly=bool(score[i] > det.thr), anomaly_threshold=det.thr,
                        kind=kind, p_kind=float(proba[i, k]), severity_est=float(np.clip(sev, 0, 1)),
                        prestress_loss_est_pct=max(0.0, pl),
                        df1_pct=float(X.df1_pct.iloc[i]), camber_res_max=float(X.camber_res_max.iloc[i]),
                        brg_slope_ratio_min=float(X.brg_slope_ratio_min.iloc[i]),
                        hcp_min_c=float(X.hcp_min_c.iloc[i]), upv_girder_min=float(X.upv_girder_min.iloc[i])))
    return out


def decide(a, row):
    """Decision level, mechanism and reasons from the ML assessment a and the raw campaign row."""
    reasons = []
    level = 0
    critical_cs = max(row["vis_cs_girders"], row["vis_cs_bearings"])
    kind = a["kind"] if a["anomaly"] or a["kind"] != "healthy" else "healthy"
    # evidence that does not need the model
    if critical_cs >= 4:
        level = 4
        reasons.append("visual CS4 on girders / bearings")
    if row["vis_cs_girders"] >= 3 or row["vis_cs_bearings"] >= 3 or row["vis_cs_joints"] >= 3:
        level = max(level, 2)
        reasons.append("visual CS3 on a structural component or joint")
    if a["hcp_min_c"] < -350:
        level = max(level, 2)
        reasons.append(f"half-cell {a['hcp_min_c']:.0f} mV (< -350: corrosion likely)")
    if a["upv_girder_min"] < 3.75:
        level = max(level, 2)
        reasons.append(f"UPV {a['upv_girder_min']:.2f} km/s (doubtful concrete)")
    if a["camber_res_max"] > 3.0:
        level = max(level, 2)
        reasons.append(f"camber {a['camber_res_max']:.1f} mm below the reference curve")
    if a["brg_slope_ratio_min"] < 0.8:
        level = max(level, 2)
        reasons.append(f"bearing movement {100*a['brg_slope_ratio_min']:.0f} % of alpha L per degC")
    # AI/ML assessment
    if a["anomaly"]:
        level = max(level, 1)
        reasons.append(f"anomaly score {a['anomaly_score']:.0f} > {a['anomaly_threshold']:.0f}")
        if a["kind"] != "healthy" and a["p_kind"] >= 0.5:
            reasons.append(f"ML: {a['kind']} (p = {a['p_kind']:.2f}), severity ~{a['severity_est']:.2f}")
            level = max(level, 2)
            if a["p_kind"] >= 0.6 and a["severity_est"] >= 0.4:
                level = max(level, 3)
            if a["kind"] in STRUCTURAL and a["severity_est"] >= 0.8:
                level = max(level, 4)
    if a["kind"] == "prestress_loss" and a["prestress_loss_est_pct"] > 20:
        level = max(level, 4)
        reasons.append(f"estimated prestress loss {a['prestress_loss_est_pct']:.0f} %")
    act = ACTIONS.get(kind, ACTIONS["healthy"])
    action = {0: act["reinspect"] if kind == "healthy" else ACTIONS["healthy"]["reinspect"],
              1: "enhanced monitoring; " + (act["investigate"] or "targeted NDT at flagged locations"),
              2: act["investigate"] or "targeted NDT at the CS3 locations and structural assessment",
              3: act["repair"] or "repair of the defects found",
              4: "restrict loading; " + (act["investigate"] or "structural assessment")}[level]
    return dict(level=level, decision=LEVELS[level], mechanism=kind, reasons=reasons or ["no evidence of change"],
                action=action, reinspect=REINSPECT[level])
