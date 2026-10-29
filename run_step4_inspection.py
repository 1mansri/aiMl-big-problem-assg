"""Step 4: inspection scheme, NDT programme and the digital inspection app.

Writes outputs/inspection_scheme.json, outputs/data/ndt_locations.csv, a figure of the NDT grid
and builds inspection_app/index.html (offline, single file) from template.html + rules.js.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bridge import config as C
from bridge import inspection as I

OUT = Path("outputs")
APP = Path("inspection_app")

EXAMPLE_RECORDS = [
    # component, location, defect, inputs, notes  (an illustrative principal inspection in year 12)
    ("Expansion joints", "EJ-A1/carriageway", "joint seal torn / leaking", {"length_m": 3.5},
     "seal torn in lane 2; water dripping onto bearing shelf after rain"),
    ("PSC girders", "G1/End-A1/web", "leakage / efflorescence", {"area_m2": 0.6, "state": "efflorescence"},
     "white deposits below the joint, end block of G1"),
    ("PSC girders", "G2/End-A1/anchorage", "leakage / efflorescence", {"area_m2": 0.3, "state": "active leakage"},
     "damp anchorage pocket"),
    ("Bearings", "BRG-A1-G1", "bearing shear deformation", {"shear_mm": 30, "elastomer_mm": 48},
     "measured at 34 degC deck temperature"),
    ("Bearings", "BRG-P1-G6", "bearing bulging / splitting", {"state": "uneven bulging"}, ""),
    ("Drainage", "D2-L", "drain blocked", {"state": "fully"}, "ponding near kerb"),
    ("Deck slab", "B1/L/2/soffit", "crack", {"width_mm": 0.15, "length_m": 1.8, "orientation": "map / random",
                                              "rust stain": False}, "shrinkage pattern"),
    ("PSC girders", "G3/L/2/bottom flange", "crack", {"width_mm": 0.05, "length_m": 0.4,
                                                       "orientation": "flexural (transverse, soffit)", "rust stain": False},
     "hairline; record for the next inspection"),
    ("Wearing coat", "Lane2/L/2", "potholes / ravelling", {"area_m2": 0.8, "depth_mm": 30}, ""),
    ("Crash barriers", "Barrier-L/L/4", "impact damage", {"length_m": 2.0, "severity": "scrape"}, ""),
    ("Pier P1", "pier cap top", "leakage / efflorescence", {"area_m2": 1.2, "state": "efflorescence"},
     "water from joint EJ-P1 not reaching the drain"),
]


def example_state():
    comps = I.components()
    recs = []
    for k, (c, loc, d, inp, notes) in enumerate(EXAMPLE_RECORDS):
        recs.append(dict(id=k + 1, time="2038-11-20T10:00:00", component=c, location=loc, defect=d, inputs=inp,
                         cs=I.condition_state(d, comps[c]["material"], inp), cs_override="", reason="",
                         notes=notes, photo=None))
    return dict(meta=dict(date="2038-11-20", inspector="(example)", type="principal (detailed)", weather="dry, 26 degC"),
                records=recs)


def build_app(scheme, example):
    html = (APP / "template.html").read_text(encoding="utf-8")
    rules = (APP / "rules.js").read_text(encoding="utf-8").replace("if (typeof module", "if (false && typeof module")
    html = html.replace("/*@@RULES@@*/", rules)
    html = html.replace("/*@@SCHEME@@*/", json.dumps(scheme))
    html = html.replace("/*@@EXAMPLE@@*/", json.dumps(example))
    (APP / "index.html").write_text(html, encoding="utf-8")


def main():
    scheme = I.scheme()
    (OUT / "inspection_scheme.json").write_text(json.dumps(scheme, indent=1))
    comps = scheme["components"]
    n_loc = sum(len(v["locations"]) for v in comps.values())
    print(f"Inspection hierarchy: {len(comps)} components, {n_loc} locations, {len(I.DEFECTS)} observation types")
    for c, v in comps.items():
        print(f"  {c:18s} weight {v['weight']:.1f}  {len(v['locations']):4d} locations  ({v['material']})")
    print("Inspection types:")
    for t in I.INSPECTION_TYPES:
        print(f"  {t['type']:22s} {t['interval']}")

    ex = example_state()
    worst = {}
    for r in ex["records"]:
        worst[r["component"]] = max(worst.get(r["component"], 1), r["cs"])
    used = {c: worst.get(c, 1) for c in comps}
    hi = I.health_index(used)
    crit = max(worst.get(c, 1) for c in I.CRITICAL)
    print(f"\nExample inspection (year 12): {len(ex['records'])} observations, HI = {hi:.0f} -> {I.action(hi, crit)}")
    for r in sorted(ex["records"], key=lambda r: -r["cs"]):
        print(f"  CS{r['cs']}  {r['component']:16s} {r['location']:24s} {r['defect']}")

    pts = I.ndt_locations()
    with open(OUT / "data" / "ndt_locations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "component", "girder", "bay", "zone", "x", "face", "material"])
        w.writeheader()
        for p in pts:
            w.writerow({k: p.get(k) for k in w.fieldnames})
    print(f"\nNDT baseline grid: {len(pts)} test points (outputs/data/ndt_locations.csv); "
          f"{len(I.NDT_PROGRAMME)} defect -> question -> test chains")

    build_app(scheme, ex)
    print(f"App: {APP / 'index.html'} ({(APP / 'index.html').stat().st_size / 1e3:.0f} kB)")
    plot_ndt(pts)


def plot_ndt(pts):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.add_patch(plt.Rectangle((0, 0), C.L, C.B_DECK, fill=False))
    for y in C.GIRDER_Y:
        ax.plot([0, C.L], [y, y], color="#aaa", lw=1)
    bay_y = {f"B{k+1}": 0.5 * (C.GIRDER_Y[k] + C.GIRDER_Y[k + 1]) for k in range(C.N_GIRDERS - 1)}
    bay_y.update({"Cantilever-L": 0.5, "Cantilever-R": C.B_DECK - 0.5})
    for p in pts:
        if p["girder"]:
            y = C.GIRDER_Y[p["girder"] - 1] + (0.25 if p["face"] == "web" else -0.25)
            ax.plot(p["x"], y, "s" if p["face"] == "soffit" else "^", color="tab:blue" if p["face"] == "soffit" else "tab:orange", ms=5)
        else:
            ax.plot(p["x"], bay_y[p["bay"]], "o", color="tab:green", ms=5)
    ax.plot([], [], "s", color="tab:blue", label="girder soffit (bottom flange)")
    ax.plot([], [], "^", color="tab:orange", label="girder web")
    ax.plot([], [], "o", color="tab:green", label="deck soffit")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="NDT baseline grid: rebound, UPV, cover, half-cell, resistivity at each point")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "step4_ndt_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
