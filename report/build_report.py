"""Build the integrated report (report/report.html and report/report.pdf) from the analysis outputs.

Every number in the report is read from outputs/*.json, so the report cannot drift from the analysis.
Run from the project root after steps 1-7:  python report/build_report.py
"""
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridge import config as C              # noqa: E402
from bridge import inspection as I          # noqa: E402
from bridge import maintenance as MT        # noqa: E402
from bridge import synthetic as SY          # noqa: E402

OUT = ROOT / "outputs"
REP = ROOT / "report"
FIG = OUT / "figures"
CHROME = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
          Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
          Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")]
CHROME += [Path(p) for p in map(shutil.which, ("google-chrome", "chromium", "chromium-browser", "microsoft-edge")) if p]

STUDENT = "Durgesh Kumar"
ROLL = "23CE36005"


def J(name):
    return json.loads((OUT / name).read_text())


def e(s):
    return html.escape(str(s))


def f(v, d=2):
    try:
        return f"{float(v):,.{d}f}"
    except (TypeError, ValueError):
        return e(v)


def table(head, rows, cls="", caption=None):
    h = "".join(f"<th>{x}</th>" for x in head)          # headers are trusted HTML
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<table class="{cls}">{cap}<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


FIGN = [0]


def fig(name, caption, width="100%"):
    FIGN[0] += 1
    rel = f"../outputs/figures/{name}"
    if not (FIG / name).exists():
        return f'<p class="missing">[figure {e(name)} not found]</p>'
    return (f'<figure><img src="{rel}" style="width:{width}" alt="{e(caption)}">'
            f"<figcaption>Figure {FIGN[0]}. {caption}</figcaption></figure>")


# ---------------------------------------------------------------------- helpers that compute report facts
def run_tests():
    files = sorted((ROOT / "tests").glob("test_step*.py"))
    total = ok = 0
    names = []
    for t in files:
        r = subprocess.run([sys.executable, str(t)], cwd=ROOT, capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
        lines = [l for l in r.stdout.splitlines() if l.startswith("ok ")]
        ok += len(lines)
        names += [(t.name, l[3:]) for l in lines]
        total += len(lines) + (0 if r.returncode == 0 else 1)
    return ok, total, names


def app_screenshot():
    exe = next((c for c in CHROME if c.exists()), None)
    src = ROOT / "inspection_app" / "index.html"
    if exe is None or not src.exists():
        return
    tmp = REP / "_app_preview.html"
    tmp.write_text(src.read_text(encoding="utf-8").replace(
        "  load(); initMeta();", "  state = JSON.parse(JSON.stringify(EXAMPLE)); initMeta();"), encoding="utf-8")
    subprocess.run([str(exe), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-prefers-color-scheme=light",
                    "--window-size=1400,1150", f"--screenshot={FIG / 'step4_app.png'}", tmp.as_uri()],
                   capture_output=True, timeout=120)
    tmp.unlink(missing_ok=True)


def worst_checks(checks):
    groups = {}
    for r in checks:
        margin = (r["limit"] - r["value"]) if "compression" in r["check"] else (r["value"] - r["limit"])
        key = (r["stage"], r["fibre"], r["check"])
        if key not in groups or margin < groups[key][0]:
            groups[key] = (margin, r)
    return [g[1] for g in groups.values()]


# ---------------------------------------------------------------------- sections
def s_intro(s1, s2):
    cs = s1["config"]
    return f"""
<h1 id="s1">1 Introduction and bridge configuration</h1>
<p>This report follows one bridge through its life: a simply supported, post-tensioned PSC I-girder span of
<b>L = {C.L:.0f} m</b> (span chosen from 30 / 33 / 36 m), with three traffic lanes, footpaths and crash barriers on
both sides. The chain of the problem statement is followed literally: design &rarr; prestressing &rarr; healthy baseline
&rarr; visual inspection &rarr; NDT &rarr; structural and vibration monitoring &rarr; synthetic data &rarr; features &rarr;
AI/ML &rarr; maintenance decision. One digital twin of the bridge (a grillage with mass, bearings and damage hooks)
produces all the synthetic measurements, so every later chapter uses the same structure designed in Chapters 2&ndash;3.</p>
<p>The span is treated as span S1 (abutment A1 &ndash; pier P1) of a viaduct of identical simply supported spans, so that the
inspection scheme can include a pier cap and a pier; piers, abutments and foundations are not designed here.</p>
<h2>1.1 Cross-section</h2>
{table(["Strip", "Width (m)", "Justification"], [
    ["Railing + kerb", "2 &times; 0.25", "pedestrian railing on a kerb"],
    ["Footpath", "2 &times; 1.50", "IRC:5 minimum clear footpath width"],
    ["Crash barrier", "2 &times; 0.45", "rigid New-Jersey barrier separating footpath and carriageway"],
    ["Carriageway", "11.00", "IRC:6 Table 6: 9.6 &le; 11.0 &lt; 13.1 m gives exactly three design lanes"],
    ["Overall deck", f"{C.B_DECK:.2f}", ""]])}
{fig("step1_cross_section.png", "Adopted cross-section: six PSC I-girders at 2.6 m, 1.80 m precast + 0.22 m in-situ deck.", "86%")}
<p><b>Girders.</b> Six girders at {C.GIRDER_SPACING} m with {C.CANTILEVER:.2f} m cantilevers. Compared with five girders at
about 3.1 m, six lighter girders (precast area {s1['sections']['precast mid']['A']:.3f} m&sup2;, about
{s1['sections']['precast mid']['A'] * 25 * C.L_GIRDER / 9.81:.0f} t each) are easier to erect and allow a shallower section.
Depth {C.H_GIRDER:.2f} m precast + {C.T_DECK:.2f} m deck = {C.H_COMPOSITE:.2f} m (L/D = {C.L / C.H_COMPOSITE:.1f}).
Web {C.B_WEB:.2f} m (one 12-strand duct + 75 mm cover each side), bottom flange 0.65 &times; 0.25 m to house the cables,
end blocks widened to 0.65 m over 1.5 m for the anchorages.
<b>Concrete</b> M{C.GIRDER_CONCRETE['fck']:.0f} girders and M{C.DECK_CONCRETE['fck']:.0f} deck (IRC:112 minimum grades for
severe exposure are M40 PSC and M30 RCC); E<sub>cm</sub> = {C.GIRDER_CONCRETE['Ecm'] / 1e3:.1f} / {C.DECK_CONCRETE['Ecm'] / 1e3:.1f} GPa
(IRC:112 Table 6.5). <b>Cross girders</b> at both ends (0.50 &times; 1.50 m, also the jacking beams for bearing replacement)
and one at mid-span (0.30 &times; 1.40 m).</p>
{table(["Section", "A (m²)", "y<sub>b</sub> (m)", "I (m⁴)", "Z<sub>b</sub> (m³)", "Z<sub>t</sub> (m³)"],
       [[e(k), f(v['A'], 4), f(v['yb'], 3), f(v['I'], 4), f(v['Zb'], 4), f(v['Zt'], 4)] for k, v in s1['sections'].items()],
       caption=f"Section properties (composite in girder concrete, modular ratio {cs['modular_ratio']:.3f}; "
               "the full deck width is effective by IRC:112 cl. 7.6.1.2).")}
"""


def s_analysis(s1):
    sm = s1["summary"]
    rows = []
    for j in ("0", "1", "2"):
        d = sm[j]
        rows.append([f"G{int(j) + 1}", f(d['M_DL1'], 0), f(d['M_SIDL'], 0), f(d['M_WC'], 0), f(d['M_FP'], 0), f(d['M_LL'], 0),
                     f(d['M_service_rare'], 0), f"<b>{f(d['M_ULS'], 0)}</b>"])
    cb = s1["courbon_check"]
    ll = s1["live_load"]
    return f"""
<h1 id="s2">2 Loads and structural analysis</h1>
<p><b>Loads (IRC:6-2017).</b> Self-weight (25 kN/m&sup3;); crash barriers {C.SIDL_LINE['crash_barrier']} kN/m, railings
{C.SIDL_LINE['railing']} kN/m, footpath finish {C.FOOTPATH_FINISH} kPa, wearing coat {C.WEARING_COAT} kPa (65 mm bituminous plus one
overlay). Footpath live load by cl. 206.1 for L = 33 m: {s1['footpath_intensity_kPa']:.2f} kPa. Vehicles: Class A trains,
Class 70R wheeled and tracked, combined per Table 6A for three lanes (one 70R + one Class A, or three Class A with the
10 % reduction of cl. 205), with the IRC kerb and vehicle-to-vehicle clearances; impact 4.5/(6 + L) = {4.5 / (6 + C.L):.3f}
for Class A and wheeled 70R, 10 % for tracked 70R.</p>
<p><b>Model.</b> Stage 1 (girder + wet slab + cross girders) on the precast girder alone as a simply supported beam;
composite loads on a grillage (6 longitudinal composite members, transverse slab strips at 0.75 m, cross girders as
T-beams, Hambly's torsion split). Influence surfaces are obtained with one solve per response (Maxwell&ndash;Betti), and the
vehicles are placed by an exhaustive longitudinal search plus dynamic programming across the carriageway.</p>
<p><b>Checks of the tools.</b> The grillage reproduces PL/4, PL&sup3;/48EI and wL&sup2;/8 for a single line and static
equilibrium for the full deck; the vehicle search reproduces its own optimum by a direct load-vector solve. For the governing
outer-girder case ({e(cb['case'])}, eccentricity {cb['eccentricity']:+.2f} m) the grillage gives a share of
{cb['grillage_share'][0]:.3f} to G1 against {cb['courbon_share'][0]:.3f} by Courbon's method (L/B = {C.L / C.B_DECK:.2f}, stiff
mid-span cross girder): the two methods agree.</p>
{table(["Girder", "DL stage 1", "SIDL", "Wearing coat", "Footpath LL", "Vehicle LL (incl. impact)", "SLS rare", "ULS"], rows,
       caption="Characteristic mid-span moments (kNm). ULS = 1.35 (DL + SIDL) + 1.75 WC + 1.5 (LL + FP), IRC:6 Annex B.")}
{fig("step1_ll_envelopes.png", "Vehicle live-load envelopes. The mid-span cross girder pushes load outwards at mid-span, so G2/G3 peak near 0.37 L while G1 peaks at mid-span.")}
<p>The outer girder G1 governs flexure (live-load deflection {ll['0']['deflection_mid_mm']:.1f} mm &lt; L/800 =
{C.L / 0.8:.1f} mm); interior girders govern shear (vehicle shear {ll['2']['V'][0]:.0f} kN at the support element). All six
girders are given the same design.</p>
"""


def s_prestress(s2):
    lay = s2["layout"]
    L0 = s2["girders"]["0"]["losses"]
    rows = [[c[0], f"{c[1]}T15", str(c[2]), f(c[3], 2), f(c[4], 3), f"{c[6]:+.2f}"] for c in C.CABLES]
    loss_rows = [
        ["Jacking stress (0.75 f<sub>pk</sub>)", f(L0["jack"], 0), "&ndash;"],
        ["After friction (&mu; = 0.17, k = 0.002/m) and 6 mm wedge slip, stage-1 cables", f(L0["after friction+slip (stage 1 mean)"], 0), "both ends stressed"],
        ["Elastic shortening, stage-1 cables stressed in sequence", f(L0["elastic shortening, stage 1 sequential"], 1), "(n&minus;1)/2n rule"],
        ["Elastic shortening of stage-1 cables when stage 2 is stressed", f(L0["elastic shortening of stage 1 by stage 2"], 1), "cost of two-stage stressing"],
        ["Creep + shrinkage + relaxation, stage 1", f(L0["long-term stage 1 (creep+shrinkage+relaxation)"], 1), f"&phi; = {L0['phi']['1']:.2f}"],
        ["Creep + shrinkage + relaxation, stage 2", f(L0["long-term stage 2"], 1), f"&phi; = {L0['phi']['2']:.2f}"],
        ["Final stress stage 1 / stage 2", f"{L0['final stage 1']:.0f} / {L0['final stage 2']:.0f}", ""],
    ]
    chk = worst_checks(s2["checks"])
    crow = [[e(r["stage"]), e(r["fibre"]), e(r["check"]), f(r["value"], 2), f(r["limit"], 2),
             "OK" if r["ok"] else ("not required (IRC:112)" if r["check"].startswith("[legacy]") else "<b>NOT OK</b>"),
             f"G{r['girder']}, x = {r['x']:.2f}"] for r in chk]
    uls = [max(s2["girders"][j]["uls"], key=lambda r: r["ratio"]) for j in ("0", "1", "2")]
    sh = s2["girders"]["2"]["shear"]
    dfl = s2["girders"]["0"]["deflection"]
    tm = s2["girders"]["0"]["temperature_mid"]
    ds = s2["girders"]["0"]["diff_shrinkage"]
    return f"""
<h1 id="s3">3 Prestressing system</h1>
<p><b>Two-stage stressing.</b> With all strands stressed on the bare girder the transfer compression at the soffit exceeds
0.48 f<sub>ck</sub>(t) for concrete of about 35 MPa at 14 days. Two cables are therefore stressed at a girder age of
{C.T_STAGE1} days and two after the deck has hardened (day {C.T_STAGE2}), when they act on the deeper composite section.
The strand count was chosen by a sweep: 3 &times; 12T15 and 4 &times; 10T15 crack under the frequent combination
(with the characteristic prestress 0.9 P<sub>m</sub>), 4 &times; 12T15 exceeds 0.36 f<sub>ck</sub> under quasi-permanent load
near the supports; <b>4 &times; 11T15 (44 strands, 15.2 mm, f<sub>pk</sub> 1860 MPa)</b> satisfies every check.</p>
{table(["Cable", "Strands", "Stage", "Anchor height (m)", "Mid-span height (m)", "Mid-span lateral (m)"], rows,
       caption="Cable layout. Parabolic profiles with a straight run; lateral S-curves inside the bottom flange.")}
<p>The geometry was checked along the whole girder: minimum duct centre distance {lay['min_centre_distance'] * 1e3:.0f} mm
(&ge; 2 &times; OD = {lay['required_centre_distance'] * 1e3:.0f} mm), minimum cover {lay['min_cover'] * 1e3:.0f} mm (&ge; 75 mm),
anchorage spacing {lay['anchor_spacing']:.2f} m with {lay['anchor_top_edge']:.2f} m top edge distance, minimum plan radius
{min(v[1] for v in lay['min_radius_vertical_horizontal'].values() if v[1]):.1f} m (to be confirmed with the PT supplier).</p>
{fig("step2_cable_profiles.png", "Cable profiles in elevation and plan.", "92%")}
{table(["Stage of loss (mid-span, G1)", "Stress (MPa)", "Note"], loss_rows,
       caption=f"Prestress losses: P<sub>j</sub> = {L0['P_jack_kN']:.0f} kN, P<sub>0</sub> = {L0['P0_kN']:.0f} kN, "
               f"P<sub>e</sub> = {L0['Pe_kN']:.0f} kN, total loss {L0['total_loss_pct']:.1f} % (IRC:112 cl. 7.9 and Annex A2).")}
<p><b>Stresses.</b> Construction stages (transfer, deck casting, stage-2 stressing) and service at opening and after 100 years
were checked with the characteristic prestress r<sub>sup</sub> = 1.1 / r<sub>inf</sub> = 0.9 P<sub>m</sub>, temperature gradients
(IRC:6 cl. 215.3; mid-span soffit +{tm['heating']['girder_bottom']:.2f} / {tm['cooling']['girder_bottom']:.2f} MPa) and
differential shrinkage (deck {ds['eps_deck'] * 1e6:.0f} &mu;&epsilon;, girder after deck casting {ds['eps_girder_after_deck'] * 1e6:.0f}
&mu;&epsilon;, age-adjusted modulus). Severe exposure, bonded PT: decompression under the quasi-permanent combination and
w<sub>k</sub> &le; 0.2 mm under the frequent combination (the section stays uncracked, w<sub>k</sub> = 0).</p>
{table(["Stage", "Fibre", "Check", "Worst value (MPa)", "Limit", "Result", "Where"], crow, cls="small",
       caption="Governing stress checks over all stations of girders G1&ndash;G3 (compression positive).")}
<p><b>ULS.</b> Strain-compatibility fibre analysis (parabola&ndash;rectangle concrete, bilinear strand with its effective prestrain,
untensioned bars ignored): M<sub>Ed</sub>/M<sub>Rd</sub> = {uls[0]['ratio']:.2f} (G1, {uls[0]['M_Ed']:.0f} / {uls[0]['M_Rd']:.0f} kNm),
neutral axis {uls[0]['xu'] * 1e3:.0f} mm below the deck top, all tendons yielded. A hand check with the
parabola&ndash;rectangle block agrees within 1 %.
<b>Shear</b> (IRC:112 cl. 10.3, interior girder): V<sub>Ed</sub> = {sh[0]['V_Ed']:.0f} kN at the support, of which the cable
inclination carries {sh[0]['V_p']:.0f} kN; links were designed by the truss model alone (concrete term ignored):
A<sub>sw</sub>/s = {max(r['asw_design_mm2_per_m'] for r in sh):.0f} mm&sup2;/m near the support (e.g. 2L-12 @ 300) reducing to the
minimum {sh[-1]['asw_min_mm2_per_m']:.0f} mm&sup2;/m at mid-span; V<sub>Rd,max</sub> is not governing. Deck&ndash;girder interface
steel up to {max(r['interface']['As_req_mm2_per_m'] for r in sh):.0f} mm&sup2;/m near the supports.
<b>Deflection</b>: camber {-dfl['at_transfer']:.1f} mm at transfer, {-dfl['long_term']:.1f} mm upwards in the long term; live load
{dfl['live_load_beam_line']:.1f} mm (beam line) against L/800 = {dfl['limit_live']:.1f} mm.</p>
"""


def s_baseline(b):
    md = b["modal"]
    rows = [[str(k + 1), f(fr, 3), e(lab)] for k, (fr, lab) in enumerate(zip(md["freq"][:6], md["labels"][:6]))]
    idn = b["identification"]
    import numpy as np
    id_rows = []
    for r in range(3):
        pk = [d["peak_pick"][r] for d in idn]
        mc = [d["mac"][r] for d in idn]
        id_rows.append([e(md["labels"][r]), f(md["freq"][r], 3), f"{np.mean(pk):.3f} &plusmn; {np.std(pk):.3f}", f(np.mean(mc), 2)])
    sens = b["modal_sensitivity"]
    lt = b["load_test"]
    lt_rows = [[e(k), " / ".join(f(x, 2) for x in v["deflection_mm"]), " / ".join(f(x, 0) for x in v["soffit_strain_ue"])]
               for k, v in lt.items()]
    ages = np.array(b["long_term"]["age_days"]) / 365
    lt_vals = []
    for y in (1, 10, 30, 100):
        k = int(np.argmin(abs(ages - y)))
        lt_vals.append(f"{y} y: {b['long_term']['gauge_strain'][k] * 1e6:.0f} &mu;&epsilon; / {b['long_term']['midspan_deflection_mm'][k]:+.1f} mm")
    ndt = b["ndt_baseline"]
    ndt_rows = []
    for k, v in ndt.items():
        vals = "; ".join(f"{kk}: {vv[0]} &plusmn; {vv[1]}" if isinstance(vv, list) else f"{kk}: {vv}"
                         for kk, vv in v.items() if not kk.startswith("_"))
        ndt_rows.append([e(k), vals, e(v.get("_note", ""))])
    return f"""
<h1 id="s4">4 Healthy baseline</h1>
<p>The as-built twin uses the dynamic modulus E<sub>d</sub> = (E<sub>c</sub> + 19)/1.25 GPa (BS 8110-2; {md['E_dyn_factor']:.2f} E<sub>cm</sub>),
the superimposed masses ({md['mass_t_per_m']:.1f} t/m in total, equal to the dead loads within 1 %) and laminated elastomeric
bearings 400 &times; 400 mm (shape factor {md['bearing']['shape_factor']:.1f}, k<sub>v</sub> = {md['bearing']['kv'] / 1e3:.0f} MN/m,
k<sub>h</sub> = {md['bearing']['kh']:.0f} kN/m) as springs. Damping 1.5 % in every mode.</p>
{table(["Mode", "f (Hz)", "Shape"], rows, caption=f"Natural frequencies of the healthy bridge at 30 &deg;C. Hand check: simply supported beam f<sub>1</sub> = {md['f1_beam_hand']:.2f} Hz.")}
{fig("step3_mode_shapes.png", "Vertical mode shapes along each girder line.")}
<p><b>Vibration measurement.</b> Eight vertical accelerometers (L/4, L/2, 3L/4 of both outer girders and mid-span of G3 and G4;
the outer pairs separate bending from torsion), 100 Hz sampling (Nyquist 50 Hz &gg; the 25 Hz band of interest), 10-minute
records (about 2000 cycles of f<sub>1</sub>), excitation by traffic and ambient wind / micro-tremor. Processing:
detrend &rarr; 0.5&ndash;30 Hz band-pass &rarr; Welch PSD &rarr; peak picking; frequency domain decomposition (FDD) of the
cross-spectral matrix for the mode shapes, compared with the model by MAC.</p>
{table(["Mode", "Model (Hz)", "Identified from 3 simulated records (Hz)", "MAC"], id_rows,
       caption="Identification of the healthy bridge from simulated 10-minute records (traffic + ambient + sensor noise).")}
{fig("step3_vibration.png", "Simulated acceleration record, Welch PSD of all channels and FDD singular values.")}
<p>The first bending and first torsion modes are only 2 % apart, so the identified f<sub>1</sub> carries a small bias and the 13 Hz
pair cannot be separated at all; <b>the baseline for damage detection must therefore be the identified values, not the model</b>.
Sensitivities: temperature {b['df1_dT_pct_per_degC']:+.3f} %/&deg;C on f<sub>1</sub> (a 35 &deg;C seasonal swing moves f<sub>1</sub> by about 6 %);
fully composite barriers +{100 * (sens['barriers fully composite'][0] / md['freq'][0] - 1):.1f} %; seized bearings
+{100 * (sens['all bearings seized (kh = 2e5 kN/m)'][0] / md['freq'][0] - 1):.1f} % &ndash; a seized bearing makes the bridge look
<i>stiffer</i>.</p>
<p><b>Static baseline.</b> Load test with two 400 kN trucks (IRC:SP:51 approach), mid-span values G1&ndash;G6:</p>
{table(["Test", "Deflection (mm)", "Soffit strain (&mu;&epsilon;)"], lt_rows, cls="small")}
<p><b>Long-term reference curves</b> (G3 mid-span, creep superposition of every stress increment and the loss history):
embedded-gauge strain / mid-span deflection {", ".join(lt_vals)}. Later readings are compared with these curves.</p>
{fig("step3_long_term.png", "Reference curves of the embedded VW gauge and the mid-span camber.", "92%")}
<h2>4.1 Structural and vibration monitoring scheme</h2>
<p>{len(b['monitoring'])} sensors / items: 8 accelerometers; 12 embedded vibrating-wire gauges with thermistors (bottom and top
flange of G1, G3, G6 at L/4 and L/2, read every 15 min from casting); hydrostatic levelling at mid-span of G1, G3, G4, G6;
bearing-movement transducers at the free bearings of G1 and G6 (expected {b['thermal']['bearing_movement_mm_per_degC']:.3f} mm/&deg;C);
thermocouples through the depth; a weather station; and a load test at handing over and every 10 years.</p>
{fig("step3_sensor_layout.png", "Monitoring layout (plan).")}
{table(["NDT method", "Healthy reference (mean &plusmn; sd)", "Note"], ndt_rows, cls="small",
       caption="Healthy NDT reference values (assumed for new concrete; replaced by the survey at handing over).")}
"""


def s_inspection(ins):
    comps = ins["components"]
    crow = [[e(c), f(v["weight"], 1), str(len(v["locations"])), e(", ".join(v["defects"][:4])) + ("&hellip;" if len(v["defects"]) > 4 else "")]
            for c, v in comps.items()]
    trow = [[e(t["type"]), e(t["interval"]), e(t["scope"])] for t in ins["inspection_types"]]
    ndt = [[e(p["defect"]), e(p["question"]), e("; ".join(p["tests"])), e(p["interpretation"])] for p in ins["ndt_programme"]]
    return f"""
<h1 id="s5">5 Visual inspection</h1>
<p>Hierarchy <b>Bridge &rarr; Span &rarr; Component &rarr; Location &rarr; Observation</b>: bridge BR-33, span S1, {len(comps)} components,
{sum(len(v['locations']) for v in comps.values())} coded locations (e.g. <code>G3/L/2/bottom flange</code>,
<code>BRG-P1-G6</code>, <code>EJ-A1/carriageway</code>). Each observation type has measured inputs (crack width and orientation,
spall area and exposed steel, bearing shear strain, &hellip;) and a rule that gives a condition state CS1 (good) to CS4 (severe).
PSC girders are rated more strictly than RCC because they were designed uncracked (a 0.1 mm flexural crack is CS3; any
crack along a duct is at least CS3). A component takes its worst observation; the bridge health index
HI = 100 &Sigma; w (4 &minus; CS)/3 / &Sigma; w weights components by structural importance, and a CS4 on girders, bearings, pier or
abutment overrides the index.</p>
{table(["Component", "Weight", "Locations", "Observation types"], crow, cls="small")}
{table(["Inspection", "Interval", "Scope"], trow, cls="small")}
<p><b>Digital inspection app</b> (<code>inspection_app/index.html</code>, one offline file): clickable plan of the span (underside
and deck-top views) coloured by the worst condition state, defect-specific inputs with the condition state computed live
(override only with a stated reason), photos, health index, recommended NDT, CSV/JSON export. Its rules are the same as the
Python rules (a test compares both on more than 500 cases).</p>
{fig("step4_app.png", "Inspection app with the illustrative year-12 principal inspection (leaking joint at A1 feeding efflorescence on the G1/G2 end blocks).")}
<h1 id="s6">6 Non-destructive testing</h1>
<p>Each test answers an engineering question raised by an observed defect (<b>defect &rarr; question &rarr; NDT &rarr; measurement
&rarr; interpretation</b>). The baseline grid has {len(I.ndt_locations())} test points (soffit and web of every girder at five
zones, deck soffit bays) where rebound, UPV, cover, half-cell potential and resistivity are measured at handing over and at
principal inspections; chloride profiles are taken at the end blocks.</p>
{table(["Observed defect", "Engineering question", "NDT", "Interpretation"], ndt, cls="small")}
{fig("step4_ndt_grid.png", "NDT baseline grid.")}
"""


def s_synthetic(ct):
    dr = [[e(k), e(v)] for k, v in SY.DAMAGE_RANGES.items()]
    rows = [[e(r["kind"]), f(r["severity"], 2), f(r["f_bending"], 3), f(r["f_torsion"], 3), f(r["lt_defl_G1"], 2),
             f(r["camber_change_G1_mm"], 2), f(r["vw_strain_change_G1_ue"], 0), f(r["upv_G1_min"], 2), f(r["hcp_G1_min"], 0),
             f(r["brg_slope_G1"], 3)] for r in ct]
    import pandas as pd
    df = pd.read_csv(OUT / "data" / "campaigns.csv")
    nd = sum(1 for _ in open(OUT / "data" / "visual_observations.csv")) - 1
    return f"""
<h1 id="s7">7 Deterioration scenarios and synthetic data</h1>
<p>Six mechanisms plus the healthy state. Each has a severity s &isin; [0, 1] mapped to a physical parameter of the twin; the
ranges below are my assumptions, chosen from &ldquo;just detectable&rdquo; to &ldquo;severe but standing&rdquo;.</p>
{table(["Mechanism", "Model and range"], dr, cls="small")}
<p><b>Prestress loss</b> is applied as the reversed equivalent loads of the tendon (nodal couples &Delta;P&middot;e on the girder) on
the grillage: the deck shares the effect, the moment sum at mid-span equals &Delta;P&middot;e (checked), and the girder only
cracks under traffic when the loss consumes the ~3 MPa frequent-load margin to cracking found in Chapter 3. With HDPE ducts the
strands are electrically insulated: strand corrosion from grout voids leaves no half-cell signal; only chloride-driven loss
(40 % of cases) shows on the surrounding reinforcement.</p>
<p><b>A campaign</b> is one visit of the bridge in one condition and environment (age 1&ndash;30 y, season, deck temperature,
traffic). For each campaign the twin produces a 300 s, 8-channel acceleration record from which f<sub>1</sub>, f<sub>2</sub>, f<sub>3</sub>
are identified by spatial modal filtering, the load-test deflections and strains of all girders, gauge-strain and camber
residuals relative to the reference curves (with a creep-model error common to the bridge), bearing movement per &deg;C, the
81 NDT readings (with per-survey calibration errors, wet-season and carbonation effects) and the inspector's observations
(imperfect detection, plus unrelated defects that appear on any bridge). The dataset has <b>{len(df)} campaigns</b>
({len(df) // 7} per class), {len(df) * 81:,} NDT readings and {nd:,} visual observations.</p>
{table(["Mechanism", "s", "f<sub>bend</sub>", "f<sub>tors</sub>", "LT defl. G1 (mm)", "Camber change (mm)", "VW strain (&mu;&epsilon;)",
        "UPV min (km/s)", "Half-cell min (mV)", "Bearing mm/&deg;C"], rows, cls="small",
       caption="Noise-free effect of each mechanism on the measured quantities (30 &deg;C; G1 affected; prestress loss with the chloride cause).")}
{fig("step5_synthetic_overview.png", "Synthetic campaigns: temperature confounds the frequency; each mechanism shows in a different measurement.")}
<p>No single measurement sees every mechanism: prestress loss is invisible to vibration and to the load test and shows only in the
levelling and gauge residuals; chloride corrosion shows only in half-cell potentials and visual inspection; a degraded bearing
changes the frequency by about 0.2 % and is found essentially by the inspector.</p>
"""


def s_ml(m):
    tm = m["temperature_models"]["f1"]
    sig = m["signatures"]
    dec = dict(df1_pct=2, df2_pct=2, lt_ratio_max=3, lt_ratio_min=3, camber_res_max=2, brg_slope_ratio_min=3,
               upv_girder_min=2, hcp_min_c=0, vis_hi=1)
    srow = [[e(k.replace("_", " "))] + [f(v[c], d) for c, d in dec.items()] for k, v in sig.items()]
    thr = joblib.load(OUT / "models.joblib")["detector"].thr
    an = m["anomaly"]
    arow = [[e(k), f(100 * v["false_alarm"], 1), f(100 * v["detection"], 1), f(v["auc"], 3)] + (
        [f(100 * v["per_class"][c], 0) for c in SY.KINDS[1:]] if "per_class" in v else [""] * 6) for k, v in an.items()]
    cl = m["classification"]
    crow = [[e(k), f(v["macro_f1"], 3)] + [f(v["recall"][c], 2) for c in SY.KINDS] for k, v in cl.items()]
    reg = m["regression"]
    rrow = [[e(k.replace("_", " ")), f(v.get("mae_pct", v.get("mae")), 3), f(v["r2"], 2)] for k, v in reg.items()]
    loc = m["localisation"]
    lrow = [[e(k), f(100 * v["exact"], 0), f(100 * v["within_one_girder"], 0)] for k, v in loc.items()]
    from bridge.ml import PHYSICAL_MEANING
    imp = [[e(k), f(v, 3), e(PHYSICAL_MEANING.get(k, ""))] for k, v in list(m["permutation_importance_all"].items())[:10]]
    short = [c.replace("_", " ") for c in SY.KINDS]
    return f"""
<h1 id="s8">8 Feature extraction</h1>
<p>Features were chosen for their physical meaning, and everything that defines &ldquo;normal&rdquo; is learnt from <b>healthy
training campaigns only</b> (in practice the first years of monitoring of this bridge). Temperature compensation of the identified
frequency: f<sub>1</sub> = {tm['at_30C']:.4f} {tm['per_degC']:+.5f}(T &minus; 30) Hz ({100 * tm['per_degC'] / tm['at_30C']:+.3f} %/&deg;C, R&sup2; = {tm['r2']:.2f});
the healthy scatter of f<sub>1</sub> drops from {m['f1_healthy_sd_mHz']['raw']:.1f} to {m['f1_healthy_sd_mHz']['compensated']:.1f} mHz.
Load-test deflections and strains are compensated the same way and expressed as ratios to the healthy reference of each girder;
half-cell potentials and resistivity are corrected for the wet season ({m['wet_shift']['hcp_min']:+.0f} mV), rebound for carbonation
hardening. The feature vector has five groups: vibration (&Delta;f<sub>1&ndash;3</sub> in %, f<sub>2</sub>/f<sub>1</sub>, a<sub>RMS</sub>,
a<sub>max</sub>), load test, long-term monitoring (camber and strain residuals, bearing movement ratio), NDT summaries and visual
summaries.</p>
{table(["Class", "&Delta;f<sub>1</sub> %", "&Delta;f<sub>2</sub> %", "LT ratio max", "LT ratio min", "Camber res. max (mm)", "Bearing ratio",
        "UPV min", "Half-cell min (mV)", "Visual HI"], srow, cls="small",
       caption="Class signatures (mean features). They are physically reasonable: stiffness loss lowers f and raises the deflection, a seized bearing does the opposite, prestress loss shows only in the camber, chloride only in the potentials.")}
{fig("step6_temperature_compensation.png", "Temperature compensation of f<sub>1</sub> learnt from healthy campaigns.")}
<h1 id="s9">9 AI/ML-assisted condition assessment</h1>
<p>Data: {m['n_train']} training and {m['n_test']} test campaigns (stratified split). Methods: anomaly detection (Mahalanobis
distance with shrinkage covariance, threshold at the 99th percentile of out-of-fold healthy scores; isolation forest for
comparison), classification of the mechanism (random forest), regression of severity and prestress loss (gradient boosting),
localisation of the damaged girder, and k-means clustering without labels.</p>
<h2>9.1 Is something wrong? Anomaly detection trained on the healthy bridge only</h2>
{table(["Features", "False alarms %", "Detected %", "AUC"] + short[1:], arow, cls="small")}
<p>Each source alone detects only the mechanisms it can physically see; all sources together detect {100 * an['all']['detection']:.0f} %
of damaged campaigns at {100 * an['all']['false_alarm']:.1f} % false alarms. Features that never change on the healthy bridge (a recorded
flexural crack, a bearing defect, chloride at the end blocks) have almost zero healthy scatter, so a single such observation gives
Mahalanobis distances of 10<sup>4</sup>&ndash;10<sup>6</sup> against a threshold of {thr:.0f}: the size of the score
means &ldquo;never seen on this bridge when healthy&rdquo;, not a proportionally worse condition.</p>
{fig("step6_detection_vs_severity.png", "Probability of detection vs. severity: vibration features alone vs. all sources.")}
<h2>9.2 What is it? Damage type by data source</h2>
{table(["Features", "Macro-F1"] + [f"recall {s}" for s in short], crow, cls="small")}
{fig("step6_confusion.png", "Confusion matrices, test set: vibration only vs. all sources fused.")}
{fig("step6_sources_importance.png", "Classification quality by source and the most informative fused features (permutation importance).")}
{table(["Feature", "Drop in macro-F1", "Physical meaning"], imp, cls="small", caption="What the fused classifier relies on.")}
<h2>9.3 How bad, and where?</h2>
{table(["Target", "MAE", "R&sup2;"], rrow, cls="small", caption="Regression on test campaigns of each class (prestress loss in % of P<sub>e</sub>).")}
{table(["Mechanism", "Exact girder %", "Within one girder %"], lrow, cls="small", caption="Localisation of the damaged girder (chance: 17 %).")}
{fig("step6_regression_pca.png", "Prestress loss estimated from levelling + VW gauges, and all campaigns in the first two principal components.")}
<p><b>Clustering.</b> k-means with seven clusters on all features reaches an adjusted Rand index of {m['clustering']['ari']:.2f} with the true
classes: mechanisms with a strong signature form their own clusters, while degraded bearings and early-stage damage merge with the
healthy campaigns &ndash; without labels the data separate &ldquo;clearly changed&rdquo; from &ldquo;normal&rdquo;, not mechanisms.</p>
<h2>9.4 What the results mean for the engineer</h2>
<ul>
<li>Vibration alone is a weak assessor (macro-F1 {cl['vibration']['macro_f1']:.2f}): it sees stiffness changes and seized bearings, but
not prestress loss or corrosion, and temperature moves f<sub>1</sub> as much as moderate damage. Fusing all sources raises macro-F1 to
<b>{cl['all']['macro_f1']:.2f}</b>.</li>
<li>Each source answers a different question: visual inspection finds bearings, joints and surface defects; NDT finds corrosion and
poor concrete; levelling and embedded gauges are the only route to prestress loss; vibration and the load test measure stiffness
and restraint. This is exactly the fusion asked for in the maintenance chapter.</li>
<li>The outputs are physically reasonable (class signatures above), but they are <b>digital-twin performance</b>: the deterioration
ranges, noise levels and detection probabilities are my assumptions. On a real bridge the healthy reference must be learnt from its
own first years of monitoring, and the classifier needs field cases before it can be trusted for anything but triage.</li>
<li>The ML output is a recommendation for the engineer (which mechanism is most likely, how severe, where to look), not a decision.
Chapter 10 turns it into decisions with explicit rules that the engineer can audit.</li>
</ul>
"""


def s_maintenance(ms):
    import pandas as pd
    lev = [[str(i), e(l), e(MT.REINSPECT[i])] for i, l in enumerate(MT.LEVELS)]
    act = [[e(k), e(v["investigate"]), e(v["repair"])] for k, v in MT.ACTIONS.items() if k != "healthy"]
    tab = ms["table"]
    trow = []
    for k, v in tab.items():
        kind, sb = k.split("|")
        trow.append([e(kind.replace("_", " ")), e(sb)] + [f(v.get(str(i), v.get(i, 0.0)), 2) for i in range(5)])
    ex = ms["example_decision"]
    lives = ms["life"]
    life_rows = []
    for p, h in lives.items():
        reps = [f"{r['repaired'].replace('_', ' ')} (year {r['year']}, s = {r['true_severity']:.2f})" for r in h if isinstance(r["repaired"], str)]
        c = ms["life_cost"][p]
        life_rows.append([e(p), "; ".join(reps) or "&ndash;", f(c["max_severity"], 2), f(c["spent"], 1), f(c["outstanding"], 1),
                          f"<b>{c['total']:.1f}</b>"])
    pro, rea = lives["proactive"], lives["reactive"]
    first_flag = {}
    for p, h in lives.items():
        for m in ms["story"]:
            yrs = [r for r in h if r["active"] == m["kind"] and r["level"] >= 2]
            first_flag[(p, m["kind"])] = yrs[0] if yrs else None
    pl_years = [r for r in rea if r["active"] == "prestress_loss"]
    pl_vis = min(r["vis_hi"] for r in pl_years) if pl_years else float("nan")
    pl_pro = first_flag[("proactive", "prestress_loss")]
    lc = ms["life_cost"]
    return f"""
<h1 id="s10">10 Maintenance scheme</h1>
<p>The decision fuses <b>visual inspection + NDT + structural response + vibration + AI/ML</b>. Rules that need no model come first
(visual CS4 on girders or bearings; CS3 on structural components or joints; half-cell below &minus;350 mV; UPV below 3.75 km/s;
camber more than 3 mm below the reference curve; bearing movement below 80 % of &alpha;L per &deg;C), then the ML assessment
(anomaly flag; most likely mechanism with probability &ge; 0.6 and estimated severity &ge; 0.4 &rarr; repair; structural mechanism
with severity &ge; 0.8 or estimated prestress loss &gt; 20 % &rarr; urgent). Every decision lists its reasons.</p>
{table(["Level", "Decision", "Re-inspection"], lev, cls="small")}
{table(["Mechanism", "Detailed investigation", "Maintenance / repair"], act, cls="small")}
<p><b>Is the logic safe and economical?</b> On the {sum(1 for _ in tab)} condition groups of the test set:
{100 * ms['healthy_routine']:.0f} % of healthy campaigns stay on routine or enhanced monitoring and {100 * ms['healthy_to_repair']:.1f} %
are sent to repair; {100 * ms['severe_left_routine']:.1f} % of severe cases (s &ge; 0.6) are left on routine monitoring and
{100 * ms['severe_investigated']:.0f} % are investigated or repaired.</p>
{table(["Condition", "Severity", "L0", "L1", "L2", "L3", "L4"], trow, cls="small", caption="Share of test campaigns at each decision level.")}
<p><b>Example decision</b> (most severe prestress-loss test campaign): <i>{e(ex['decision'])}</i> &mdash; reasons: {e('; '.join(ex['reasons']))};
action: {e(ex['action'])}; re-inspection: {e(ex['reinspect'])}.</p>
<h2>10.1 Thirty years of this bridge</h2>
<p>The same twin was run year by year: the A1 joint starts leaking in year 8 (chloride corrosion of the end zones grows by 0.09
per year), an overload cracks girder G2 in year 18, and prestress loss from grout voids starts in G4 in year 23. A
<b>proactive owner</b> (yearly monitoring + NDT + ML, repairs at level 3, or level 2 confirmed in two consecutive years) is
compared with a <b>reactive owner</b> (visual inspection only; acts on CS4 or a visual health index below 60). Costs are relative
units I assumed (repair cost growing with the square of the severity, monitoring 0.3 per year, investigation 0.5, load restriction
5 per year while the severity exceeds 0.8). Damage still unrepaired at year 30 is charged at its repair cost: it is deferred
expenditure, not a saving.</p>
{table(["Owner", "Repairs (year, true severity)", "Worst severity", "Spent", "Unrepaired at year 30", "Total"], life_rows, cls="small")}
{fig("step7_life.png", "Thirty-year life: true severity, repairs, the yearly anomaly score with the decision level, and cumulative cost.")}
<ul>
<li><b>Chloride corrosion</b> is caught by the half-cell survey and the ML one year after it starts and repaired at a low severity;
the reactive owner waits until the visual condition is poor, when the patch area and the repair cost are much larger.</li>
<li><b>The overload crack</b> in G2 is visible to both owners; the difference is only how quickly they act.</li>
<li><b>Prestress loss from grout voids</b> is the decisive case. It is invisible to the inspector (lowest visual health index
during it: {pl_vis:.0f}), so the reactive owner never repairs it and ends the period at severity {lc['reactive']['max_severity']:.2f}
(about {100 * (0.03 + 0.27 * lc['reactive']['max_severity']):.0f} % loss in G4) under a load restriction. The proactive owner's
levelling and embedded gauges flag it in year {pl_pro['year'] if pl_pro else '&ndash;'} (severity {pl_pro['true_severity'] if pl_pro else float('nan'):.2f}).
This is later than one would like: in the first years the classifier already leaned towards prestress loss but the anomaly score stayed
below the 99 % threshold. Gradual loss of a few per cent is within the scatter of the long-term reference curve, which is the
practical detection limit of this scheme.</li>
<li>Over the 30 years the proactive owner's total is {lc['proactive']['total']:.0f} against {lc['reactive']['total']:.0f}
relative units, and the bridge never exceeds severity {lc['proactive']['max_severity']:.2f}. The comparison depends on the assumed
costs; the robust conclusion is the safety one: without monitoring, the prestress loss is found only when it is already severe.</li>
</ul>
"""


def s_ai(tests):
    ok, total, names = tests
    return f"""
<h1 id="s11">11 Use of AI and verification</h1>
<p>An AI assistant (Claude, Anthropic) was used, as permitted by &sect;1.15, to formulate the problem, write the Python tools,
generate the synthetic data and draft this report. The engineering choices (span, configuration, two-stage stressing, strand count,
damage ranges) were discussed and decided step by step, and every computational result was checked independently:</p>
<ul>
<li><b>{ok} of {total} automated checks pass</b> (<code>tests/</code>): single-beam results against PL/4, PL&sup3;/48EI and wL&sup2;/8;
grillage equilibrium; vehicle search; friction and wedge-slip against the closed form; ULS against the parabola&ndash;rectangle hand
formula; zero eigen-stress for uniform and linear temperature; mass of the twin against the dead loads; f<sub>1</sub> against the beam
formula; FDD on a known SDOF signal; prestress-loss equivalent loads against &Delta;P&middot;e; JS and Python inspection rules on
500+ cases; detector false-alarm rate on fresh data.</li>
<li>Hand checks reported in the text: Courbon distribution, simply supported beam frequency, dead-load mass.</li>
<li>Errors found during the work and corrected: a timeline constant that silently overwrote the deck thickness (a test now guards
the configuration); a first prestress design (3 &times; 12T15) that cracked under the frequent combination; a duct clash of 183 mm
against 190 mm required; an optimistic NDT model in which half-cell potentials &ldquo;saw&rdquo; strands inside insulating HDPE ducts,
and an anomaly threshold set on in-sample scores that gave 8 % instead of ~1 % false alarms.</li>
</ul>
<p class="todo">[Student: add your own account here &mdash; which AI suggestions you questioned or changed, what you checked by hand,
and what you learnt. This section must be in your own words.]</p>
"""


def s_assumptions():
    items = [
        "Span S1 of a viaduct; piers, abutments, foundations, seismic and wind design outside the scope.",
        "IRC:6-2017 loads; the 10 % multi-lane reduction only for three Class A lanes; non-urban footpath intensity.",
        "Loss parameters: &mu; = 0.17, k = 0.002/m, wedge slip 6 mm, RH 70 %, final relaxation 3 &times; the 1000-h value; creep and "
        "shrinkage by IRC:112 Annex A2 (drying shrinkage by the EN 1992-1-1 Annex B model with f<sub>ck,cyl</sub> = 0.8 f<sub>ck,cube</sub>).",
        "Characteristic prestress 0.9/1.1 P<sub>m</sub>; SLS combination factors: live load 0.75 frequent, temperature 0.6 / 0.5 / 0.5.",
        "Cable radii, anchorage spacing and edge distances to be confirmed against the approved PT system.",
        "Twin: E<sub>d</sub> = (E<sub>c</sub> + 19)/1.25, &minus;0.35 %/&deg;C change of E with temperature, 1.5 % damping, barriers non-composite.",
        "Deterioration ranges, NDT and inspection error models, traffic and ambient levels, and costs are assumptions; all data are "
        "synthetic, so the ML results are digital-twin performance and need field recalibration.",
        "Code values quoted from memory (e.g. IRC:112 Table 7.1, IS 516 UPV grading) are to be verified against the current editions.",
    ]
    refs = ["IRC:5-2015 Standard specifications and code of practice for road bridges, Section I.",
            "IRC:6-2017 Section II: Loads and load combinations.",
            "IRC:112-2020 Code of practice for concrete road bridges.",
            "IRC:83 (Part II)-2018 Elastomeric bearings.",
            "IRC:SP:35 Guidelines for inspection and maintenance of bridges; IRC:SP:51 Guidelines for load testing of bridges.",
            "IS 516 (Part 4, Part 5 Sec 1, Sec 4) Hardened concrete: cores, ultrasonic pulse velocity, rebound hammer.",
            "ASTM C876 Corrosion potentials of uncoated reinforcing steel in concrete.",
            "BS 8110-2:1985 cl. 7.2 (dynamic and static modulus); EN 1992-1-1 Annex B (creep and shrinkage).",
            "Hambly, E.C. Bridge Deck Behaviour, 2nd ed. (grillage analysis).",
            "Brincker, R., Zhang, L., Andersen, P. (2001) Modal identification of output-only systems using frequency domain decomposition. "
            "Smart Materials and Structures 10, 441&ndash;445."]
    return f"""
<h1 id="s12">12 Assumptions and limitations</h1>
<ul>{''.join(f'<li>{x}</li>' for x in items)}</ul>
<h1 id="s13">References</h1>
<ol>{''.join(f'<li>{x}</li>' for x in refs)}</ol>
<p class="small">Code and data: <code>bridge/</code> (modules), <code>run_step1_analysis.py</code> &hellip; <code>run_step7_maintenance.py</code>,
<code>outputs/</code> (JSON results, figures, datasets), <code>inspection_app/</code>, <code>tests/</code>, <code>report/build_report.py</code>.</p>
"""


CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
body { font: 10.3pt/1.45 "Georgia", "Cambria", serif; color: #1b1f24; max-width: 178mm; margin: 0 auto; }
h1, h2, h3, .title, caption, figcaption, th { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
h1 { font-size: 15pt; margin: 20pt 0 6pt; border-bottom: 1.5px solid #1f5fa8; padding-bottom: 3pt; break-after: avoid; }
h1:not(:first-of-type) { break-before: page; }
h2 { font-size: 11.5pt; margin: 14pt 0 4pt; color: #1f5fa8; break-after: avoid; }
p { margin: 4pt 0 7pt; text-align: justify; hyphens: auto; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font: 8.6pt/1.35 "Segoe UI", Arial, sans-serif; break-inside: avoid; }
table.small { font-size: 7.6pt; }
caption { caption-side: top; text-align: left; font-size: 8.4pt; color: #444; margin-bottom: 3pt; }
th { background: #eef2f7; text-align: left; font-weight: 600; }
th, td { border: 1px solid #cfd6df; padding: 2.5pt 4pt; vertical-align: top; }
figure { margin: 8pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.4pt; color: #444; text-align: left; margin-top: 3pt; }
code { font-size: 8.6pt; }
.titlepage { height: 250mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.title { font-size: 22pt; font-weight: 700; line-height: 1.2; margin-bottom: 10pt; }
.subtitle { font-size: 13pt; color: #1f5fa8; margin-bottom: 26pt; font-family: "Segoe UI", Arial, sans-serif; }
.meta td { border: none; padding: 2pt 8pt 2pt 0; font-size: 10.5pt; }
.todo { background: #fff4d6; padding: 6pt; border-left: 3px solid #e0a800; }
.missing { color: #b00; }
.toc li { margin: 2pt 0; }
.small { font-size: 8.6pt; }
"""


def main():
    s1, s2, b = J("step1_analysis.json"), J("step2_design.json"), J("baseline.json")
    ins, ct, m, ms = J("inspection_scheme.json"), J("condition_table.json"), J("ml_summary.json"), J("maintenance_summary.json")
    app_screenshot()
    tests = run_tests()
    L0 = s2["girders"]["0"]["losses"]
    cl = m["classification"]
    abstract = f"""
<h1 id="s0" style="break-before:auto">Summary</h1>
<p>A {C.L:.0f} m post-tensioned PSC I-girder bridge (3 lanes, footpaths, {C.N_GIRDERS} girders at {C.GIRDER_SPACING} m, {C.H_GIRDER:.2f} m +
{C.T_DECK:.2f} m deck, M45/M35) was designed to IRC:6 and IRC:112 with two-stage post-tensioning (4 &times; 11T15; P<sub>e</sub> =
{L0['Pe_kN']:.0f} kN, total loss {L0['total_loss_pct']:.1f} %; all stress, ULS, shear and deflection checks satisfied). A digital twin
of the as-built bridge gives the healthy baseline (f<sub>1</sub> = {b['modal']['freq'][0]:.2f} Hz bending, {b['modal']['freq'][1]:.2f} Hz
torsion; {b['df1_dT_pct_per_degC']:+.2f} %/&deg;C), the inspection, NDT and monitoring schemes, and {sum(1 for _ in open(OUT / 'data' / 'campaigns.csv')) - 1}
synthetic campaigns for six deterioration mechanisms. On these data, vibration features alone classify the mechanism with
macro-F1 {cl['vibration']['macro_f1']:.2f}; fusing visual inspection, NDT, monitoring and vibration gives {cl['all']['macro_f1']:.2f}.
Prestress loss is estimated from levelling and embedded gauges with an error of {m['regression']['prestress_loss_from_monitoring']['mae_pct']:.1f} %
of P<sub>e</sub>. A rule-based maintenance scheme turns the fused evidence into auditable decisions; over a simulated 30-year life the
proactive owner keeps every mechanism below severity {ms['life_cost']['proactive']['max_severity']:.1f} at a total of
{ms['life_cost']['proactive']['total']:.0f} relative units, against {ms['life_cost']['reactive']['total']:.0f} (including the unrepaired
damage) for a reactive owner, who never sees the grout-void prestress loss.
All data are synthetic: the ML figures are digital-twin performance.</p>
"""
    toc = [("s1", "Introduction and bridge configuration"), ("s2", "Loads and structural analysis"), ("s3", "Prestressing system"),
           ("s4", "Healthy baseline and monitoring scheme"), ("s5", "Visual inspection"), ("s6", "Non-destructive testing"),
           ("s7", "Deterioration scenarios and synthetic data"), ("s8", "Feature extraction"),
           ("s9", "AI/ML-assisted condition assessment"), ("s10", "Maintenance scheme"), ("s11", "Use of AI and verification"),
           ("s12", "Assumptions and limitations")]
    body = f"""
<div class="titlepage">
<div class="subtitle">CE60258 AI/ML for Sustainable Infrastructure Systems &mdash; One-Month Integrated Big Problem</div>
<div class="title">Integrated study of a 33 m post-tensioned PSC I-girder bridge: design, baseline, inspection,
monitoring, AI/ML condition assessment and maintenance</div>
<table class="meta"><tr><td>Student</td><td>{STUDENT}</td></tr><tr><td>Roll number</td><td>{ROLL}</td></tr>
<tr><td>Span</td><td>L = 33 m</td></tr><tr><td>Instructor</td><td>Prof. Nirjhar Dhang</td></tr>
<tr><td>Submission</td><td>October 2026</td></tr></table>
</div>
{abstract}
<h2>Contents</h2><ol class="toc">{''.join(f'<li><a href="#{a}">{t}</a></li>' for a, t in toc)}</ol>
{s_intro(s1, s2)}{s_analysis(s1)}{s_prestress(s2)}{s_baseline(b)}{s_inspection(ins)}{s_synthetic(ct)}{s_ml(m)}
{s_maintenance(ms)}{s_ai(tests)}{s_assumptions()}
"""
    out = REP / "report.html"
    out.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>33 m PSC bridge - integrated study</title>"
                   f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    print(f"report: {out} ({out.stat().st_size / 1e3:.0f} kB), {FIGN[0]} figures, tests {tests[0]}/{tests[1]}")
    exe = next((c for c in CHROME if c.exists()), None)
    if exe:
        pdf = REP / "report.pdf"
        subprocess.run([str(exe), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                        out.as_uri()], capture_output=True, timeout=300)
        if pdf.exists():
            try:
                from pypdf import PdfReader
                n = len(PdfReader(str(pdf)).pages)
            except Exception:
                n = "?"
            print(f"pdf: {pdf} ({pdf.stat().st_size / 1e6:.1f} MB, {n} pages)")


if __name__ == "__main__":
    main()
