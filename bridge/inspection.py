"""Visual-inspection scheme and NDT programme.

Hierarchy:  Bridge -> Span -> Component -> Location -> Observation.
Condition states CS1 (good) .. CS4 (severe) per observation; a component takes the worst state of its
observations; the bridge health index HI (0-100) weights components by structural importance.
Everything here is plain data so that the web app (inspection_app/) is generated from it.
"""
from . import config as C

BRIDGE = dict(id="BR-33", name="33 m PSC I-girder viaduct (study bridge)", span="S1",
              span_ends=["A1 (abutment)", "P1 (pier)"], span_m=C.L, deck_width_m=C.B_DECK,
              girders=C.N_GIRDERS, built="2026 (assumed)", exposure="severe")

ZONES = ["End-A1", "L/4", "L/2", "3L/4", "End-P1"]
ZONE_X = {"End-A1": 0.0, "L/4": C.L / 4, "L/2": C.L / 2, "3L/4": 3 * C.L / 4, "End-P1": C.L}
BAYS = [f"B{k}" for k in range(1, C.N_GIRDERS)] + ["Cantilever-L", "Cantilever-R"]


def components():
    """Component -> dict(material, weight, locations, defects)."""
    g_loc = [f"G{g}/{z}/{part}" for g in range(1, C.N_GIRDERS + 1) for z in ZONES
             for part in (["bottom flange", "web", "top flange"] + (["anchorage"] if z.startswith("End") else []))]
    out = {
        "PSC girders": dict(material="PSC", weight=1.0, locations=g_loc,
                            defects=["crack", "spalling", "corrosion staining", "leakage / efflorescence",
                                     "honeycombing", "surface deterioration", "exposed steel"]),
        "Deck slab": dict(material="RCC", weight=0.8,
                          locations=[f"{b}/{z}/{s}" for b in BAYS for z in ZONES for s in ("soffit", "top")],
                          defects=["crack", "spalling", "corrosion staining", "leakage / efflorescence",
                                   "honeycombing", "surface deterioration", "exposed steel"]),
        "Cross girders": dict(material="RCC", weight=0.6,
                              locations=[f"CG-{p}/{b}" for p in ("A1", "Mid", "P1") for b in BAYS[:5]],
                              defects=["crack", "spalling", "corrosion staining", "leakage / efflorescence",
                                       "honeycombing", "exposed steel"]),
        "Bearings": dict(material="elastomer", weight=0.7,
                         locations=[f"BRG-{e}-G{g}" for e in ("A1", "P1") for g in range(1, C.N_GIRDERS + 1)],
                         defects=["bearing shear deformation", "bearing bulging / splitting", "bearing displacement / walking",
                                  "bearing seized / debris", "pedestal cracking"]),
        "Expansion joints": dict(material="strip seal", weight=0.5,
                                 locations=[f"EJ-{e}/{s}" for e in ("A1", "P1") for s in ("footpath L", "carriageway", "footpath R")],
                                 defects=["joint debris", "joint seal torn / leaking", "joint anchorage damage"]),
        "Drainage": dict(material="spouts / pipes", weight=0.3,
                         locations=[f"D{k}-{side}" for side in ("L", "R") for k in range(1, 5)],
                         defects=["drain blocked", "drain damaged / missing", "discharge onto structure"]),
        "Crash barriers": dict(material="RCC", weight=0.4,
                               locations=[f"Barrier-{s}/{z}" for s in ("L", "R") for z in ZONES],
                               defects=["crack", "spalling", "impact damage", "exposed steel"]),
        "Footpaths": dict(material="RCC / paving", weight=0.2,
                          locations=[f"Footpath-{s}/{z}" for s in ("L", "R") for z in ZONES],
                          defects=["surface deterioration", "crack", "spalling"]),
        "Railings": dict(material="steel / RCC", weight=0.2,
                         locations=[f"Railing-{s}/{z}" for s in ("L", "R") for z in ZONES],
                         defects=["impact damage", "corrosion staining", "missing members"]),
        "Wearing coat": dict(material="bituminous", weight=0.3,
                             locations=[f"Lane{l}/{z}" for l in (1, 2, 3) for z in ZONES],
                             defects=["wearing coat cracking", "potholes / ravelling", "rutting"]),
        "Abutment A1": dict(material="RCC", weight=0.8,
                            locations=["bearing shelf", "dirt wall", "wing wall L", "wing wall R", "stem"],
                            defects=["crack", "spalling", "leakage / efflorescence", "corrosion staining", "exposed steel"]),
        "Pier P1": dict(material="RCC", weight=0.9,
                        locations=["pier cap top", "pier cap soffit", "pier cap cantilever L", "pier cap cantilever R",
                                   "shaft"],
                        defects=["crack", "spalling", "leakage / efflorescence", "corrosion staining", "exposed steel"]),
    }
    return out


# ---------------------------------------------------------------------- defects and their inputs
# input kinds: number (with unit), select (options), bool
DEFECTS = {
    "crack": dict(inputs={"width_mm": "number", "length_m": "number",
                          "orientation": ["map / random", "flexural (transverse, soffit)", "shear (diagonal, web)",
                                          "longitudinal along duct", "bursting (anchorage zone)", "other"],
                          "rust stain": "bool"}),
    "spalling": dict(inputs={"area_m2": "number", "depth_mm": "number", "steel exposed": ["none", "rebar", "duct / strand"]}),
    "corrosion staining": dict(inputs={"area_m2": "number", "severity": ["light", "moderate", "heavy / section loss"]}),
    "exposed steel": dict(inputs={"length_m": "number", "steel": ["rebar", "duct / strand"], "section loss": "bool"}),
    "leakage / efflorescence": dict(inputs={"area_m2": "number", "state": ["damp patch", "active leakage", "efflorescence",
                                                                          "leakage with rust stain"]}),
    "honeycombing": dict(inputs={"area_m2": "number", "depth_mm": "number", "duct visible": "bool"}),
    "surface deterioration": dict(inputs={"area_m2": "number", "depth_mm": "number"}),
    "impact damage": dict(inputs={"length_m": "number", "severity": ["scrape", "spalled", "displaced / broken"]}),
    "missing members": dict(inputs={"count": "number"}),
    "bearing shear deformation": dict(inputs={"shear_mm": "number", "elastomer_mm": "number"}),
    "bearing bulging / splitting": dict(inputs={"state": ["uniform bulging", "uneven bulging", "surface cracks",
                                                          "splits / exposed plates"]}),
    "bearing displacement / walking": dict(inputs={"offset_mm": "number"}),
    "bearing seized / debris": dict(inputs={"state": ["debris around", "restrained / seized"]}),
    "pedestal cracking": dict(inputs={"width_mm": "number"}),
    "joint debris": dict(inputs={"extent": ["partial", "full length"]}),
    "joint seal torn / leaking": dict(inputs={"length_m": "number"}),
    "joint anchorage damage": dict(inputs={"length_m": "number"}),
    "drain blocked": dict(inputs={"state": ["partly", "fully"]}),
    "drain damaged / missing": dict(inputs={}),
    "discharge onto structure": dict(inputs={}),
    "wearing coat cracking": dict(inputs={"area_m2": "number"}),
    "potholes / ravelling": dict(inputs={"area_m2": "number", "depth_mm": "number"}),
    "rutting": dict(inputs={"depth_mm": "number"}),
}


def _band(v, edges):
    """CS from value v and the upper limits of CS1, CS2, CS3."""
    for cs, e in zip((1, 2, 3), edges):
        if v <= e:
            return cs
    return 4


def condition_state(defect, material, d):
    """Condition state 1..4 for one observation (d: dict of inputs). Rules mirrored in the app."""
    g = lambda k, default=0: d.get(k, default) if d.get(k) not in (None, "") else default
    if defect == "crack":
        w = float(g("width_mm"))
        if material == "PSC":
            cs = _band(w, (0.05, 0.10, 0.30))           # a PSC girder designed uncracked: any open crack matters
            o = g("orientation", "")
            if o.startswith("flexural") or o.startswith("shear"):
                cs = max(cs, 3) if w >= 0.1 else max(cs, 2)
            if o.startswith("longitudinal") or o.startswith("bursting"):
                cs = max(cs, 3)
        else:
            cs = _band(w, (0.10, 0.30, 1.0))
        if g("rust stain", False):
            cs = max(cs, 3)
        return min(cs, 4)
    if defect == "spalling":
        s = g("steel exposed", "none")
        cs = _band(float(g("area_m2")), (0.0, 0.10, 0.50))
        cs = max(cs, 2)
        if s == "rebar":
            cs = max(cs, 3)
        if s.startswith("duct"):
            cs = 4
        return cs
    if defect == "corrosion staining":
        return {"light": 2, "moderate": 3}.get(g("severity", "light"), 4)
    if defect == "exposed steel":
        if g("steel", "rebar").startswith("duct") or g("section loss", False):
            return 4
        return 3
    if defect == "leakage / efflorescence":
        return {"damp patch": 2, "active leakage": 3, "efflorescence": 3}.get(g("state", "damp patch"), 4)
    if defect == "honeycombing":
        if g("duct visible", False):
            return 4
        return _band(float(g("depth_mm")), (0.0, 10.0, 25.0)) if float(g("depth_mm")) > 0 else 2
    if defect == "surface deterioration":
        return max(2, _band(float(g("depth_mm")), (0.0, 5.0, 15.0)))
    if defect == "impact damage":
        return {"scrape": 2, "spalled": 3}.get(g("severity", "scrape"), 4)
    if defect == "missing members":
        return 3 if float(g("count")) <= 2 else 4
    if defect == "bearing shear deformation":
        ratio = float(g("shear_mm")) / max(float(g("elastomer_mm", 48)), 1e-6)
        return _band(ratio, (0.5, 0.7, 1.0))           # shear strain; IRC:83 (Part II) design limit 0.7
    if defect == "bearing bulging / splitting":
        return {"uniform bulging": 1, "uneven bulging": 2, "surface cracks": 3}.get(g("state", ""), 4)
    if defect == "bearing displacement / walking":
        return _band(float(g("offset_mm")), (5.0, 15.0, 30.0))
    if defect == "bearing seized / debris":
        return 2 if g("state", "").startswith("debris") else 4
    if defect == "pedestal cracking":
        return _band(float(g("width_mm")), (0.1, 0.3, 1.0))
    if defect == "joint debris":
        return 2 if g("extent", "partial") == "partial" else 3
    if defect == "joint seal torn / leaking":
        return 3
    if defect == "joint anchorage damage":
        return 4
    if defect == "drain blocked":
        return 2 if g("state", "partly") == "partly" else 3
    if defect in ("drain damaged / missing", "discharge onto structure"):
        return 3
    if defect == "wearing coat cracking":
        return _band(float(g("area_m2")), (0.0, 5.0, 20.0)) if float(g("area_m2")) > 0 else 2
    if defect == "potholes / ravelling":
        return max(2, _band(float(g("depth_mm")), (0.0, 25.0, 50.0)))
    if defect == "rutting":
        return _band(float(g("depth_mm")), (5.0, 15.0, 25.0))
    return 2


CS_TEXT = {1: "CS1 good: no or insignificant defects",
           2: "CS2 fair: minor defects, no effect on capacity or durability yet",
           3: "CS3 poor: defects affecting durability; may affect capacity if not treated",
           4: "CS4 severe: capacity or safety may be affected; act now"}


def health_index(component_cs):
    """HI = 100 * sum w (4 - CS)/3 / sum w over inspected components; component CS = worst observation."""
    comps = components()
    num = den = 0.0
    for c, cs in component_cs.items():
        w = comps[c]["weight"]
        num += w * (4 - cs) / 3
        den += w
    return 100 * num / den if den else 100.0


def action(hi, worst_critical_cs):
    """Rating -> action; a CS4 on girders / bearings / pier overrides the index."""
    if worst_critical_cs >= 4:
        return "special inspection now; restrict loading until assessed"
    if hi >= 85 and worst_critical_cs <= 2:
        return "routine monitoring and next scheduled inspection"
    if hi >= 70:
        return "enhanced monitoring; targeted NDT at CS3 locations"
    if hi >= 50:
        return "detailed investigation (NDT + structural assessment)"
    return "urgent detailed assessment and repair design"


CRITICAL = ("PSC girders", "Bearings", "Pier P1", "Abutment A1")

INSPECTION_TYPES = [
    dict(type="superficial (patrol)", interval="monthly", by="maintenance staff",
         scope="drive-over and walk-along: joints, drains, barriers, wearing coat, visible damage"),
    dict(type="routine", interval="twice a year (pre- and post-monsoon)", by="bridge engineer",
         scope="all components from deck and ground with binoculars; bearings and joints close up; app records"),
    dict(type="principal (detailed)", interval="every 3 years and at handing over", by="experienced bridge engineer",
         scope="hands-on, under-bridge unit; every location of the hierarchy; crack mapping; baseline NDT grid"),
    dict(type="special", interval="after flood, earthquake, vehicle impact, fire, overload, or a monitoring alarm",
         by="specialist", scope="targeted: NDT, load test, structural assessment"),
]

# ---------------------------------------------------------------------- NDT programme
NDT_PROGRAMME = [
    dict(defect="longitudinal crack / rust staining along a duct line (girder bottom flange)",
         question="Are the prestressing strands corroding? Is the duct fully grouted?",
         tests=["half-cell potential grid over the duct line", "concrete resistivity (Wenner)", "cover meter",
                "GPR / ultrasonic tomography over the duct", "chloride profile from drilled dust",
                "if indicated: endoscope through a drilled hole into the duct"],
         interpretation="potential < -350 mV CSE with resistivity < 10 kOhm cm: active corrosion likely; "
                        "chloride at duct depth > 0.2 % of cement: depassivation of strand likely; "
                        "tomography anomaly: grout void -> re-grouting (vacuum injection)",
         locations="along each cracked duct line, 0.25 m grid, +-1 m beyond the visible crack"),
    dict(defect="transverse flexural cracks at the soffit near mid-span",
         question="Has the girder lost prestress (designed uncracked)?",
         tests=["crack width / length mapping and tell-tales or crack gauges", "UPV across the crack (crack depth)",
                "VW strain and levelling trend vs. reference curve", "load test (deflection, strain, crack opening)"],
         interpretation="cracks opening under traffic and a strain / camber trend away from the reference curve "
                        "indicate prestress loss; compare decompression load in the load test with design",
         locations="L/4 .. 3L/4 of the affected girder, all faces of the bottom flange"),
    dict(defect="diagonal web cracks near the supports",
         question="Is the shear capacity or the end-block reinforcement adequate?",
         tests=["crack mapping, width monitoring", "UPV (direct, through the web)", "cover meter for link spacing",
                "load test with trucks near the support"],
         interpretation="cracks > 0.2 mm or growing: detailed shear assessment with as-built links",
         locations="from the bearing to 1.5 d, both faces of the web"),
    dict(defect="honeycombing / surface deterioration / doubtful concrete",
         question="Is the in-situ concrete quality and strength adequate?",
         tests=["rebound hammer (IS 516 Part 5/Sec 4)", "UPV (IS 516 Part 5/Sec 1)",
                "combined rebound + UPV (SonReb) calibrated on cores (IS 516 Part 4)"],
         interpretation="UPV well below the healthy reference (e.g. < 3.75 km/s, 'doubtful' in IS 516-5-1) or rebound "
                        "below reference - 2 sd: take cores; estimate strength from the calibrated curve",
         locations="grid of test areas around the defect plus reference areas on sound concrete of the same element"),
    dict(defect="spalling / delamination / exposed rebar on the deck soffit or cross girders",
         question="How far does corrosion-induced delamination extend? What is causing it?",
         tests=["hammer sounding", "half-cell potential", "cover meter", "carbonation depth (phenolphthalein)",
                "chloride profile"],
         interpretation="carbonation depth >= cover or chloride > 0.4 % of cement at rebar: corrosion initiated; "
                        "repair extent = hollow-sounding area + 0.3 m",
         locations="the bay concerned on a 0.5 m grid, plus one sound bay as reference"),
    dict(defect="leakage / efflorescence at girder ends below an expansion joint",
         question="Is chloride-laden water reaching the end blocks, anchorages and bearings?",
         tests=["chloride profile at anchorage zones", "half-cell potential on end blocks", "resistivity",
                "visual check of anchorage protection (open the pocket if necessary)"],
         interpretation="chloride rising with depth and potentials more negative than -350 mV: anchorage at risk; "
                        "replace the joint seal first, then repair",
         locations="end blocks of all girders under the leaking joint (0.25 m grid over 1.5 m)"),
    dict(defect="bursting / splitting cracks in the anchorage zone",
         question="Is the anchorage-zone reinforcement working? Is the anchorage itself sound?",
         tests=["crack mapping", "UPV", "GPR for bursting steel", "impact-echo on the anchorage pocket"],
         interpretation="cracks > 0.2 mm or voids behind the plate: specialist assessment of the anchorage",
         locations="end face and 1 m of the end block"),
    dict(defect="bearing shear deformation, bulging, splitting, walking or seizure",
         question="Is the bearing accommodating movement and rotation as designed?",
         tests=["measure shear deformation and bulging with a gauge", "bearing-displacement transducer: movement vs. temperature",
                "Shore A hardness of the elastomer", "level survey of the bearing seat"],
         interpretation="shear strain > 0.7, splits or exposed plates, or a movement / temperature slope well below "
                        f"alpha L = {C.ALPHA_T * C.L * 1e3:.3f} mm/degC: replace the bearing (jacking under the end cross girder)",
         locations="all 12 bearings"),
]


def ndt_locations():
    """Standard NDT test points used for the baseline survey and the synthetic data."""
    pts = []
    for g in range(1, C.N_GIRDERS + 1):
        for z in ZONES:
            for face in ("soffit", "web"):
                pts.append(dict(id=f"G{g}-{z}-{face}", component="PSC girders", girder=g, zone=z, x=ZONE_X[z],
                                face=face, material="girder (M45)"))
    for b in BAYS:
        for z in ("L/4", "L/2", "3L/4"):
            pts.append(dict(id=f"D-{b}-{z}", component="Deck slab", girder=None, zone=z, x=ZONE_X[z], face="soffit",
                            material="deck (M35)", bay=b))
    return pts


def scheme():
    """Everything the app needs, as one JSON-serialisable dict."""
    comps = components()
    return dict(bridge=BRIDGE, components=comps, defects=DEFECTS, zones=ZONES, cs_text=CS_TEXT,
                critical=list(CRITICAL), inspection_types=INSPECTION_TYPES, ndt_programme=NDT_PROGRAMME,
                girder_y=C.GIRDER_Y, deck_width=C.B_DECK, span=C.L)
