"""Structural-monitoring scheme and healthy NDT reference values of the 33 m bridge."""
from . import config as C
from .vibration import FS, sensor_layout

L = C.L


def monitoring_scheme():
    """Sensors: what is measured, where, how often, for how long and why."""
    rows = []
    for s in sensor_layout():
        rows.append(dict(id=s.name, type="accelerometer (force-balance / low-noise MEMS, +-2 g, < 1 micro-g/rtHz)",
                         quantity="vertical acceleration", girder=s.girder + 1, x=round(s.x, 2), level="deck soffit",
                         sampling=f"{FS:.0f} Hz", duration="10-min records, every hour (continuous DAQ)",
                         purpose="natural frequencies (bending, torsion), mode shapes, a_rms / a_max; "
                                 "outer girders at L/4, L/2, 3L/4 separate bending from torsion"))
    for g in (1, 3, 6):
        for x, tag in ((L / 2, "M"), (L / 4, "Q")):
            for lvl, y in (("bottom flange", 0.20), ("top flange", 1.70)):
                rows.append(dict(id=f"VW-G{g}-{tag}-{'B' if y < 1 else 'T'}", type="embedded vibrating-wire strain gauge "
                                 "with thermistor", quantity="concrete strain + temperature", girder=g, x=round(x, 2),
                                 level=f"{lvl} (y = {y} m)", sampling="1 per 15 min",
                                 duration="from casting, whole life",
                                 purpose="creep/shrinkage vs. reference curve -> prestress loss; strain change "
                                         "under load test; curvature from bottom/top pair"))
    for g in (1, 3, 4, 6):
        rows.append(dict(id=f"HL-G{g}-M", type="hydrostatic levelling cell (reference at the abutment)",
                         quantity="vertical displacement", girder=g, x=round(L / 2, 2), level="soffit",
                         sampling="1 per 15 min", duration="whole life",
                         purpose="long-term camber / sag vs. reference curve; differential movement between girders"))
    for g in (1, 6):
        rows.append(dict(id=f"BD-G{g}", type="LVDT / potentiometric displacement transducer", quantity=
                         "longitudinal bearing movement", girder=g, x=round(L, 2), level="free-end bearing",
                         sampling="1 per 10 min", duration="whole life",
                         purpose="movement vs. temperature (expected alpha*L per degC): a falling slope = bearing "
                                 "stiffening / seizure"))
    for lvl, y in (("deck top", 2.02), ("girder web", 1.0), ("girder soffit", 0.05)):
        rows.append(dict(id=f"T-{lvl.replace(' ', '_')}", type="thermocouple / RTD", quantity="temperature",
                         girder=3, x=round(L / 2, 2), level=f"{lvl} (y = {y} m)", sampling="1 per 10 min",
                         duration="whole life",
                         purpose="effective temperature and gradient for compensation of frequencies, strain, movement"))
    rows.append(dict(id="WS", type="weather station", quantity="air temperature, RH, wind, rain", girder=None, x=None,
                     level="deck edge", sampling="1 per 10 min", duration="whole life",
                     purpose="environmental covariates for the ML models"))
    rows.append(dict(id="LT", type="load test: dial gauges / LVDTs on a reference frame + 2 test trucks",
                     quantity="deflection, strain", girder="all", x="L/4, L/2, 3L/4", level="soffit",
                     sampling="static readings; 100 Hz for dynamic passes", duration="at handing over, then every 10 y",
                     purpose="as-built stiffness and transverse distribution (IRC:SP:51)"))
    return rows


# Healthy reference values for the NDT programme (mean, standard deviation between test points).
# They are assumptions for new M45 / M35 concrete with the specified covers; the baseline survey at
# handing-over replaces them with measured values.
NDT_BASELINE = {
    "rebound_number": {"girder (M45)": (41.0, 2.5), "deck (M35)": (37.0, 2.5),
                       "_note": "IS 516 (Part 5/Sec 4):2020; horizontal impact on the web, vertical upward on the soffit"},
    "upv_km_s": {"girder (M45)": (4.60, 0.12), "deck (M35)": (4.45, 0.12),
                 "_note": "IS 516 (Part 5/Sec 1):2018, direct transmission through the web where possible"},
    "cover_mm": {"girder links (spec 50)": (55.0, 4.0), "ducts (spec 75)": (80.0, 5.0), "deck (spec 50)": (54.0, 5.0),
                 "_note": "cover meter; tolerance +10/-0 mm"},
    "half_cell_mV_CSE": {"all": (-120.0, 40.0),
                         "_note": "ASTM C876: > -200 mV: < 10 % probability of active corrosion"},
    "resistivity_kOhm_cm": {"girder": (35.0, 8.0), "deck": (25.0, 6.0),
                            "_note": "Wenner probe; > 20 kOhm cm: low corrosion rate"},
    "carbonation_depth_mm": {"model": "x = k sqrt(t), k = 1.5 mm/sqrt(year) for M45, 2.0 for M35",
                             "_note": "phenolphthalein on cores / drilled holes"},
    "chloride_at_steel_pct_cement": {"all": (0.05, 0.02), "_note": "threshold ~0.3-0.4 % by mass of cement"},
    "duct_grouting": {"all": "fully grouted", "_note": "GPR / ultrasonic tomography over the ducts, "
                      "impact-echo at anchorages; voids are a construction defect to rule out at handing over"},
}
