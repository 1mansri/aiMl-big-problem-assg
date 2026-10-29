"""Adopted configuration of the 33 m post-tensioned PSC I-girder bridge.

Units throughout the package: m, kN, kN·m, kPa (kN/m^2) unless a name says otherwise
(e.g. *_MPa, *_mm).  Transverse coordinate y is measured from the left deck edge,
longitudinal coordinate x from the left bearing line.
"""
import math

# --------------------------------------------------------------------------- span
L = 33.0                 # effective span, c/c of bearings
END_OVERHANG = 0.40      # girder end beyond bearing centreline
L_GIRDER = L + 2 * END_OVERHANG

# --------------------------------------------------------------- cross-section
# Left-to-right strips of the deck (name, width).  Total = deck width.
DECK_STRIPS = [
    ("railing", 0.25),
    ("footpath", 1.50),
    ("crash_barrier", 0.45),
    ("carriageway", 11.00),
    ("crash_barrier", 0.45),
    ("footpath", 1.50),
    ("railing", 0.25),
]
B_DECK = sum(w for _, w in DECK_STRIPS)          # 15.40 m


def strip_extents():
    """[(name, y_left, y_right)] for DECK_STRIPS."""
    out, y = [], 0.0
    for name, w in DECK_STRIPS:
        out.append((name, y, y + w))
        y += w
    return out


CW_LEFT, CW_RIGHT = [(a, b) for n, a, b in strip_extents() if n == "carriageway"][0]  # kerb (barrier) faces
N_LANES = 3                                       # IRC:6 Table 6: 9.6 <= 11.0 < 13.1 m

N_GIRDERS = 6
GIRDER_SPACING = 2.60
CANTILEVER = (B_DECK - (N_GIRDERS - 1) * GIRDER_SPACING) / 2      # 1.20 m
GIRDER_Y = [CANTILEVER + i * GIRDER_SPACING for i in range(N_GIRDERS)]

# --------------------------------------------------------------- PSC I-girder
# Mid-span precast section as trapezoidal layers from the soffit up:
# (width at bottom of layer, width at top of layer, height)
GIRDER_LAYERS = [
    (0.65, 0.65, 0.25),   # bottom flange (houses the cables)
    (0.65, 0.25, 0.15),   # bottom taper
    (0.25, 0.25, 1.15),   # web
    (0.25, 0.90, 0.10),   # top taper
    (0.90, 0.90, 0.15),   # top flange
]
H_GIRDER = sum(h for _, _, h in GIRDER_LAYERS)    # 1.80 m
B_WEB = 0.25
# End block: web thickened to the bottom-flange width near the anchorages
END_BLOCK_WEB = 0.65
END_BLOCK_LEN = 1.50     # full end-block length from girder end
END_BLOCK_TAPER = 1.00   # transition length to the normal web

T_DECK = 0.22            # in-situ deck slab
H_COMPOSITE = H_GIRDER + T_DECK

# --------------------------------------------------------------- cross girders
# x positions measured from the left bearing; the end cross girders sit on the bearing lines
CROSS_GIRDERS = [
    # (x, web thickness, depth below deck soffit)
    (0.0, 0.50, 1.50),
    (L / 2, 0.30, 1.40),
    (L, 0.50, 1.50),
]

# --------------------------------------------------------------- materials
GAMMA_CONC = 25.0        # kN/m^3, PSC and RCC (IRC:6 Table 1)
GAMMA_ASPHALT = 22.0


def concrete(fck):
    """IRC:112-2020 cl. 6.4.2 (Table 6.5) properties for grade fck (cube, MPa)."""
    fcm = fck + 10.0
    return {
        "fck": fck,
        "fcm": fcm,
        "fctm": 0.259 * fck ** (2 / 3) if fck <= 60 else 2.27 * math.log(1 + fcm / 12.5),
        "Ecm": 22.0 * (fcm / 12.5) ** 0.3 * 1e3,       # MPa
    }


GIRDER_CONCRETE = concrete(45.0)
DECK_CONCRETE = concrete(35.0)
POISSON = 0.2

# --------------------------------------------------------------- superimposed dead loads
SIDL_LINE = {                 # kN/m per side, at the strip centre
    "crash_barrier": 8.5,     # New Jersey type, ~0.34 m^2 x 25
    "railing": 2.0,           # pedestrian railing with its kerb/plinth
}
FOOTPATH_FINISH = 3.0         # kPa: 120 mm lean fill + paving over the footpath slab
WEARING_COAT = 2.0            # kPa: 65 mm bituminous (1.43 kPa) plus an allowance for one overlay

# --------------------------------------------------------------- footpath live load
FOOTPATH_P = 400.0            # kg/m^2 basic intensity, IRC:6 cl. 206.1 (non-urban, no crowd)

# --------------------------------------------------------------- prestressing steel
STRAND_AREA = 140.0e-6        # m^2, 15.2 mm 7-wire strand (IRC:112 Table 18.4 / IS 14268 class 2)
STRAND_FPK = 1860.0           # MPa
STRAND_EP = 195.0e3           # MPa
JACK_RATIO = 0.75             # sigma_jack / fpk
MU = 0.17                     # friction coefficient, corrugated HDPE duct (IRC:112 Table 7.1)
WOBBLE = 0.0020               # per m (IRC:112 Table 7.1)
SLIP = 0.006                  # m wedge draw-in (to be confirmed by the PT supplier)
DUCT_OD = 0.095               # m, 12T15 corrugated HDPE duct (ID ~85 mm)
DUCT_COVER = 0.075            # m, severe exposure
RELAX_1000H = [(0.5, 0.0), (0.6, 1.0), (0.7, 2.5), (0.8, 4.5)]   # % loss at sigma/fpk, low relaxation
RELAX_FINAL_FACTOR = 3.0      # final relaxation = 3 x 1000 h value (IRC:112 cl. 6.3.6)

# Cables: (name, strands, stage, anchor height at girder end, height at mid-span,
#          distance of the parabola vertex from the girder end, lateral offset at mid-span,
#          lateral S-curve start and end, measured from the girder end)
CABLES = [
    ("C1", 11, 1, 0.40, 0.125, 3.2, -0.19, 2.7, 5.3),
    ("C2", 11, 1, 0.75, 0.125, 8.5, +0.19, 8.5, 11.0),
    ("C3", 11, 2, 1.10, 0.125, None, 0.0, None, None),      # None: full parabola, vertex at mid-span
    ("C4", 11, 2, 1.45, 0.315, None, 0.0, None, None),
]
# 4 x 11T15 chosen from a strand-count sweep: 3 x 12T15 and 4 x 10T15 crack under the frequent
# combination (with 0.9 P_m), 4 x 12T15 exceeds 0.36 fck under quasi-permanent load near the supports.
# Profiles from a grid search: duct c/c >= 2 x OD everywhere, cover >= 75 mm, anchor spacing 0.35 m.

# --------------------------------------------------------------- construction timeline (girder age, days)
T_STAGE1 = 14                 # stage-1 cables stressed on the precast girder
T_DECK_CAST = 45                   # girders erected, deck slab cast (deck age 0)
T_STAGE2 = 75                 # stage-2 cable stressed on the composite section (deck 30 days old)
T_SIDL = 90                   # barriers, footpaths, wearing coat
T_FINAL = 36500               # 100-year design life
RH = 70.0                     # % ambient relative humidity
T_CURE = 3                    # days of moist curing (start of drying)
ALPHA_T = 11.7e-6             # /degC (IRC:6 cl. 215.4)
