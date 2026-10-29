"""Cable geometry, short-term losses and time-dependent material models.

x is measured from the left bearing (girder ends at -END_OVERHANG and L + END_OVERHANG);
xg = x + END_OVERHANG is measured from the left girder end.  Heights are above the soffit.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import config as C


# ====================================================================== geometry
@dataclass
class Cable:
    name: str
    n: int
    stage: int
    y_anchor: float
    y_mid: float
    x_vertex: float      # from girder end; None -> mid-length
    z_mid: float
    dev0: float
    dev1: float

    @property
    def area(self):
        return self.n * C.STRAND_AREA

    def _xg(self, x):
        half = C.L_GIRDER / 2
        xg = np.asarray(x, float) + C.END_OVERHANG
        return np.where(xg > half, C.L_GIRDER - xg, xg)       # symmetric: distance from nearer end

    def y(self, x):
        xg = self._xg(x)
        xv = C.L_GIRDER / 2 if self.x_vertex is None else self.x_vertex
        r = np.clip(1 - xg / xv, 0, None)
        return self.y_mid + (self.y_anchor - self.y_mid) * r ** 2

    def z(self, x):
        if self.dev0 is None:
            return np.zeros_like(np.asarray(x, float))
        t = np.clip((self._xg(x) - self.dev0) / (self.dev1 - self.dev0), 0, 1)
        return self.z_mid * (3 * t ** 2 - 2 * t ** 3)


def cables():
    return [Cable(*c) for c in C.CABLES]


def stage_cables(stage):
    return [c for c in cables() if c.stage == stage]


def anchor_angle(c):
    """Vertical inclination at the anchorage (rad)."""
    xv = C.L_GIRDER / 2 if c.x_vertex is None else c.x_vertex
    return math.atan(2 * (c.y_anchor - c.y_mid) / xv)


def min_radius(c):
    xv = C.L_GIRDER / 2 if c.x_vertex is None else c.x_vertex
    Rv = xv ** 2 / (2 * (c.y_anchor - c.y_mid))
    if c.dev0 is None:
        return Rv, None
    d = c.dev1 - c.dev0
    Rh = d ** 2 / (6 * abs(c.z_mid))           # max curvature of the cubic S-curve is 6 z / d^2
    return Rv, Rh


# ====================================================================== friction and slip
def _profile(c, n=3381):
    """Fine grid from the left girder end to mid-length; cumulative angle and length."""
    xg = np.linspace(0, C.L_GIRDER / 2, n)
    x = xg - C.END_OVERHANG
    y, z = c.y(x), c.z(x)
    ay = np.arctan(np.gradient(y, xg))
    az = np.arctan(np.gradient(z, xg))
    dtheta = np.hypot(np.diff(ay), np.diff(az))
    theta = np.concatenate([[0.0], np.cumsum(dtheta)])
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(xg) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2))])
    return x, theta, s


def stress_after_friction_slip(c, sigma_j=None):
    """Stress (MPa) along half the cable after friction and wedge slip; stressed from both ends.

    Returns x (from bearing), sigma_jack-friction, sigma after slip, set length.
    """
    sigma_j = C.JACK_RATIO * C.STRAND_FPK if sigma_j is None else sigma_j
    x, theta, s = _profile(c)
    sf = sigma_j * np.exp(-C.MU * (theta + C.WOBBLE * s))
    # slip: the stress diagram is mirrored about sigma(l_set); area between = Ep * slip
    target = C.STRAND_EP * C.SLIP
    area = np.array([np.trapezoid(2 * (sf[: i + 1] - sf[i]), s[: i + 1]) for i in range(len(s))])
    if area[-1] < target:
        # set length exceeds half the cable: whole half-length loses an extra uniform amount
        extra = (target - area[-1]) / s[-1]
        sa = 2 * sf[-1] - sf - extra
        lset = s[-1]
    else:
        i = int(np.searchsorted(area, target))
        lset = s[i]
        sa = np.where(s < lset, 2 * sf[i] - sf, sf)
    return x, sf, sa, lset


def half_to_full(x_half, v_half, x):
    """Evaluate a symmetric half-span quantity at any x (from bearing)."""
    xm = np.where(np.asarray(x) > C.L / 2, C.L - np.asarray(x), np.asarray(x))
    return np.interp(xm, x_half, v_half)


# ====================================================================== concrete time functions (IRC:112 Annex A2)
def fcm_t(fck, t, s=0.25):
    """Mean cube strength at age t (IRC:112 A2.2, cement class N: s = 0.25)."""
    return math.exp(s * (1 - math.sqrt(28 / t))) * (fck + 10)


def fck_t(fck, t):
    return fcm_t(fck, t) - 10 if t < 28 else fck


def ecm_t(fck, t):
    return (fcm_t(fck, t) / (fck + 10)) ** 0.3 * 22.0 * ((fck + 10) / 12.5) ** 0.3 * 1e3


def fctm_t(fck, t):
    fcm = fck + 10
    return (fcm_t(fck, t) / fcm) * 0.259 * fck ** (2 / 3)


def creep_coefficient(fck, h0_mm, t0, t, rh=C.RH):
    """phi(t, t0), IRC:112 Annex A2 (cube-strength form, 43.75 MPa threshold)."""
    fcm = fck + 10
    a1, a2, a3 = (43.75 / fcm) ** 0.7, (43.75 / fcm) ** 0.2, (43.75 / fcm) ** 0.5
    if fcm <= 43.75:
        a1 = a2 = a3 = 1.0
    phi_rh = (1 + (1 - rh / 100) / (0.1 * h0_mm ** (1 / 3)) * a1) * a2
    beta_fcm = 18.78 / math.sqrt(fcm)
    beta_t0 = 1 / (0.1 + t0 ** 0.2)
    beta_h = min(1.5 * (1 + (0.012 * rh) ** 18) * h0_mm + 250 * a3, 1500 * a3)
    beta_c = ((t - t0) / (beta_h + t - t0)) ** 0.3 if t > t0 else 0.0
    return phi_rh * beta_fcm * beta_t0 * beta_c


def shrinkage(fck, h0_mm, t, ts=C.T_CURE, rh=C.RH):
    """Total shrinkage strain (positive = shortening) at age t: drying + autogenous.

    Drying per the EN 1992-1-1 Annex B model on which IRC:112 A2 is based (cement class N),
    with the cylinder strength taken as 0.8 x cube strength.
    """
    fck_cyl = 0.8 * fck
    fcm_cyl = fck_cyl + 8
    beta_rh = 1.55 * (1 - (rh / 100) ** 3)
    ecd0 = 0.85 * ((220 + 110 * 4) * math.exp(-0.12 * fcm_cyl / 10)) * 1e-6 * beta_rh
    kh = float(np.interp(h0_mm, [100, 200, 300, 500], [1.0, 0.85, 0.75, 0.70]))
    beta_ds = (t - ts) / ((t - ts) + 0.04 * math.sqrt(h0_mm ** 3)) if t > ts else 0.0
    eca_inf = 2.5 * (fck_cyl - 10) * 1e-6
    beta_as = 1 - math.exp(-0.2 * math.sqrt(t))
    return beta_ds * kh * ecd0 + beta_as * eca_inf


def relaxation_loss(sigma_pi):
    """Final relaxation loss (MPa) of low-relaxation strand from initial stress sigma_pi."""
    r, pct = zip(*C.RELAX_1000H)
    rho = float(np.interp(sigma_pi / C.STRAND_FPK, r, pct)) / 100
    return C.RELAX_FINAL_FACTOR * rho * sigma_pi


def time_dependent_loss(sigma_pi, sigma_c_qp, phi, eps_cs, Ac, Ic, zcp, Ap, Ecm):
    """Creep + shrinkage + relaxation loss (MPa), IRC:112 eq. 7.4 (cf. EN 1992-1-1 eq. 5.46).

    sigma_c_qp: concrete stress (MPa, compression +) at the tendon under the quasi-permanent
    actions; zcp: tendon eccentricity; Ac, Ic: section of the concrete that the tendon is bonded to.
    """
    n = C.STRAND_EP / Ecm
    d_pr = relaxation_loss(sigma_pi)
    num = eps_cs * C.STRAND_EP + 0.8 * d_pr + n * phi * sigma_c_qp
    den = 1 + n * Ap / Ac * (1 + Ac * zcp ** 2 / Ic) * (1 + 0.8 * phi)
    return num / den, d_pr
