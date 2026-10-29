"""Acceleration records by modal superposition, and the signal-processing chain

    a(t) -> detrend / filter -> Welch PSD / FFT -> peak picking -> frequency domain decomposition.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import config as C
from . import grillage as G

FS = 100.0                 # Hz, recorded sampling rate
FS_SIM = 400.0             # Hz, internal simulation rate
DAMPING = 0.015            # modal damping ratio, PSC deck (assumed for all modes)
SENSOR_NOISE = 2e-4        # m/s^2 RMS (about 20 micro-g), accelerometer + DAQ noise


@dataclass
class Sensor:
    name: str
    x: float
    girder: int


def sensor_layout():
    """8 vertical accelerometers: L/4, L/2, 3L/4 on both outer girders + mid-span of G3 and G4."""
    out = []
    for g in (0, C.N_GIRDERS - 1):
        for f, tag in ((0.25, "Q1"), (0.5, "M"), (0.75, "Q3")):
            out.append(Sensor(f"A-G{g+1}-{tag}", f * C.L, g))
    out += [Sensor("A-G3-M", C.L / 2, 2), Sensor("A-G4-M", C.L / 2, 3)]
    return out


def sensor_rows(bg, sensors):
    """Matrix S (n_sensors x ndof) picking the vertical displacement at each sensor."""
    Smat = np.zeros((len(sensors), bg.grid.ndof))
    for k, s in enumerate(sensors):
        Smat[k] += -G.point_load_vector(bg, s.x, bg.ys[s.girder], 1.0)     # lever weights, +w
    return Smat


# ---------------------------------------------------------------------- excitation
VEHICLE_TYPES = [
    # name, axle loads (kN), axle spacing (m), share of traffic
    ("car", [7.0, 7.0], [2.6], 0.55),
    ("LCV", [25.0, 45.0], [3.4], 0.20),
    ("2-axle truck", [60.0, 100.0], [4.2], 0.12),
    ("3-axle truck", [60.0, 95.0, 95.0], [4.3, 1.35], 0.10),
    ("multi-axle", [60.0, 100.0, 100.0, 80.0, 80.0], [3.5, 1.35, 5.0, 1.35], 0.03),
]
LANE_Y = [C.CW_LEFT + 1.85, C.CW_LEFT + 5.5, C.CW_LEFT + 9.15]     # wheel-path centres of the 3 lanes


def traffic(duration, rate_per_min, rng, heavy_factor=1.0):
    """Random vehicles: (t_enter, speed m/s, lane y, axle loads, axle offsets)."""
    n = rng.poisson(rate_per_min * duration / 60)
    shares = np.array([v[3] for v in VEHICLE_TYPES])
    out = []
    for _ in range(n):
        k = rng.choice(len(VEHICLE_TYPES), p=shares / shares.sum())
        name, P, sp, _ = VEHICLE_TYPES[k]
        P = np.array(P) * rng.uniform(0.7, 1.15) * (heavy_factor if k >= 2 else 1.0)
        off = np.concatenate([[0.0], np.cumsum(sp)])
        out.append((rng.uniform(0, duration), rng.uniform(30, 70) / 3.6, LANE_Y[rng.integers(3)], P, off))
    return out


def modal_ordinates(bg, modes):
    """phi[r, g, i]: vertical mode ordinates along each girder line at the stations."""
    return np.transpose(modes.w_shapes, (2, 1, 0))


def modal_forces(bg, modes, t, vehicles, rng, ambient_rms=0.2):
    """Generalised forces Q[r, t] from moving axles (lever rule across, linear along) + ambient noise."""
    phi = modal_ordinates(bg, modes)
    nm = phi.shape[0]
    Q = np.zeros((nm, len(t)))
    xs, ys = bg.xs, bg.ys
    for t0, v, y, P, off in vehicles:
        for wy_side in (-0.9, 0.9):                    # two wheel lines
            lw = G.lever_weights(y + wy_side, ys)
            for Pa, oa in zip(P, off):
                x = v * (t - t0) - oa
                on = (x >= 0) & (x <= C.L)
                if not on.any():
                    continue
                for j, wy, _ in lw:
                    # downward axle load: generalised force = -P * phi(+w up)
                    Q[:, on] -= 0.5 * Pa * wy * np.array([np.interp(x[on], xs, phi[r, j]) for r in range(nm)])
    # ambient (wind, micro-tremor): independent white noise of RMS ambient_rms (kN) on every vertical
    # dof.  Its modal projection Phi_w^T n has covariance ambient_rms^2 Phi_w^T Phi_w, sampled directly.
    nodes = np.arange(len(xs) * len(ys))
    Phi_w = modes.shapes[3 * nodes]                      # (n_nodes, nm)
    Lc = np.linalg.cholesky(Phi_w.T @ Phi_w + 1e-12 * np.eye(nm))
    Q += ambient_rms * (Lc @ rng.standard_normal((nm, len(t))))
    return Q


def modal_acceleration(Q, freq, zeta=DAMPING, fs=FS_SIM):
    """Modal accelerations for generalised forces Q (mass-normalised modes), exact ZOH discretisation."""
    out = np.zeros_like(Q)
    for r, f in enumerate(freq):
        w = 2 * math.pi * f
        A = np.array([[0, 1], [-w * w, -2 * zeta * w]])
        B = np.array([[0], [1]])
        Cm = np.array([[-w * w, -2 * zeta * w]])
        D = np.array([[1.0]])
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, Cm, D), 1 / fs, method="zoh")
        b, a = signal.ss2tf(Ad, Bd, Cd, Dd)
        out[r] = signal.lfilter(b[0], a, Q[r])
    return out


def simulate(bg, modes, duration=600.0, rate_per_min=12.0, seed=0, ambient_rms=0.2, sensors=None,
             heavy_factor=1.0, noise=SENSOR_NOISE):
    """Acceleration record at the sensors, sampled at FS. Returns t, a (n_sensors x n), sensors."""
    rng = np.random.default_rng(seed)
    sensors = sensor_layout() if sensors is None else sensors
    t = np.arange(0, duration, 1 / FS_SIM)
    veh = traffic(duration, rate_per_min, rng, heavy_factor)
    Q = modal_forces(bg, modes, t, veh, rng, ambient_rms)
    qdd = modal_acceleration(Q, modes.freq)
    Smat = sensor_rows(bg, sensors)
    a = (Smat @ modes.shapes) @ qdd                        # +w up
    q = int(FS_SIM // FS)
    a = signal.decimate(a, q, ftype="fir", axis=1, zero_phase=True)
    a += rng.normal(0, noise, a.shape)
    return t[::q], a, sensors


# ---------------------------------------------------------------------- processing
def preprocess(a, fs=FS, band=(0.5, 30.0)):
    a = signal.detrend(a, axis=-1)
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, a, axis=-1)


def welch_psd(a, fs=FS, nperseg=4096):
    f, P = signal.welch(a, fs=fs, nperseg=nperseg, axis=-1)
    return f, P


def peak_pick(f, P, bands):
    """Frequency of the maximum of the channel-averaged (normalised) PSD inside each band."""
    Pn = (P / P.max(axis=-1, keepdims=True)).mean(axis=0) if P.ndim > 1 else P
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        k = np.argmax(Pn[m])
        # parabolic interpolation of the peak
        idx = np.flatnonzero(m)[k]
        if 0 < idx < len(f) - 1:
            y0, y1, y2 = np.log(Pn[idx - 1:idx + 2])
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            out.append(float(f[idx] + d * (f[1] - f[0])))
        else:
            out.append(float(f[idx]))
    return out


def fdd(a, fs=FS, nperseg=4096):
    """Frequency domain decomposition: singular values and first singular vectors of the CSD matrix."""
    n = a.shape[0]
    f, _ = signal.csd(a[0], a[0], fs=fs, nperseg=nperseg)
    Gm = np.zeros((len(f), n, n), complex)
    for i in range(n):
        for j in range(i, n):
            _, Gij = signal.csd(a[i], a[j], fs=fs, nperseg=nperseg)
            Gm[:, i, j] = Gij
            Gm[:, j, i] = np.conj(Gij)
    U, s, _ = np.linalg.svd(Gm)
    return f, s, U


def fdd_modes(f, s, U, bands):
    """Peak of the first singular value in each band -> frequency and real-normalised mode shape."""
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        idx = np.flatnonzero(m)[np.argmax(s[m, 0])]
        u = U[idx, :, 0]
        u = np.real(u * np.exp(-1j * np.angle(u[np.argmax(abs(u))])))
        out.append((float(f[idx]), u / np.max(abs(u))))
    return out


def csd_matrix(a, fs=FS, nperseg=4096):
    n = a.shape[0]
    f, _ = signal.csd(a[0], a[0], fs=fs, nperseg=nperseg)
    Gm = np.zeros((len(f), n, n), complex)
    for i in range(n):
        for j in range(i, n):
            _, Gij = signal.csd(a[i], a[j], fs=fs, nperseg=nperseg)
            Gm[:, i, j] = Gij
            Gm[:, j, i] = np.conj(Gij)
    return f, Gm


def modal_filter_peaks(f, Gm, refs, bands):
    """Spatial (modal) filtering: p_r(f) = phi_r^T G(f) phi_r / phi_r^T phi_r for a reference shape phi_r
    (sensor level, e.g. the baseline mode).  Orthogonal shapes (bending vs. torsion) are separated even
    when their frequencies are close or cross.  Returns the interpolated peak frequency of each filter."""
    out = []
    for phi, (lo, hi) in zip(refs, bands):
        phi = np.asarray(phi, float)
        p = np.real(np.einsum("i,fij,j->f", phi, Gm, phi)) / (phi @ phi)
        m = np.flatnonzero((f >= lo) & (f <= hi))
        k = m[np.argmax(p[m])]
        if 0 < k < len(f) - 1 and p[k - 1] > 0 and p[k + 1] > 0:
            y0, y1, y2 = np.log(p[k - 1:k + 2])
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            out.append(float(f[k] + d * (f[1] - f[0])))
        else:
            out.append(float(f[k]))
    return out


def features(a):
    """Time-domain features per record: RMS and peak acceleration (max over channels)."""
    return dict(a_rms=float(np.sqrt((a ** 2).mean(axis=1)).max()), a_max=float(abs(a).max()))
