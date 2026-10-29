"""Worst placement of IRC:6 vehicles on a grillage response (influence-surface search).

For a response with sensitivity s (response = s . F), the downward-load influence line of
girder line g is eta_g(x_i) = -s[w_dof(i, g)].  Wheel loads between girder lines are shared
by the lever rule, so for a vehicle whose centre is at (x0, y0):

    E(x0, y0) = sum_lines sum_g phi_g(y0 + dy_line) * A_g(x0),
    A_g(x0)   = sum_axles (P_axle / n_lines) * eta_g(x0 -/+ axle_x)

A_g does not depend on y0, so the search is a small matrix product.  Vehicles in different
lanes are independent, so each one is placed at its own worst x0; the transverse placement
of the vehicles side by side is then optimised by dynamic programming under the IRC
kerb and vehicle-to-vehicle clearances.
"""
from dataclasses import dataclass

import numpy as np

from . import config as C
from . import loads
from .grillage import lever_weights

DX0 = 0.05          # longitudinal search step of the leading axle
DY0 = 0.05          # transverse search step of the vehicle centre


def influence_lines(bg, s):
    """eta[g, i]: downward-unit-load influence ordinates at the grillage stations."""
    ng = len(bg.ys)
    return np.array([[-s[bg.w_dof(i, g)] for i in range(len(bg.xs))] for g in range(ng)])


def _lever_matrix(y, ys):
    """phi[k, g] for an array of y inside the outer girder lines."""
    phi = np.zeros((len(y), len(ys)))
    for k, yy in enumerate(y):
        for g, w, e in lever_weights(yy, ys):
            phi[k, g] += w            # wheels never fall outside the outer girders here
    return phi


def vehicle_y_grid(v):
    lo = C.CW_LEFT + v.kerb_clear + v.width / 2
    hi = C.CW_RIGHT - v.kerb_clear - v.width / 2
    return np.arange(lo, hi + 1e-9, DY0)


def vehicle_envelope(bg, eta, v, sign=1.0):
    """Max of sign*E over x0 (both directions of travel) for every centre y on the grid.

    Returns y grid, best value (impact included), and the x0/direction of each best value.
    """
    xs = bg.xs
    x0 = np.arange(-v.length - 1.0, C.L + v.length + 1.0, DX0)
    y = vehicle_y_grid(v)
    best = np.full(len(y), -np.inf)
    best_info = [None] * len(y)
    for direction in (+1, -1):
        pos = x0[:, None] - direction * v.axle_x[None, :]            # (n_x0, n_axles)
        on = (pos >= xs[0]) & (pos <= xs[-1])
        A = np.zeros((len(x0), eta.shape[0]))
        for g in range(eta.shape[0]):
            ordin = np.interp(pos, xs, eta[g]) * on
            A[:, g] = ordin @ (v.axle_P / len(v.wheel_dy))
        E = np.zeros((len(y), len(x0)))
        for dy in v.wheel_dy:
            E += _lever_matrix(y + dy, bg.ys) @ A.T
        E *= sign * (1 + v.impact)
        k = E.argmax(axis=1)
        val = E[np.arange(len(y)), k]
        upd = val > best
        best[upd] = val[upd]
        for idx in np.flatnonzero(upd):
            best_info[idx] = (float(x0[k[idx]]), direction)
    return y, best, best_info


@dataclass
class Placement:
    case: str
    value: float
    vehicles: list          # [(name, y_centre, x_lead, direction)]


def best_case(bg, s, case, sign=1.0, cache=None):
    """Optimal transverse arrangement of the vehicles of a LoadCase (left to right order)."""
    eta = influence_lines(bg, s)
    env = []
    for v in case.vehicles:
        key = (id(s), v.name, sign)
        if cache is not None and key in cache:
            env.append(cache[key])
        else:
            r = vehicle_envelope(bg, eta, v, sign)
            if cache is not None:
                cache[key] = r
            env.append(r)
    # DP over vehicles left -> right
    prev_val, prev_arg = None, []
    for k, (v, (y, val, _)) in enumerate(zip(case.vehicles, env)):
        if k == 0:
            cur = val.copy()
            arg = np.full(len(y), -1)
        else:
            pv, py, pvveh = prev_val, env[k - 1][0], case.vehicles[k - 1]
            # running max of previous best over admissible previous centres
            run = np.empty(len(pv))
            runarg = np.zeros(len(pv), int)
            m, mi = -np.inf, 0
            for i in range(len(pv)):
                if pv[i] > m:
                    m, mi = pv[i], i
                run[i], runarg[i] = m, mi
            limit = y - v.width / 2 - loads.VEHICLE_GAP - pvveh.width / 2   # max previous centre
            idx = np.searchsorted(py, limit + 1e-9, side="right") - 1
            cur = np.full(len(y), -np.inf)
            arg = np.full(len(y), -1)
            ok = idx >= 0
            cur[ok] = val[ok] + run[idx[ok]]
            arg[ok] = runarg[idx[ok]]
        prev_arg.append(arg)
        prev_val = cur
    j = int(np.argmax(prev_val))
    total = prev_val[j]
    if not np.isfinite(total):
        return None
    picks = [j]
    for k in range(len(case.vehicles) - 1, 0, -1):
        j = int(prev_arg[k][j])
        picks.append(j)
    picks = picks[::-1]
    veh = []
    for v, (y, _, info), p in zip(case.vehicles, env, picks):
        veh.append((v.name, float(y[p]), info[p][0], info[p][1]))
    return Placement(case.name, float(total * case.reduction), veh)


def envelope(bg, s, cases=None, sign=1.0):
    """Governing placement over all IRC load cases for one response."""
    cases = loads.irc_live_load_cases() if cases is None else cases
    cache = {}
    best = None
    for case in cases:
        p = best_case(bg, s, case, sign, cache)
        if p is not None and (best is None or p.value > best.value):
            best = p
    if best is not None and sign < 0:
        best.value = -best.value
    return best


def placement_load_vector(bg, placement):
    """Nodal loads of a placement (impact and reduction included) for checking the search."""
    from .grillage import point_load_vector
    allv = {v.name: v for c in loads.irc_live_load_cases() for v in c.vehicles}
    red = {c.name: c.reduction for c in loads.irc_live_load_cases()}[placement.case]
    F = np.zeros(bg.grid.ndof)
    for name, yc, x0, direction in placement.vehicles:
        v = allv[name]
        for xa, P in zip(v.axle_x, v.axle_P):
            for dy in v.wheel_dy:
                F += point_load_vector(bg, x0 - direction * xa, yc + dy,
                                       red * (1 + v.impact) * P / len(v.wheel_dy))
    return F
