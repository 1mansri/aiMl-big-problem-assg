"""Hand-check tests for the analysis tools. Run: python -m pytest tests  (or python tests/test_step1.py)."""
import numpy as np

from bridge import config as C
from bridge import grillage as G
from bridge import loads, moving
from bridge.section import props


def single_beam(n=10, Lb=10.0, EI=1e6):
    g = G.Grid()
    nodes = [g.add_node(i * Lb / n, 0.0) for i in range(n + 1)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        g.add_member(a, b, EI, 1e5)
    g.fixed |= {0, 3 * n}
    g.fixed.add(1)          # twist restraint (single line has no torsional support otherwise)
    return g, nodes


def test_beam_point_load():
    g, nodes = single_beam()
    F = np.zeros(g.ndof)
    F[3 * nodes[5]] = -100.0
    u = g.solve(F)
    m = g.members[4]
    M = g.moment_row(m, 1) @ u
    assert abs(M - 100 * 10 / 4) < 1e-6
    assert abs(-u[3 * nodes[5]] - 100 * 10 ** 3 / (48 * 1e6)) < 1e-9
    V = g.shear_row(g.members[0]) @ u
    assert abs(V - 50) < 1e-6


def test_beam_lumped_udl():
    g, nodes = single_beam()
    F = np.zeros(g.ndof)
    for k in nodes:
        F[3 * k] = -10.0 * (0.5 if k in (nodes[0], nodes[-1]) else 1.0)       # q = 10 kN/m
    u = g.solve(F)
    assert abs(g.moment_row(g.members[4], 1) @ u - 10 * 100 / 8) < 1e-6


def test_trapezoid_props():
    p = props([(0.3, 0.3, 0.6)])
    assert abs(p.I - 0.3 * 0.6 ** 3 / 12) < 1e-12 and abs(p.yb - 0.3) < 1e-12
    p = props([(0.0, 1.0, 1.0)])           # triangle apex down
    assert abs(p.yb - 2 / 3) < 1e-12 and abs(p.I - 1.0 / 36) < 1e-12


def test_ss_beam_udl_with_overhang():
    xs, M, V, (RA, RB) = loads.ss_beam(lambda x: 10.0, xs=[0.0, C.L / 2])
    a = C.END_OVERHANG
    total = 10 * (C.L + 2 * a)
    assert abs(RA - total / 2) < 1e-6
    Mmid = RA * C.L / 2 - 10 * (C.L / 2 + a) ** 2 / 2
    assert abs(M[1] - Mmid) < 1e-3


def test_grillage_static_equilibrium():
    bg = G.build()
    i_mid = len(bg.xs) // 2
    F = G.point_load_vector(bg, C.L / 2, 8.0, 100.0)
    u = bg.grid.solve(F)
    Msum = sum(bg.grid.moment_row(bg.grid.members[bg.long[(i_mid - 1, j)]], 1) @ u for j in range(len(bg.ys)))
    assert abs(Msum - 100 * C.L / 4) < 1e-6
    R = sum(bg.grid.reaction_sensitivity(d) @ F for _, _, d in bg.supports)
    assert abs(R - 100) < 1e-6


def test_load_outside_girders_torque():
    bg = G.build()
    F = G.line_load_vector(bg, 0.1, 10.0)          # line load on the cantilever
    R = sum(bg.grid.reaction_sensitivity(d) @ F for _, _, d in bg.supports)
    assert abs(R - 10 * C.L) < 1e-6
    # moment of the reactions about the deck edge equals that of the load
    Ry = sum((bg.grid.reaction_sensitivity(d) @ F) * bg.ys[j] for j, _, d in bg.supports)
    assert abs(Ry - 10 * C.L * 0.1) < 1e-6


def test_moving_search_reproduces_value():
    bg = G.build()
    i_mid = len(bg.xs) // 2
    s = G.girder_moment_sensitivities(bg)[(0, i_mid)]
    p = moving.envelope(bg, s)
    F = moving.placement_load_vector(bg, p)
    assert abs(s @ F - p.value) < 1e-6 * abs(p.value)


def test_config_names_assigned_once():
    """Guard against a later constant silently overwriting an earlier one (happened with T_DECK)."""
    import ast
    import pathlib
    src = pathlib.Path(C.__file__).read_text()
    names = [t.id for n in ast.parse(src).body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)]
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})


def test_footpath_intensity():
    # (400 - 260 + 4800/33) * (16.5 - 1.5)/15 = 285.45 kg/m^2
    assert abs(loads.footpath_intensity() - 285.4545 * 9.81e-3) < 1e-4


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
