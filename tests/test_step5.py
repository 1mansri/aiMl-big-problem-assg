"""Checks of the deterioration models and the campaign generator."""
import numpy as np

from bridge import config as C
from bridge import synthetic as SY
from bridge import twin as T


def test_prestress_loss_equivalent_couples_are_in_equilibrium():
    bs = T.build(T.State(), dynamic=False)
    g = 2
    SY._PL_CACHE.clear()
    d, s, so = SY.prestress_loss_unit(bs, g)
    # rebuild the response to read the girder moments: their sum at mid-span = dP * e (static)
    grid = bs.grid
    F = np.zeros(grid.ndof)
    xs = bs.xs
    for i in range(len(xs) - 1):
        m = 1000.0 * SY._tendon_eccentricity(0.5 * (xs[i] + xs[i + 1]), g)
        F[3 * bs.node[i, g] + 2] += m
        F[3 * bs.node[i + 1, g] + 2] -= m
    u = grid.solve(F)
    i_mid = len(xs) // 2
    Msum = sum(grid.moment_row(grid.members[bs.long[(i_mid - 1, j)]], 1) @ u for j in range(6))
    e = SY._tendon_eccentricity(0.5 * (xs[i_mid - 1] + xs[i_mid]), g)
    # the elastomer shear springs (eccentric restraint at the bearings) take a small end moment: allow 1 %
    assert abs(Msum - 1000.0 * e) < 0.01 * 1000.0 * e
    assert d[g] > 0 and so[g] < 0                  # loss: more sag, less soffit compression


def test_prestress_loss_single_beam_matches_hand_formula():
    """Unit loss on a determinate single beam: mid-span deflection = dP * integral(e * mbar) / EI."""
    bs = T.build(T.State(), dynamic=False)
    SY._PL_CACHE.clear()
    d, _, _ = SY.prestress_loss_unit(bs, 2)
    # the grillage shares the effect, so the loaded girder deflects less than an isolated beam
    xs = np.linspace(0, C.L, 2001)
    e = np.array([SY._tendon_eccentricity(x, 2) for x in xs])
    mbar = np.where(xs <= C.L / 2, xs / 2, (C.L - xs) / 2)
    EI = bs.grid.members[bs.long[(22, 2)]].EI
    d_beam = 1000.0 * np.trapezoid(e * mbar, xs) / EI * 1e3
    total = sum(d)                                   # sum of girder deflections ~ 6x the mean
    assert d[2] < d_beam and abs(total / 6 - d_beam / 6) / (d_beam / 6) < 0.25


def test_damage_moves_frequencies_the_right_way():
    gen = SY.Generator()
    f_h = T.modal(T.build(T.State()), 1).freq[0]
    seized = SY.Scenario("bearing_seized", 1.0, tuple(range(6)), end=1, params=dict(kh=1e6))
    cracked = SY.Scenario("girder_cracking", 1.0, (2,), params=dict(EI=0.55, half_zone=6.0, crack_mm=0.5))
    assert T.modal(T.build(SY.twin_state(seized, 30.0)), 1).freq[0] > f_h
    assert T.modal(T.build(SY.twin_state(cracked, 30.0)), 1).freq[0] < f_h
    assert gen.Pe > 5000


def test_visual_observations_use_valid_states():
    rng = np.random.default_rng(1)
    for kind in SY.KINDS:
        scn = SY.sample_scenario(kind, rng, severity=0.9)
        scn.params["cracked"] = True
        scn.params.setdefault("crack_mm", 0.2)
        env = SY.environment(rng)
        for o in SY.visual_observations(scn, env, rng):
            assert 1 <= o["cs"] <= 4


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
