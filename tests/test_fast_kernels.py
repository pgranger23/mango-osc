"""Performance kernels (mango._small sincos / arccos; probability_earth's packed
path for a concrete zenith grid; propagate_layers_pdg(mass_states=True)): the
same values and derivatives as the plain formulations to floating-point
precision.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import nufit_no, probability_earth, LayeredEarth, prem_layered
from mango import constants as C
from mango import earth as E
from mango import layers as L
from mango import oscillator as O
from mango import _small

FOUR = LayeredEarth(outer=jnp.asarray([1221.5, 3480.0, 5701.0, 6371.0]),
                    density=jnp.asarray([13.0, 11.3, 5.0, 3.3]),
                    ye=jnp.asarray([0.468, 0.468, 0.496, 0.496]))
E_GEV = jnp.asarray([0.4, 2.0, 9.0, 60.0])
CZ = np.asarray([-1.0, -0.97, -0.6, -0.2, -1e-3, -1e-9, 0.0, 1e-9, 0.6, 1.0])
W = jnp.asarray([[0.3, -1.2, 2.0], [1.7, 0.4, -0.8], [-0.6, 2.5, 1.1]])


def _p(**over):
    p = nufit_no()
    return dataclasses.replace(p, **{k: jnp.asarray(v) for k, v in over.items()})


def test_sincos_matches_jnp():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-10, 10, 20000), rng.uniform(-1e6, 1e6, 20000),
                        np.exp(rng.uniform(np.log(1e-300), np.log(1e8), 20000)),
                        np.arange(1, 2001) * (np.pi / 2), [0.0, -0.0, 1e-310]])
    s, c = jax.jit(_small.sincos)(jnp.asarray(x))
    assert np.max(np.abs(np.asarray(s) - np.sin(x))) <= 2.3e-16
    assert np.max(np.abs(np.asarray(c) - np.cos(x))) <= 2.3e-16
    sp = jnp.asarray([np.inf, -np.inf, np.nan])
    assert np.all(np.isnan(np.asarray(_small.sincos(sp))))
    # out of range: bounded, on the unit circle
    s, c = _small.sincos(jnp.asarray([1e17, -3e300]))
    assert np.allclose(np.asarray(s * s + c * c), 1.0, atol=1e-15)
    # pinned: same values (to the fusion-dependent rounding of a separate jit)
    s2, c2 = jax.jit(lambda x: _small.sincos(x, pinned=True))(jnp.asarray(x))
    assert np.max(np.abs(np.asarray(s2) - np.sin(x))) <= 2.3e-16
    assert np.max(np.abs(np.asarray(c2) - np.cos(x))) <= 2.3e-16


def test_arccos_matches_jnp():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(-1, 1, 40000), 1 - np.exp(rng.uniform(-38, -1, 5000)),
                        -1 + np.exp(rng.uniform(-38, -1, 5000)), [-1.0, 1.0, 0.0, 0.5, -0.5]])
    a = np.asarray(jax.jit(_small.arccos)(jnp.asarray(x)))
    assert np.max(np.abs(a - np.arccos(x))) <= 4.5e-16
    assert np.all(np.isnan(np.asarray(_small.arccos(jnp.asarray([1.5, -1.0000001, np.nan])))))


def test_trig_derivatives():
    x = jnp.asarray(np.random.default_rng(2).uniform(-40, 40, 300))
    f1 = lambda x: (lambda s, c: s * 1.3 + c * c)(*_small.sincos(x))
    f0 = lambda x: jnp.sin(x) * 1.3 + jnp.cos(x) ** 2
    for op in (jax.grad, jax.jacfwd, jax.hessian):
        a, b = jax.vmap(op(f1))(x), jax.vmap(op(f0))(x)
        assert np.max(np.abs(np.asarray(a - b))) < 1e-14
    r = jnp.asarray(np.random.default_rng(3).uniform(-0.99, 0.99, 300))
    for op in (jax.grad, jax.hessian):
        a, b = jax.vmap(op(_small.arccos))(r), jax.vmap(op(jnp.arccos))(r)
        assert np.max(np.abs(np.asarray(a - b)) / (1 + np.abs(np.asarray(b)))) < 1e-14


def _both(p, kw, cz=CZ):
    """probability_earth on a concrete zenith grid (packed path) and with the
    same grid traced (per-point path)."""
    new = jax.jit(lambda q: probability_earth(q, E_GEV, cz, **kw))(p)
    old = jax.jit(lambda q, c: probability_earth(q, E_GEV, c, **kw))(p, jnp.asarray(cz))
    return np.asarray(new), np.asarray(old)


@pytest.mark.parametrize("det_depth", [0.0, 1.5])
@pytest.mark.parametrize("model", [None, "four", "prem2"])
def test_packed_earth_matches_per_point(det_depth, model):
    kw = dict(det_depth_km=det_depth, h_atm_km=15.0, n_sub=2)
    if model == "four":
        kw["earth_model"] = FOUR
    elif model == "prem2":
        kw["earth_model"] = prem_layered(2)
    outer = E.shell_table(2).outer if model is None else kw["earth_model"].outer
    mirror = O._earth_mirror(outer, det_depth, C.R_EARTH_KM if model is None else None)
    trim = O._grid_trim(CZ, outer, det_depth, mirror,
                        C.R_EARTH_KM if model is None else None)
    assert trim is not None and trim.sum() > 0
    cases = [(_p(dm31=-2.45e-3, deltacp=4.1), True, False), (_p(theta13=0.0), False, True)]
    if model is None:
        cases.append((_p(dm21=0.0), True, True))
    for p, anti, ms in cases:
        new, old = _both(p, dict(kw, anti=anti, mass_states=ms))
        # (the traced and the constant-folded geometry differ by rounding;
        # grazing chords amplify it, see test_packed_earth_exact)
        assert np.max(np.abs(new - old)) < 1e-12


def test_packed_earth_exact():
    """Same layer products as the per-point path: with the geometry evaluated
    identically (both on a concrete grid), equal to rounding."""
    kw = dict(h_atm_km=0.0, n_sub=1)
    new = np.asarray(jax.jit(lambda q: probability_earth(q, E_GEV, CZ, **kw))(nufit_no()))
    orig = O._grid_trim
    O._grid_trim = lambda *a: None
    try:
        old = np.asarray(jax.jit(lambda q: probability_earth(q, E_GEV, CZ, **kw))(nufit_no()))
    finally:
        O._grid_trim = orig
    assert np.max(np.abs(new - old)) < 1e-15


def test_packed_earth_gradients():
    kw = dict(h_atm_km=15.0, n_sub=2)
    loss = lambda p, cz: jnp.sum(probability_earth(p, E_GEV, cz, **kw) * W)
    a = jax.jit(jax.grad(lambda p: loss(p, CZ)))(nufit_no())
    b = jax.jit(jax.grad(loss))(nufit_no(), jnp.asarray(CZ))
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert abs(float(x) - float(y)) <= 1e-11 * (1.0 + abs(float(y)))
    jf = jax.jit(jax.jacfwd(lambda p: probability_earth(p, E_GEV, CZ, **kw)))(nufit_no())
    jo = jax.jit(jax.jacfwd(lambda p, c: probability_earth(p, E_GEV, c, **kw)))(nufit_no(), jnp.asarray(CZ))
    for x, y in zip(jax.tree_util.tree_leaves(jf), jax.tree_util.tree_leaves(jo)):
        assert np.max(np.abs(np.asarray(x) - np.asarray(y))) <= 1e-10 * (1 + np.max(np.abs(np.asarray(y))))
    # LayeredEarth leaves (radii traced: only down-going chords are trimmed)
    lm = lambda m, cz: jnp.sum(probability_earth(nufit_no(), E_GEV, cz, earth_model=m,
                                                 h_atm_km=15.0) * W)
    a = jax.jit(jax.grad(lambda m: lm(m, CZ)))(FOUR)
    b = jax.jit(jax.grad(lm))(FOUR, jnp.asarray(CZ))
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert np.max(np.abs(np.asarray(x) - np.asarray(y))) <= 1e-11 * (1 + np.max(np.abs(np.asarray(y))))


def test_packed_earth_shapes_and_parallel(monkeypatch):
    p = nufit_no()
    for e, cz in ((5.0, -0.3), (E_GEV, -0.3), (5.0, CZ)):
        a = probability_earth(p, e, cz, h_atm_km=15.0)
        b = jax.jit(lambda c: probability_earth(p, e, c, h_atm_km=15.0))(jnp.asarray(cz))
        assert a.shape == b.shape
        assert np.max(np.abs(np.asarray(a - b))) < 1e-12
    sel = probability_earth(p, E_GEV, CZ, h_atm_km=15.0, flavor_in=1, flavor_out=0)
    assert sel.shape == (len(CZ), len(E_GEV))
    new, old = _both(p, dict(h_atm_km=15.0))
    orig = L._propagate_pdg_rows
    monkeypatch.setattr(L, "_propagate_pdg_rows",
                        lambda *a, **k: orig(*a, **dict(k, parallel=True)))
    for dd in (0.0, 1.5):
        par, old = _both(p, dict(h_atm_km=15.0, det_depth_km=dd))
        assert np.max(np.abs(par - old)) < 1e-12


def test_pack():
    lengths = np.array([5, 0, 3, 3, 2, 5, 1, 1, 4])
    cap, n_bins, place = L._pack(lengths)
    assert cap == 5 and n_bins == 5  # 24 slots in bins of 5: 5 bins
    used = np.zeros((n_bins, cap), int)
    for n, (b, t0) in zip(lengths, place):
        if n:
            used[b, t0:t0 + n] += 1
    assert used.max() == 1 and used.sum() == lengths.sum()


@pytest.mark.parametrize("anti", [False, True])
def test_propagate_layers_pdg_mass_states(anti):
    p = _p(deltacp=1.3)
    rho, ye, lk = E.layered_chord_segments(FOUR, jnp.asarray(-0.7), h_atm_km=15.0)
    v = C.matter_potential_eV(rho[1:], ye[1:])
    lv = lk[1:] * C.KM_TO_INV_EV
    s = L.propagate_layers_pdg(p, 3e9, v, lv, anti=anti, mirror=(0, 4))
    a = L.propagate_layers_pdg(p, 3e9, v, lv, anti=anti, mirror=(0, 4), mass_states=True)
    u = jnp.conj(p.pmns()) if anti else p.pmns()
    su = np.asarray(s @ u)
    a = np.asarray(a)
    # equal up to a phase per column (the third: exp(-+i delta))
    np.testing.assert_allclose(np.abs(a), np.abs(su), atol=1e-14)
    ph = np.exp((1j if anti else -1j) * float(p.deltacp))
    np.testing.assert_allclose(a[:, 2] * ph, su[:, 2], atol=1e-14)
    np.testing.assert_allclose(a[:, :2], su[:, :2], atol=1e-14)
