"""Reduced-basis fast path (mango.layers.propagate_layers_pdg) vs the general
flavor-basis propagation: same operator / probabilities / gradients to
floating-point precision, including the mirror-symmetric chord shortcut.

The general path is forced with an all-zero ``NSI()`` (identical physics: it
only adds ``V * 0``), which keeps everything differentiable.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import (nufit_no, probability_earth, probability_profile, NSI,
                   LayeredEarth, prem_layered)
from mango import constants as C
from mango import earth as E
from mango import layers as L

ZERO_NSI = NSI()
FOUR = LayeredEarth(outer=jnp.asarray([1221.5, 3480.0, 5701.0, 6371.0]),
                    density=jnp.asarray([13.0, 11.3, 5.0, 3.3]),
                    ye=jnp.asarray([0.468, 0.468, 0.496, 0.496]))
E_GEV = jnp.asarray([0.4, 2.0, 9.0])
CZ = jnp.asarray([-1.0, -0.9, -0.45, -0.08, -1e-9, 0.0, 0.6])


def _p(**over):
    p = nufit_no()
    return dataclasses.replace(p, **{k: jnp.asarray(v) for k, v in over.items()})


@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_propagate_layers_pdg_matches_general(anti, parallel):
    p = _p(dm31=-2.45e-3, deltacp=1.1)
    for cz in (-0.97, -0.5, -0.1):
        rho, ye, lk = E.chord_segments(jnp.asarray(cz), E.shell_table(2),
                                       h_atm_km=15.0)
        v = C.matter_potential_eV(rho, ye)
        lv = lk * C.KM_TO_INV_EV
        n = (lk.shape[0] - 1) // 2
        ref = L.propagate_layers(p.pmns(), p.msquared(), 3e9, v, lv, anti=anti)
        for mirror in (None, (1, n), (2, n - 1)):
            s = L.propagate_layers_pdg(p, 3e9, v, lv, anti=anti, mirror=mirror,
                                       parallel=parallel)
            assert np.max(np.abs(np.asarray(s - ref))) < 1e-13, (cz, mirror)
        # external-analysis layout: Earth part only, LayeredEarth, surface detector
        rho, ye, lk = E.layered_chord_segments(FOUR, jnp.asarray(cz),
                                               h_atm_km=15.0)
        v = C.matter_potential_eV(rho[1:], ye[1:])
        ref = L.propagate_layers(p.pmns(), p.msquared(), 3e9, v,
                                 lk[1:] * C.KM_TO_INV_EV, anti=anti)
        s = L.propagate_layers_pdg(p, 3e9, v, lk[1:] * C.KM_TO_INV_EV,
                                   anti=anti, mirror=(0, 4), parallel=parallel)
        assert np.max(np.abs(np.asarray(s - ref))) < 1e-13, cz


def test_mirror_validation():
    with pytest.raises(ValueError):
        L.propagate_layers_pdg(nufit_no(), 3e9, jnp.zeros(4), jnp.ones(4),
                               mirror=(1, 2))


@pytest.mark.parametrize("det_depth", [0.0, 1.5])
@pytest.mark.parametrize("model", [None, "four"])
def test_probability_earth_fast_path(det_depth, model):
    kw = dict(det_depth_km=det_depth, h_atm_km=15.0, n_sub=1)
    if model == "four":
        kw["earth_model"] = FOUR
    cases = ((_p(), False, False), (_p(dm31=-2.45e-3), True, False),
             (_p(theta13=0.0), False, True), (_p(dm21=0.0), True, True))
    for p, anti, ms in cases:
        a = probability_earth(p, E_GEV, CZ, anti=anti, mass_states=ms, **kw)
        b = probability_earth(p, E_GEV, CZ, anti=anti, mass_states=ms,
                              nsi=ZERO_NSI, **kw)
        # dm21 = 0: the general path's Cayley is itself off by up to ~1e-11
        # at the exactly degenerate vacuum leg (cz ~ 0), where the fast path's
        # closed-form vacuum propagator is exact.
        tol = 1e-10 if float(p.dm21) == 0.0 else 1e-12
        assert np.max(np.abs(np.asarray(a - b))) < tol


def test_probability_profile_fast_path():
    rng = np.random.default_rng(0)
    dens = jnp.asarray(rng.uniform(0, 13, 9))
    ye = jnp.asarray(rng.uniform(0.4, 0.5, 9))
    lens = jnp.asarray(rng.uniform(0, 900, 9)).at[2].set(0.0)
    p = nufit_no()
    for anti in (False, True):
        for ms in (False, True):
            a = probability_profile(p, E_GEV, dens, ye, lens, anti=anti,
                                    mass_states=ms)
            b = probability_profile(p, E_GEV, dens, ye, lens, anti=anti,
                                    mass_states=ms, nsi=ZERO_NSI)
            assert np.max(np.abs(np.asarray(a - b))) < 1e-12


def test_fast_path_gradients_match_general():
    w = jnp.asarray([[0.3, -1.2, 2.0], [1.7, 0.4, -0.8], [-0.6, 2.5, 1.1]])

    def loss(p, cz, model, nsi):
        return jnp.sum(probability_earth(p, E_GEV, cz, earth_model=model,
                                         h_atm_km=15.0, nsi=nsi) * w)

    for p, model in ((_p(), FOUR), (_p(theta13=0.0), prem_layered(1))):
        ga = jax.grad(loss, argnums=(0, 1, 2))(p, CZ, model, None)
        gb = jax.grad(loss, argnums=(0, 1, 2))(p, CZ, model, ZERO_NSI)
        for x, y in zip(jax.tree_util.tree_leaves(ga),
                        jax.tree_util.tree_leaves(gb)):
            x, y = np.asarray(x), np.asarray(y)
            assert np.all(np.isfinite(x))
            assert np.max(np.abs(x - y)) <= 1e-9 * (1.0 + np.max(np.abs(y)))
