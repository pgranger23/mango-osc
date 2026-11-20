"""Tests for ``segmentation="chord"`` (mango.earth.chord_segments).

The default ``"radial"`` cut puts a segment boundary at every sub-shell radius,
and every such radius is an angle at which the chord grazes it: an artificial
``sqrt`` cusp in ``P(cz)``.  ``"chord"`` cuts each PREM region's chord interval
into equal pieces instead.  Covered here:

  1. identical to ``"radial"`` at the nadir (detector at the surface), and to
     the chord-interval prototype used as the reference in
     ``benchmarks/fig_nsub_convergence.py``;
  2. unitarity over the full zenith range, with detector depth, atmosphere and
     antineutrinos;
  3. no cusp at the old sub-shell grazing angles (the radial cut, checked on
     the same angles, does have one -- so the test discriminates);
  4. ``P`` and ``dP/dcz`` converge quadratically in ``n_sub``, and agree with
     the continuous-profile ODE backend;
  5. autodiff matches finite differences in ``cz`` at the old cusp angles and in
     the traced geometry / composition parameters;
  6. ``critical_cos_zenith``, the non-unitarity front-end, and argument errors.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import mango
from mango import earth as E, nonunitarity, ode, Flavor
from mango.nonunitarity import NonUnitarity

P0 = mango.nufit_no()


def _pmm(cz, e=6.0, n_sub=4, seg="chord", **kw):
    return mango.probability_earth(P0, e, cz, n_sub=n_sub, segmentation=seg,
                                   flavor_in=Flavor.MU, flavor_out=Flavor.MU, **kw)


def _artificial_angles(n_sub=4, lo=-0.95, hi=-0.3):
    real = E.critical_cos_zenith(segmentation="chord")
    crit = E.critical_cos_zenith(n_sub=n_sub, segmentation="radial")
    return np.array([c for c in crit
                     if lo < c < hi and np.min(np.abs(real - c)) > 1e-9])


# --- 1. identities ------------------------------------------------------------

def test_nadir_identical_to_radial():
    for n in (1, 4, 16):
        a = mango.probability_earth(P0, jnp.array([1.0, 4.0, 11.0]), -1.0, n_sub=n,
                                    h_atm_km=15.0, segmentation="chord")
        b = mango.probability_earth(P0, jnp.array([1.0, 4.0, 11.0]), -1.0, n_sub=n,
                                    h_atm_km=15.0, segmentation="radial")
        assert float(jnp.abs(a - b).max()) < 1e-13, n


def test_segment_lengths_and_count():
    """Same shape as radial, same total length, positive pieces only."""
    tab = E.shell_table(4)
    for cz in (-0.97, -0.84, -0.5, -0.1, 0.3):
        for kw in (dict(), dict(det_depth_km=1.5, h_atm_km=15.0)):
            rc, yc, lc = E.chord_segments(cz, tab, segmentation="chord", **kw)
            rr, yr, lr = E.chord_segments(cz, tab, segmentation="radial", **kw)
            assert lc.shape == lr.shape == (2 * len(tab.outer) + 1,)
            assert abs(float(lc.sum() - lr.sum())) < 1e-8, (cz, kw)
            assert float(lc.min()) >= 0.0
            # every non-empty piece samples the density of its own PREM region,
            # so the path-weighted mean density is close to the radial one
            assert abs(float((rc * lc).sum() - (rr * lr).sum())) <= 2e-3 * float((rr * lr).sum())


def test_matches_reference_prototype():
    """The library cut equals the stand-alone chord-interval construction."""
    r_e = E.R_EARTH_KM
    radii = [0.0] + [hi for hi, _, _ in E._PREM]
    ye_reg = [ye for _, _, ye in E._PREM]
    counts = [max(1, int(round((radii[k + 1] - radii[k]) / (r_e / 40))))
              for k in range(len(ye_reg))]

    def proto(cz, e, h=15.0):
        from mango import constants as C, oscillator as osc
        rmin2 = E._rmin2(r_e, cz)
        d = lambda r: E._d(r**2 - rmin2)
        L, rho, ye = [], [], []
        for k, m in enumerate(counts):
            da, db = d(jnp.asarray(radii[k])), d(jnp.asarray(radii[k + 1]))
            dm = da + (jnp.arange(m) + 0.5) / m * (db - da)
            L.append(jnp.full(m, 1.0) * (db - da) / m)
            rho.append(E.prem_density_jax(jnp.sqrt(rmin2 + dm**2)))
            ye.append(jnp.full(m, ye_reg[k]))
        L, rho, ye = (jnp.concatenate(x)[::-1] for x in (L, rho, ye))
        Ls = jnp.concatenate([(d(r_e + h) - d(jnp.asarray(r_e)))[None], L, L[::-1]])
        rs = jnp.concatenate([jnp.zeros(1), rho, rho[::-1]])
        ys = jnp.concatenate([jnp.full(1, ye_reg[-1]), ye, ye[::-1]])
        v, _ = C.matter_potentials(rs, ys)
        s = osc.propagate_layers(P0.pmns(), P0.msquared(), e * C.GEV_TO_EV, v,
                                 Ls * C.KM_TO_INV_EV, backend="cayley")
        return osc.prob_from_amplitude(s)

    cz = jnp.linspace(-0.995, -0.05, 25)
    a = mango.probability_earth(P0, 6.0, cz, n_sub=4, h_atm_km=15.0, segmentation="chord")
    b = jax.vmap(lambda c: proto(c, 6.0))(cz)
    assert float(jnp.abs(a - b).max()) < 1e-13


# --- 2. unitarity ----------------------------------------------------------------

def test_unitarity_full_range():
    cz = jnp.linspace(-1.0, 1.0, 201)
    e = jnp.array([0.5, 3.0, 20.0])
    for kw in (dict(), dict(det_depth_km=1.5, h_atm_km=15.0),
               dict(det_depth_km=30.0, h_atm_km=15.0, anti=True)):
        p = mango.probability_earth(P0, e, cz, segmentation="chord", **kw)
        assert bool(jnp.isfinite(p).all()), kw
        assert float(jnp.abs(p.sum(-1) - 1.0).max()) < 1e-12, kw
        assert float(jnp.abs(p.sum(-2) - 1.0).max()) < 1e-12, kw


# --- 3. no artificial cusps -------------------------------------------------------

def test_no_cusp_at_subshell_grazing_angles():
    """dP/dcz is continuous across the angles where the radial cut has a cusp."""
    angles = jnp.asarray(_artificial_angles())
    assert angles.size > 10
    d = 1e-6

    def jump(seg):
        g = jax.vmap(jax.grad(lambda c: _pmm(c, seg=seg, h_atm_km=15.0)))
        return np.abs(np.asarray(g(angles + d) - g(angles - d)))

    scale = 30.0                                  # |dP/dcz| peak at 6 GeV
    chord, radial = jump("chord"), jump("radial")
    assert chord.max() < 1e-4 * scale, chord.max()
    # the radial cut does jump there: one-sided slopes ~ 1/sqrt(d)
    assert radial.max() > 0.1 * scale, radial.max()


def test_ad_matches_fd_at_old_cusps():
    h = 1e-6
    for kw in (dict(h_atm_km=15.0), dict(det_depth_km=1.5, h_atm_km=15.0)):
        f = lambda c: _pmm(c, **kw)
        for c in _artificial_angles()[::3]:
            ad = float(jax.grad(f)(jnp.asarray(c)))
            fd = (float(f(c + h)) - float(f(c - h))) / (2 * h)
            assert abs(ad - fd) < 1e-6 * (1.0 + abs(fd)), (c, kw, ad, fd)


# --- 4. convergence -----------------------------------------------------------------

_CZ = jnp.linspace(-0.98, -0.2, 14)
_E = jnp.array([2.0, 5.0, 12.0])


def test_probability_converges_quadratically():
    ref = mango.probability_earth(P0, _E, _CZ, n_sub=128, segmentation="chord")
    err = {n: float(jnp.abs(mango.probability_earth(P0, _E, _CZ, n_sub=n,
                                                    segmentation="chord") - ref).max())
           for n in (16, 32)}
    assert err[32] < 3e-5, err
    assert err[16] / err[32] > 2.5, err          # ~4 for 1/n^2


def test_zenith_derivative_converges_quadratically():
    g = lambda n: jax.vmap(jax.grad(lambda c: _pmm(c, e=5.0, n_sub=n)))(_CZ)
    ref = g(128)
    scale = float(jnp.abs(ref).max())
    err = {n: float(jnp.abs(g(n) - ref).max()) / scale for n in (8, 16)}
    assert err[16] < 1e-4, err
    assert err[8] / err[16] > 2.5, err


def test_agrees_with_continuous_ode():
    """Independent check of the converged answer: the ODE integrates through
    continuous PREM (constant Y_e) with no segmentation at all."""
    ye = 0.4957
    for cz, e in ((-0.95, 3.0), (-0.6, 8.0)):
        po = ode.probability_earth_continuous(P0, e, cz, ye=ye, rtol=1e-10, atol=1e-10)
        pc = mango.probability_earth(P0, e, cz, n_sub=64, segmentation="chord",
                                     ye_core=ye, ye_mantle=ye)
        assert float(jnp.abs(pc - po).max()) < 5e-6, (cz, e)


# --- 5. traced parameters -------------------------------------------------------------

@pytest.mark.parametrize("name,fn,x0", [
    ("ye_core", lambda y: mango.probability_earth(
        P0, 6.0, -0.9, segmentation="chord", ye_core=y)[1, 0], 0.4656),
    ("ye_mantle", lambda y: mango.probability_earth(
        P0, 6.0, -0.6, segmentation="chord", ye_mantle=y)[1, 0], 0.4957),
    ("h_atm_km", lambda h: mango.probability_earth(
        P0, 1.0, 0.1, segmentation="chord", h_atm_km=h)[1, 1], 15.0),
    ("det_depth_km up", lambda h: mango.probability_earth(
        P0, 6.0, -0.5, segmentation="chord", det_depth_km=h)[1, 1], 1.5),
    ("det_depth_km down", lambda h: mango.probability_earth(
        P0, 0.5, 0.3, segmentation="chord", det_depth_km=h, h_atm_km=15.0)[1, 1], 1.5),
])
def test_parameter_gradients(name, fn, x0):
    ad = float(jax.grad(fn)(x0))
    h = 1e-5 * max(1.0, abs(x0))
    fd = (float(fn(x0 + h)) - float(fn(x0 - h))) / (2 * h)
    assert abs(ad - fd) < 1e-6 * (1.0 + abs(fd)), (name, ad, fd)


# --- 6. helpers, other front-ends, errors -----------------------------------------------

def test_critical_cos_zenith_chord_is_prem_boundaries_only():
    c = E.critical_cos_zenith(segmentation="chord")
    radii = np.array([hi for hi, _, _ in E._PREM[:-1]])
    assert np.allclose(c, np.sort(-np.sqrt(1 - (radii / E.R_EARTH_KM) ** 2)))
    # independent of n_sub; a strict subset of the radial angles
    assert np.array_equal(c, E.critical_cos_zenith(n_sub=16, segmentation="chord"))
    radial = E.critical_cos_zenith(n_sub=4, segmentation="radial")
    assert np.array_equal(c, E.critical_cos_zenith())       # chord is the default
    assert all(np.min(np.abs(radial - x)) < 1e-12 for x in c)
    assert len(radial) > 4 * len(c)


def test_nonunitarity_front_end_alpha_zero():
    nu0 = NonUnitarity()
    cz = jnp.linspace(-0.97, -0.2, 9)
    a = nonunitarity.probability_earth(P0, nu0, 6.0, cz, h_atm_km=15.0,
                                       segmentation="chord")
    b = mango.probability_earth(P0, 6.0, cz, h_atm_km=15.0, segmentation="chord")
    assert float(jnp.abs(a - b).max()) < 1e-12


def test_default_segmentation():
    """Default: chord on the PREM path, the model's own shells for a LayeredEarth."""
    cz = jnp.linspace(-0.97, 0.3, 11)
    kw = dict(h_atm_km=15.0)
    d = mango.probability_earth(P0, 6.0, cz, **kw)
    assert float(jnp.abs(d - mango.probability_earth(
        P0, 6.0, cz, segmentation="chord", **kw)).max()) == 0.0
    assert float(jnp.abs(d - mango.probability_earth(
        P0, 6.0, cz, segmentation="radial", **kw)).max()) > 1e-6
    nu0 = NonUnitarity()
    assert float(jnp.abs(nonunitarity.probability_earth(P0, nu0, 6.0, cz, **kw)
                         - d).max()) < 1e-12
    model = mango.prem_layered(2)
    for front in (lambda **k: mango.probability_earth(P0, 6.0, cz, **kw, **k),
                  lambda **k: nonunitarity.probability_earth(P0, nu0, 6.0, cz, **kw, **k)):
        a = front(earth_model=model)
        b = front(earth_model=model, segmentation="radial")
        assert float(jnp.abs(a - b).max()) == 0.0
    tab = E.shell_table(4)
    for c in (-0.8, 0.2):
        assert all(np.array_equal(x, y) for x, y in zip(
            E.chord_segments(c, tab, h_atm_km=15.0),
            E.chord_segments(c, tab, h_atm_km=15.0, segmentation="chord")))


def test_errors():
    with pytest.raises(ValueError, match="segmentation"):
        mango.probability_earth(P0, 6.0, -0.5, segmentation="shells")
    with pytest.raises(ValueError, match="LayeredEarth"):
        mango.probability_earth(P0, 6.0, -0.5, earth_model=mango.prem_layered(2),
                                segmentation="chord")
    with pytest.raises(ValueError, match="LayeredEarth"):
        nonunitarity.probability_earth(P0, NonUnitarity(), 6.0, -0.5,
                                       earth_model=mango.prem_layered(2),
                                       segmentation="chord")
    # a hand-built table whose shells straddle a PREM discontinuity
    bad = E.ShellTable(outer=np.array([2000.0, E.R_EARTH_KM]), inner=np.array([0.0, 2000.0]),
                       rho=np.array([12.0, 4.0]), ye=jnp.array([0.4656, 0.4957]))
    with pytest.raises(ValueError, match="region boundary"):
        E.chord_segments(-0.5, bad, segmentation="chord")
