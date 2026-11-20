"""PREM Earth model and chord geometry.

Implements the Preliminary Reference Earth Model (Dziewonski & Anderson 1981):
density as piecewise polynomials in ``x = r / R_earth``.  A neutrino arriving with
``cos(zenith) = cz`` (up-going => ``cz < 0``) traverses a chord; we split it into
constant-density segments and hand them to the layer propagator.  Within each
PREM region the chord is cut either into equal-length pieces of the region's
chord interval (``segmentation="chord"``, the default since v0.3.0, cusp-free in
``cz`` apart from the real PREM discontinuities) or at radial sub-shell
boundaries (``segmentation="radial"``, the v0.2.x default); see
:func:`chord_segments`.

Design choice for ``jit`` / ``vmap``: the shell radii and per-shell densities are
*static* (built from numpy, independent of ``cz``).  Only the per-segment path
**lengths** depend on ``cz``, computed with clamps so the array shape is fixed --
shells that the chord does not reach simply get zero length (identity
propagators).  Lengths are smooth in ``cz`` (=> differentiable in zenith angle).
"""

from __future__ import annotations

import dataclasses
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

from .constants import R_EARTH_KM

# PREM regions: (outer_radius_km, density polynomial coeffs in x=r/R_E, Y_e).
# Y_e ~ 0.4656 in the (iron) core, ~0.4957 in the mantle/crust.
_PREM = [
    (1221.5, [13.0885, 0.0, -8.8381, 0.0], 0.4656),
    (3480.0, [12.5815, -1.2638, -3.6426, -5.5281], 0.4656),
    (5701.0, [7.9565, -6.4761, 5.5283, -3.0807], 0.4957),
    (5771.0, [5.3197, -1.4836, 0.0, 0.0], 0.4957),
    (5971.0, [11.2494, -8.0298, 0.0, 0.0], 0.4957),
    (6151.0, [7.1089, -3.8045, 0.0, 0.0], 0.4957),
    (6346.6, [2.6910, 0.6924, 0.0, 0.0], 0.4957),
    (6356.0, [2.9000, 0.0, 0.0, 0.0], 0.4957),
    (6368.0, [2.6000, 0.0, 0.0, 0.0], 0.4957),
    (R_EARTH_KM, [1.0200, 0.0, 0.0, 0.0], 0.4957),
]

ShellTable = namedtuple("ShellTable", ["outer", "inner", "rho", "ye"])

# Core / mantle boundary (outer core radius) — sets where Y_e switches.
CORE_RADIUS_KM = 3480.0
# Default electron fractions (PREM-standard: iron core vs silicate mantle).
YE_CORE_DEFAULT = 0.4656
YE_MANTLE_DEFAULT = 0.4957


def prem_density(r_km):
    """PREM density (g/cm^3) at radius ``r_km`` (scalar or array)."""
    r = np.atleast_1d(np.asarray(r_km, dtype=float))
    x = r / R_EARTH_KM
    rho = np.zeros_like(r)
    lower = 0.0
    filled = np.zeros_like(r, dtype=bool)
    for outer, coeffs, _ye in _PREM:
        mask = (~filled) & (r <= outer + 1e-9)
        val = sum(c * x**i for i, c in enumerate(coeffs))
        rho = np.where(mask, val, rho)
        filled |= mask
        lower = outer
    rho = np.where(filled, rho, 0.0)  # outside Earth
    return rho if np.ndim(r_km) else float(rho[0])


def shell_table(n_sub: int = 4, ye_core: float = YE_CORE_DEFAULT,
                ye_mantle: float = YE_MANTLE_DEFAULT) -> ShellTable:
    """Table of constant-density shells: static radii/density, traceable Y_e.

    PREM region boundaries (density discontinuities) are always respected; within
    each region the number of equal-radius sub-shells is allocated **proportional
    to the region's radial thickness** (target step ``R_E / (10*n_sub)``, at least
    one per region).  This puts resolution in the thick core/mantle -- where the
    oscillation phase accumulates -- instead of wasting it on thin crust layers,
    converging much faster in ``n_sub`` than uniform subdivision.

    Density is a placeholder shell mid-radius value (``chord_segments`` resamples
    it at the path midpoint).  The electron fraction is two-zone: ``ye_core``
    inside ``CORE_RADIUS_KM`` (3480 km), ``ye_mantle`` outside (3480 km is a region
    boundary, so no shell straddles it).

    ``n_sub`` fixes the shell **count**, and the loop below only ever compares
    ``mid`` (a concrete, numpy-derived radius) against ``CORE_RADIUS_KM`` to pick
    which zone a shell falls in -- so radii and shape stay static/concrete even
    if this is traced through ``jax.jit``/``jax.grad``.  ``ye_core``/``ye_mantle``
    themselves may be JAX tracers (e.g. under ``jax.grad(..., argnums=...)`` for
    ``dP/dY_e``): they only ever flow into the *values* placed in ``ye`` below, so
    that column is built as a ``jnp`` array (not ``np.array``, which would force a
    tracer through ``__array__`` and raise) while ``outer``/``inner``/``rho`` --
    built from an independent numpy expression -- stay plain numpy.
    """
    dr_target = R_EARTH_KM / (10 * n_sub)
    outer, inner, rho, ye = [], [], [], []
    lo = 0.0
    for hi, coeffs, _ye_region in _PREM:
        n_k = max(1, int(round((hi - lo) / dr_target)))
        edges = np.linspace(lo, hi, n_k + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (a + b)
            x = mid / R_EARTH_KM
            inner.append(a)
            outer.append(b)
            rho.append(sum(c * x**i for i, c in enumerate(coeffs)))
            ye.append(ye_core if mid < CORE_RADIUS_KM else ye_mantle)
        lo = hi
    return ShellTable(
        outer=np.array(outer),
        inner=np.array(inner),
        rho=np.array(rho),
        ye=jnp.array(ye),
    )


_SQRT_FLOOR = 1e-9  # km^2


def _d(r2_minus_rmin2):
    """Distance from closest approach: ``sqrt(max(r^2 - r_min^2, 0))``.

    Gradient-safe: at the chord turning point ``r -> r_min`` the argument hits 0
    where ``d(sqrt)/dx = 1/(2 sqrt(x))`` diverges (the geometric vertical
    tangent), which would feed NaN into ``dP/dcz``.  Below the floor we return 0
    with zero gradient; the forward value is unchanged.
    """
    x = jnp.clip(r2_minus_rmin2, 0.0, None)
    big = x > _SQRT_FLOOR
    x_safe = jnp.where(big, x, 1.0)
    return jnp.where(big, jnp.sqrt(x_safe), 0.0)


def _rmin2(r_det, cz):
    """Closest-approach radius squared: ``r_det^2 * sin^2(zenith)``.

    Gradient-safe at the poles: ``s2 = 1 - cz**2`` is mathematically >= 0 for
    ``cz`` in ``[-1, 1]`` and only dips below zero by floating-point roundoff
    right at ``cz = +-1``.  A plain ``jnp.clip(s2, 0.0, None)`` lowers to
    ``jnp.maximum(s2, 0.0)``, whose JAX gradient rule splits the tie 50/50
    exactly where ``s2 == 0.0`` -- i.e. exactly at the poles -- halving
    ``d(rmin2)/dcz`` there (and everything downstream: baselines, chord
    lengths, ``dP/dcz``).  Selecting with ``>=`` instead keeps the gradient on
    the smooth (physical) branch straight through the tie; the forward value
    is unchanged.
    """
    s2 = 1.0 - cz**2
    return r_det**2 * jnp.where(s2 >= 0.0, s2, 0.0)


SEGMENTATIONS = ("radial", "chord")
DEFAULT_SEGMENTATION = "chord"


def _check_segmentation(segmentation):
    if segmentation not in SEGMENTATIONS:
        raise ValueError(f"segmentation must be one of {SEGMENTATIONS}, "
                         f"got {segmentation!r}")


def resolve_segmentation(segmentation, earth_model=None):
    """Resolve the ``segmentation`` argument of the ``probability_earth`` front ends.

    ``None`` (the default) means ``DEFAULT_SEGMENTATION`` (``"chord"``) on the PREM
    polynomial path and the model's own shells for a :class:`LayeredEarth`, which
    is piecewise-constant by construction (returned as ``None``).  An explicit
    ``"radial"`` with a ``LayeredEarth`` is accepted (its shells *are* radial
    cuts); ``"chord"`` with one raises.
    """
    if segmentation is not None:
        _check_segmentation(segmentation)
    if earth_model is None:
        return DEFAULT_SEGMENTATION if segmentation is None else segmentation
    if segmentation == "chord":
        raise ValueError("segmentation='chord' applies to the PREM polynomial "
                         "profile; a LayeredEarth is piecewise-constant by "
                         "construction, so its shell boundaries are the model")
    return None


def _chord_pieces(table: ShellTable):
    """Static per-shell bookkeeping for ``segmentation="chord"``.

    For every shell of ``table`` (ascending radius) returns the inner/outer radius
    of the PREM region that contains it and the shell's fractional position
    ``[j/m, (j+1)/m]`` among the ``m`` shells of that region.  Raises if a PREM
    region boundary is not a shell boundary (a hand-built table that straddles a
    density discontinuity has no well-defined region chord).
    """
    inner = np.asarray(table.inner, dtype=float)
    outer = np.asarray(table.outer, dtype=float)
    region_hi = np.array([hi for hi, _, _ in _PREM])
    region_lo = np.concatenate([[0.0], region_hi[:-1]])
    edges = np.concatenate([inner, outer[-1:]])
    for b in region_hi:
        if not np.any(np.isclose(edges, b, rtol=0.0, atol=1e-6)):
            raise ValueError(f"segmentation='chord' needs every PREM region boundary "
                             f"to be a shell boundary; {b} km is not")
    k = np.searchsorted(region_hi, 0.5 * (inner + outer))   # region of each shell
    m = np.bincount(k, minlength=len(region_hi))[k]
    j = np.concatenate([np.arange(c) for c in np.bincount(k, minlength=len(region_hi))])
    return region_lo[k], region_hi[k], j / m, (j + 1) / m


def _leg_pieces(rmin2, leg_lo, leg_hi, r_a, r_b, f_lo, f_hi):
    """Equal-``d`` pieces of each PREM region's chord interval on one leg.

    The region's interval ``[d(r_a), d(r_b)]`` is clamped to the leg
    ``[leg_lo, leg_hi]`` and split at the static fractions ``f_lo``/``f_hi``.
    Piece ends are linear in the two clamped region ends, so they move smoothly
    with ``cz``; a piece can only appear when the chord grazes a PREM region
    boundary (where the density really is discontinuous).
    """
    a = jnp.clip(_d(r_a**2 - rmin2), leg_lo, leg_hi)
    b = jnp.clip(_d(r_b**2 - rmin2), leg_lo, leg_hi)
    lo = a + f_lo * (b - a)
    hi = a + f_hi * (b - a)
    return lo, hi


def chord_segments(cz, table: ShellTable, h_atm_km=0.0, det_depth_km=0.0,
                   ye_core=YE_CORE_DEFAULT, ye_mantle=YE_MANTLE_DEFAULT,
                   segmentation=DEFAULT_SEGMENTATION):
    """Path segments along the neutrino trajectory for ``cos(zenith) = cz``.

    Unified geometry covering the full range of ``cz``: up-going through the Earth,
    down-going, and the optional atmospheric production height ``h_atm_km``
    (neutrinos produced at ``R_E + h_atm``, giving a vacuum baseline for
    down-going / near-horizon directions). PREM region boundaries are respected
    exactly (no density smearing). Returns ``(rho, ye, length_km)`` of fixed length
    ``2 * n_shells + 1`` ordered source -> detector.

    ``segmentation`` chooses how the chord is cut within each PREM region:

    * ``"radial"`` (the v0.2.x default): at every sub-shell boundary of ``table``.  Each
      sub-shell boundary is a radius the chord can graze, which puts an
      artificial ``sqrt`` cusp in ``P(cz)`` at every such angle (see
      :func:`critical_cos_zenith`); pointwise ``dP/dcz`` therefore converges
      slowly in ``n_sub``.
    * ``"chord"`` (default): the chord interval of each PREM region is cut into as many
      equal-length pieces as ``table`` has sub-shells in that region (same
      segment count and cost).  Piece ends slide smoothly with ``cz``, so the only
      cusps left are those at the real PREM discontinuities, and ``P`` and
      ``dP/dcz`` converge as ``1/n_sub^2``.  For a detector at the surface it
      is identical to ``"radial"`` at the nadir (``cz = -1``), where the two
      cuts coincide.

    Differentiable in ``cz`` *and* in ``ye_core`` / ``ye_mantle`` (the two-zone
    electron fraction, boundary at ``CORE_RADIUS_KM``): these are read here as
    (possibly traced) JAX values, so ``d/d(ye_core)`` flows through -- e.g. for a
    core electron-density oscillograd. (PREM mass density and shell-boundary radii
    are still static constants.)

    Geometry uses the distance-from-closest-approach coordinate
    ``d(r) = sqrt(r^2 - r_min^2)`` with ``r_min = r_det*sqrt(1-cz^2)``. The
    descending leg (production side) spans ``d in [d_low, s_prod]`` and, for
    up-going only, an ascending leg spans ``d in [0, |s_det|]``; one vacuum segment
    carries the part above ``R_E``.
    """
    _check_segmentation(segmentation)
    r_det = R_EARTH_KM - det_depth_km
    r_prod = R_EARTH_KM + h_atm_km
    outer = jnp.asarray(table.outer)
    inner = jnp.asarray(table.inner)

    cz = jnp.asarray(cz, dtype=jnp.float64)
    upgoing = cz < 0.0
    rmin2 = _rmin2(r_det, cz)
    s_prod = _d(r_prod**2 - rmin2)            # production point (d-coordinate)
    s_det = r_det * cz                        # detector: + down-going, - up-going
    d_low = jnp.where(upgoing, 0.0, s_det)    # bottom of the descending leg
    sdet_abs = jnp.abs(s_det)

    def ye_at(r):  # two-zone electron fraction (traced in ye_core/ye_mantle)
        return jnp.where(r < CORE_RADIUS_KM, ye_core, ye_mantle)

    # Per-leg clamped d-ranges -> segment lengths and path-midpoint radii.  Sampling
    # density at the *path* midpoint (not the shell's geometric midpoint) makes the
    # constant-per-segment approximation converge far faster in n_sub (NuFast-style).
    if segmentation == "radial":
        d_out = _d(outer**2 - rmin2)
        d_in = _d(inner**2 - rmin2)
        d_hi_desc = jnp.clip(d_out, d_low, s_prod)
        d_lo_desc = jnp.clip(d_in, d_low, s_prod)
        d_hi_asc = jnp.clip(d_out, 0.0, sdet_abs)
        d_lo_asc = jnp.clip(d_in, 0.0, sdet_abs)
    else:
        r_a, r_b, f_lo, f_hi = _chord_pieces(table)
        d_lo_desc, d_hi_desc = _leg_pieces(rmin2, d_low, s_prod, r_a, r_b, f_lo, f_hi)
        d_lo_asc, d_hi_asc = _leg_pieces(rmin2, 0.0, sdet_abs, r_a, r_b, f_lo, f_hi)

    len_desc = d_hi_desc - d_lo_desc
    r_desc = jnp.sqrt(rmin2 + (0.5 * (d_hi_desc + d_lo_desc)) ** 2)
    rho_desc = prem_density_jax(r_desc)

    len_asc = jnp.where(upgoing, d_hi_asc - d_lo_asc, 0.0)
    r_asc = jnp.sqrt(rmin2 + (0.5 * (d_hi_asc + d_lo_asc)) ** 2)
    rho_asc = prem_density_jax(r_asc)

    # Atmospheric vacuum segment: descending part with radius in [R_E, r_prod].
    d_re = _d(R_EARTH_KM**2 - rmin2)
    len_atm = s_prod - jnp.clip(d_re, d_low, s_prod)

    # Source -> detector: atmosphere, descending (outer->inner), ascending.
    one = jnp.ones((1,))
    lengths = jnp.concatenate([len_atm[None], len_desc[::-1], len_asc])
    rho_seg = jnp.concatenate([0.0 * one, rho_desc[::-1], rho_asc])
    ye_seg = jnp.concatenate([ye_mantle * one, ye_at(r_desc)[::-1], ye_at(r_asc)])
    return rho_seg, ye_seg, lengths


def earth_segments(cz, table: ShellTable, det_depth_km=0.0,
                   segmentation=DEFAULT_SEGMENTATION):
    """Earth-only path segments (``h_atm_km = 0``); see :func:`chord_segments`."""
    return chord_segments(cz, table, h_atm_km=0.0, det_depth_km=det_depth_km,
                          segmentation=segmentation)


# --- fully parametric, differentiable layered Earth --------------------------

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LayeredEarth:
    """A constant-density-shell Earth whose parameters are differentiable.

    ``outer`` : ascending outer radii of the shells (km), ``outer[-1]`` = surface.
    ``density``: mass density of each shell (g/cm^3).
    ``ye``    : electron fraction of each shell.

    All three are JAX arrays (a PyTree), so probabilities are differentiable w.r.t.
    the **shell boundary positions** *and* the **shell densities** / Y_e -- not just
    the oscillation parameters. Build one from PREM with :func:`prem_layered`.
    """

    outer: jax.Array
    density: jax.Array
    ye: jax.Array


def prem_layered(n_sub: int = 2, ye_core: float = YE_CORE_DEFAULT,
                 ye_mantle: float = YE_MANTLE_DEFAULT) -> LayeredEarth:
    """A :class:`LayeredEarth` sampling PREM into constant-density shells.

    Uses the same thickness-proportional shell placement as :func:`shell_table`;
    each shell's density is PREM at its mid radius. The result is a differentiable
    approximation to PREM (exact as ``n_sub -> infinity``).
    """
    tab = shell_table(n_sub, ye_core=ye_core, ye_mantle=ye_mantle)
    return LayeredEarth(outer=jnp.asarray(tab.outer),
                        density=jnp.asarray(tab.rho),
                        ye=jnp.asarray(tab.ye))


def layered_chord_segments(model: LayeredEarth, cz, h_atm_km=0.0, det_depth_km=0.0):
    """Path segments through a :class:`LayeredEarth` (cf. :func:`chord_segments`).

    Differentiable in ``cz`` *and* in ``model.outer`` (boundary positions),
    ``model.density`` and ``model.ye``. The surface radius is ``model.outer[-1]``.
    Returns ``(rho, ye, length_km)`` of fixed length ``2 * n_shells + 1`` ordered
    source -> detector.
    """
    outer = model.outer
    inner = jnp.concatenate([jnp.zeros(1, outer.dtype), outer[:-1]])
    density, ye = model.density, model.ye
    r_e = outer[-1]
    r_det = r_e - det_depth_km
    r_prod = r_e + h_atm_km

    cz = jnp.asarray(cz, dtype=jnp.float64)
    upgoing = cz < 0.0
    rmin2 = _rmin2(r_det, cz)   # see _rmin2: clip would halve d/dcz at cz = +-1
    s_prod = _d(r_prod**2 - rmin2)
    s_det = r_det * cz
    d_low = jnp.where(upgoing, 0.0, s_det)

    d_out = _d(outer**2 - rmin2)
    d_in = _d(inner**2 - rmin2)
    sdet_abs = jnp.abs(s_det)

    len_desc = jnp.clip(d_out, d_low, s_prod) - jnp.clip(d_in, d_low, s_prod)
    len_asc = jnp.where(
        upgoing, jnp.clip(d_out, 0.0, sdet_abs) - jnp.clip(d_in, 0.0, sdet_abs), 0.0)
    d_re = _d(r_e**2 - rmin2)
    len_atm = s_prod - jnp.clip(d_re, d_low, s_prod)

    one = jnp.ones((1,))
    lengths = jnp.concatenate([len_atm[None], len_desc[::-1], len_asc])
    rho_seg = jnp.concatenate([0.0 * one, density[::-1], density])
    ye_seg = jnp.concatenate([ye[-1] * one, ye[::-1], ye])
    return rho_seg, ye_seg, lengths


def baseline_km(cz, det_depth_km=0.0):
    """Total in-Earth chord length (km) for ``cos(zenith) = cz``."""
    r_det = R_EARTH_KM - det_depth_km
    cz = jnp.asarray(cz, dtype=jnp.float64)
    r_min2 = _rmin2(r_det, cz)
    # Use the guarded _d() rather than a bare sqrt(clip(...)): at the horizon
    # (cz = 0) the ascending-leg argument r_det^2 - r_min2 is exactly 0, and
    # sqrt'(0) = inf.  The outer jnp.where hides that in forward mode but
    # reverse mode still pulls the NaN back through the unselected branch, so
    # jax.grad(baseline_km)(0.0) returned NaN while jacfwd returned 0.
    desc = _d(R_EARTH_KM**2 - r_min2)
    asc = _d(r_det**2 - r_min2)
    return jnp.where(cz < 0.0, desc + asc, 0.0)


def critical_cos_zenith(table: ShellTable | None = None, det_depth_km: float = 0.0,
                        n_sub: int = 4,
                        segmentation: str = DEFAULT_SEGMENTATION) -> np.ndarray:
    """Cos(zenith) values at which the chord grazes a shell boundary.

    At these angles the turning-point radius ``r_min(cz)`` exactly equals a
    boundary radius ``r_b``, so the layer the chord is entering contributes a
    length proportional to ``sqrt(r_b^2 - r_min^2)``: a sqrt-cusp in ``cz``
    where ``dP/dcz`` diverges like ``1/sqrt(|cz - cz_crit|)``.

    Two kinds of boundary produce one.  At a PREM region boundary the density
    really is discontinuous and continuous PREM has the same cusp (physical).
    A sub-shell boundary inside a region is an artefact of the
    ``segmentation="radial"`` discretisation: continuous PREM has no feature
    there, the cusp shrinks as ``n_sub`` grows but their number grows with it.
    With ``segmentation="chord"`` (the default; see :func:`chord_segments`) only
    the PREM region boundaries remain, and that is what is returned; pass
    ``segmentation="radial"`` for the sub-shell grazing angles as well.

    Why you would call this: gradient-based optimizers that move cos(zenith)
    bin edges continuously (e.g. Fisher-optimized binning) should avoid
    parking an edge exactly on one of these angles -- the local gradient
    there is unbounded/ill-conditioned even though ``P`` itself is finite and
    well-defined, so a solver can get stuck or take pathological steps.

    Returns the sorted, unique cos(zenith) (up-going, ``cz < 0``) for every
    boundary radius ``r_b < r_det = R_EARTH_KM - det_depth_km``, as a plain
    numpy array -- this is a diagnostic helper, not a differentiable path.
    ``table`` defaults to ``shell_table(n_sub)`` (boundary radii do not depend
    on ``ye_core``/``ye_mantle``); it is ignored for ``segmentation="chord"``.
    """
    _check_segmentation(segmentation)
    r_det = R_EARTH_KM - det_depth_km
    if segmentation == "chord":
        radii = np.array([hi for hi, _, _ in _PREM])
    else:
        if table is None:
            table = shell_table(n_sub)
        radii = np.unique(np.concatenate(
            [np.asarray(table.outer), np.asarray(table.inner)]))
    radii = radii[(radii > 0.0) & (radii < r_det)]
    return np.sort(-np.sqrt(1.0 - (radii / r_det) ** 2))


def prem_density_jax(r_km):
    """JAX-traceable PREM density (g/cm^3) at radius ``r_km`` (0 above R_E)."""
    r = jnp.asarray(r_km, dtype=jnp.float64)
    x = r / R_EARTH_KM
    rho = jnp.zeros_like(r)
    prev = -1.0
    for outer, coeffs, _ye in _PREM:
        val = sum(c * x**i for i, c in enumerate(coeffs))
        in_region = (r > prev) & (r <= outer)
        rho = jnp.where(in_region, val, rho)
        prev = outer
    return rho


