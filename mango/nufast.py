"""NuFast-style direct constant-density probability (3-flavor, standard).

Ports the algorithm of Parke & Denton's NuFast-LBL (the DMP "Rosetta" relations,
Denton & Parke, Phys. Rev. D 110, 073005 (2024), arXiv:2405.02400): the matter
mixing-matrix magnitudes ``|U^m_{alpha i}|^2`` and the matter Jarlskog are obtained
*analytically* from the matter eigenvalues, and plugged straight into the standard
``sin^2`` oscillation formula.  No eigenvectors, no complex matrix exponential, no
matmuls -- just scalar algebra, so it is several times faster than the
matrix-exponential backends while remaining exact and differentiable.

Single constant-density layer only (it returns probabilities, not the evolution
operator, so it cannot be chained across layers) and standard 3-flavor (no NSI /
steriles).  Use the matrix-exponential backends for those.

Provenance / attribution
-------------------------
This module is a close, line-by-line **JAX transcription of the reference C++
implementation** that Peter Denton and Stephen Parke publish alongside the paper
above as NuFast-LBL (https://github.com/PeterDenton/NuFast-LBL) -- it is not an
independent re-derivation from the published formulas alone. The intermediate
variable names below (``See``, ``Smm``, ``Tee``, ``Tmm``, ``Amatter``, ``Jrr``,
``Jmatter``, ``lambda2``/``lambda3``, ``Dl21``/``Dl32``/``Dl31``, ``rootAsqB``,
``ss0``, ``PiInv``, ``Xp2``/``Xp3``, ``D21``/``D32``, ``Pme_TC``/``Pme_TV``, ...)
track the original C++ source's naming (``Ue2sq``, ``Amatter``, ``See``, ``Smm``,
``Tee``, ``Tmm``, ``lambda2``/``lambda3``, ``Dlambda21``/``Dlambda31``/
``Dlambda32``, ``rootAsqB``, ``ss0``, ``Jrr``, ``Jmatter``, ``Xp2``/``Xp3``,
``PiDlambdaInv``, ``D21``/``D32``, ``triple_sin``, ``Pme_TC``/``Pme_TV``) closely
enough that this is best described honestly as a port/transliteration into JAX,
not a clean-room reimplementation from the paper's equations with independent
structure.

NuFast-LBL is itself released under the MIT License (Copyright (c) 2024 Peter B.
Denton), so this port is used under the terms of that license; the original
authors' copyright is credited here and in this repository's top-level
``README.md``. If you use this backend, please also cite Denton & Parke, Phys.
Rev. D 110, 073005 (2024) [arXiv:2405.02400], as the original authors request.

Performance structure
---------------------
The algebra is split into an energy-independent part (:func:`_vacuum_invariants`,
evaluated once per parameter point) and a purely elementwise per-energy part
(:func:`_channels`), so it can be evaluated directly on arrays of energies.
Two things matter for speed under XLA (measured on CPU with jax 0.12):

* XLA treats ``sin``/``cos`` as "cheap" and freely *duplicates* them into every
  fusion that consumes them -- for the 3x3 output (whose 9 entries are stacked by
  a concatenate that evaluates its operands once per output element) and,
  much worse, into each of the many reverse-mode fusions.  A float64 sine
  costs ~10 ns on CPU, so this duplication (together with the stacking layout,
  see :func:`prob_matrix`) made the probability matrix ~5x and its gradient
  ~2.5x slower than necessary (1e5 energies, one core).  The per-energy
  ``sin``/``cos`` are therefore evaluated through :func:`_sin`/:func:`_cos`,
  which pin their value (and, under differentiation, the derivative
  ``cos``/``-sin``) behind an ``optimization_barrier``, i.e. compute each
  exactly once into a buffer.  The barrier changes no value, only XLA's fusion
  decisions; gradients, forward mode and higher derivatives are unaffected
  (custom JVP, not custom VJP).  On CPU the ``sin``/``cos`` (and the
  ``arccos`` of the cubic) themselves come from :mod:`mango._small`, plain-
  arithmetic fdlibm formulations that XLA vectorises (~3 ns for a sin/cos
  pair instead of ~25 ns of scalar libm calls; within ~1 ulp of them).
* :func:`event_probability` returns only the one channel each event needs
  (elementwise select), with per-event neutrino/antineutrino sign, so an event
  sample is weighted in one elementwise pass with no ``(N, 3, 3)``
  intermediate, gather/scatter or concatenate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from . import constants as C
from ._small import arccos, sincos

_TWO_PI = 2.0 * jnp.pi

# Relative floor on the 1-2 matter eigenvalue gap, in units of |dm31|.
# The Rosetta relations give |U^m_{alpha 2}|^2 ~ 1/Dl21, which diverges when the
# two lower matter eigenstates become exactly degenerate (dm21 -> 0).  The
# probability itself stays finite because those magnitudes always appear
# multiplied by s21 = 2 sin^2(Dl21 L/4E) ~ Dl21^2, so the divergence cancels
# analytically -- but in floating point 1/0 * 0 is NaN, and dm21 = 0 is a
# perfectly reasonable thing to ask for (vacuum limits, parameter scans that
# cross zero).  Flooring Dl21 evaluates the exact probability for a system split
# by 1e-12 |dm31| instead of 0; since P is continuous in dm21 the induced error
# is O(floor * L/4E), i.e. ~1e-11 rad of phase for a 1300 km baseline.
_DL21_FLOOR_REL = 1e-12


def _safe_sqrt(x):
    """``sqrt(x)`` with a finite gradient at ``x = 0`` (double-where guard)."""
    big = x > 0.0
    return jnp.where(big, jnp.sqrt(jnp.where(big, x, 1.0)), 0.0)


# --- transcendentals evaluated exactly once ----------------------------------
#
# XLA's fusion heuristics consider sin/cos cheap and duplicate them into every
# consumer fusion instead of storing them; for this kernel that means each
# per-energy sine was recomputed ~9x in the forward 3x3 matrix and ~10-20x in the
# reverse pass (where cos, its derivative, lands in a dozen separate fusions).
# These wrappers pin the value -- and, under differentiation, the value together
# with its derivative, computed in the same fusion so sin and cos share their
# argument reduction (mango._small.sincos, pinned as one buffer) -- behind an
# ``optimization_barrier``, so each is materialised once.  Only the *primal* values go through the barrier: the
# tangent ``c * t`` (and hence the transposed cotangent) stays unbarriered, so XLA
# keeps fusing the linear part freely (barriers on cotangents measured slower).
# A custom JVP, not a custom VJP, so forward mode and higher-order derivatives
# keep working (the rule is itself differentiable).  Values are unchanged.


@jax.custom_jvp
def _sin(x):
    """``sin(x)``, materialised once (see above)."""
    return lax.optimization_barrier(sincos(x)[0])


@_sin.defjvp
def _sin_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    s, c = sincos(x, pinned=True)
    return s, c * t


@jax.custom_jvp
def _cos(x):
    """``cos(x)``, materialised once (see above)."""
    return lax.optimization_barrier(sincos(x)[1])


@_cos.defjvp
def _cos_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    s, c = sincos(x, pinned=True)
    return c, -s * t


def _vacuum_invariants(params, anti):
    """Energy-independent ("vacuum") pieces of the NuFast algebra.

    Computed once per parameter point and broadcast into the per-energy part
    (:func:`_channels`).  ``anti`` (a Python bool) flips the CP phase; the sign
    of the matter potential is applied per energy in :func:`_channels`.
    """
    s12sq = jnp.sin(params.theta12) ** 2
    s13sq = jnp.sin(params.theta13) ** 2
    s23sq = jnp.sin(params.theta23) ** 2
    delta = params.deltacp
    dm21 = params.dm21
    dm31 = params.dm31
    if anti:
        delta = -delta

    c13sq = 1 - s13sq
    Ue2sq_t = c13sq * s12sq
    Ue3sq_t = s13sq
    Um3sq_t = c13sq * s23sq
    Ut2sq_t = s13sq * s12sq * s23sq
    Um2sq_t = (1 - s12sq) * (1 - s23sq)
    # Jrr = sqrt(Um2sq_t * Ut2sq_t) factorises as sqrt(...) * |sin(theta13)|.
    # Written that way the sqrt no longer sits on an argument that vanishes
    # linearly in s13sq ~ theta13^2, which used to make d/dtheta13 NaN at
    # theta13 = 0 (a standard null point).  sin(theta13) is used unsigned-safe:
    # for the physical range theta13 in [0, pi/2] it equals |sin(theta13)|, and
    # taking it signed makes the derivative at theta13 = 0 the correct one-sided
    # limit rather than a subgradient.
    s13 = jnp.sin(params.theta13)
    Jrr = _safe_sqrt(Um2sq_t * s12sq * s23sq) * s13
    sind, cosd = jnp.sin(delta), jnp.cos(delta)
    Um2sq_t = Um2sq_t + Ut2sq_t - 2 * Jrr * cosd
    Jmatter_t = 8 * Jrr * c13sq * sind
    A_tmp = dm21 + dm31
    See = A_tmp - dm21 * Ue2sq_t - dm31 * Ue3sq_t
    Tmm_tmp = dm21 * dm31
    Tee = Tmm_tmp * (1 - Ue3sq_t - Ue2sq_t)
    return dict(
        dm21=dm21, dm31=dm31, A_tmp=A_tmp, See=See, Tee=Tee, Tmm_tmp=Tmm_tmp,
        Um2sq_t=Um2sq_t, Um3sq_t=Um3sq_t,
        # energy-independent prefactors of Tmm and Jmatter below, grouped
        # exactly as the per-energy expressions associate them
        Tmm_vac=Tmm_tmp * (1 - Um3sq_t - Um2sq_t),
        Jfac=Jmatter_t * dm21 * dm31 * (dm31 - dm21),
    )


def _channels(inv, energy_eV, length_invEV, v_cc_eV, sign=None):
    """The 9 probabilities ``P(alpha -> beta)`` in NuFast order
    ``(Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt)`` (index ``3*alpha + beta``).

    Purely elementwise in ``energy_eV`` (any shape); ``inv`` from
    :func:`_vacuum_invariants`.  ``sign`` (``None`` or an array of +-1.0
    broadcastable to ``energy_eV``) flips neutrino -> antineutrino *per energy*:
    an antineutrino sees ``-V_CC`` and ``-delta``, and ``delta`` enters the
    energy-dependent part only through the Jarlskog prefactor ``Jfac`` (odd in
    ``delta``; ``cos(delta)`` is even), so ``sign * Jfac`` with ``inv`` built at
    ``anti=False`` is the antineutrino prefactor.  Negation is exact, so
    this matches evaluating with ``anti=True`` invariants.
    """
    dm21, dm31 = inv["dm21"], inv["dm31"]
    See, Tee, Tmm_tmp = inv["See"], inv["Tee"], inv["Tmm_tmp"]
    Jfac = inv["Jfac"]
    Amatter = 2.0 * energy_eV * v_cc_eV
    if sign is not None:
        Amatter = Amatter * sign
        Jfac = Jfac * sign
    Cc = Amatter * Tee
    A = inv["A_tmp"] + Amatter
    B = Tmm_tmp + Amatter * See

    # lambda3 from the exact cubic, then the other eigenvalue gaps analytically.
    rootAsqB = jnp.sqrt(A * A - 3.0 * B)
    arg = (A**3 - 4.5 * A * B + 13.5 * Cc) / rootAsqB**3
    ss0 = arccos(jnp.clip(arg, -1.0, 1.0))
    ss0 = jnp.where(dm31 < 0.0, ss0 + _TWO_PI, ss0)
    lambda3 = (A + 2.0 * rootAsqB * _cos(ss0 / 3.0)) / 3.0

    tmp = A - lambda3
    # Floored so that Dl21 > 0 strictly: see _DL21_FLOOR_REL above.  The
    # double-where also keeps d(sqrt)/dx finite when the discriminant is 0.
    _disc = tmp * tmp - 4.0 * Cc / lambda3
    _floor2 = (_DL21_FLOOR_REL * jnp.abs(dm31)) ** 2
    _big = _disc > _floor2
    Dl21 = jnp.sqrt(jnp.where(_big, _disc, _floor2))
    lambda2 = 0.5 * (A - lambda3 + Dl21)
    Dl32 = lambda3 - lambda2
    Dl31 = Dl32 + Dl21
    PiInv = 1.0 / (Dl31 * Dl32 * Dl21)
    Xp3 = PiInv * Dl21
    Xp2 = -PiInv * Dl31

    # Rosetta: matter mixing magnitudes from eigenvalues + invariants.
    Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
    Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2
    Smm = A - dm21 * inv["Um2sq_t"] - dm31 * inv["Um3sq_t"]
    Tmm = inv["Tmm_vac"] + Amatter * (See + Smm - A)
    Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
    Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2
    Jmatter = Jfac * PiInv

    Ue1sq = 1 - Ue3sq - Ue2sq
    Um1sq = 1 - Um3sq - Um2sq
    Ut3sq = 1 - Um3sq - Ue3sq
    Ut2sq = 1 - Um2sq - Ue2sq
    Ut1sq = 1 - Um1sq - Ue1sq

    Lover4E = length_invEV / (4.0 * energy_eV)
    D21 = Dl21 * Lover4E
    D32 = Dl32 * Lover4E
    sinD21 = _sin(D21)
    sinD31 = _sin(D32 + D21)
    sinD32 = _sin(D32)
    triple = sinD21 * sinD31 * sinD32
    s21 = 2 * sinD21**2
    s31 = 2 * sinD31**2
    s32 = 2 * sinD32**2

    Pme_TC = ((Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * s21
              + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * s31
              + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * s32)
    Pme_TV = -Jmatter * triple
    Pmm = 1 - 2 * (Um2sq * Um1sq * s21 + Um3sq * Um1sq * s31 + Um3sq * Um2sq * s32)
    Pee = 1 - 2 * (Ue2sq * Ue1sq * s21 + Ue3sq * Ue1sq * s31 + Ue3sq * Ue2sq * s32)
    Pem = Pme_TC - Pme_TV
    Pme = Pme_TC + Pme_TV
    Pet = 1 - Pee - Pem
    Pmt = 1 - Pme - Pmm
    Pte = 1 - Pee - Pme
    Ptm = 1 - Pem - Pmm
    Ptt = 1 - Pet - Pmt
    return (Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt)


def prob_matrix(params, energy_eV, length_invEV, v_cc_eV, anti=False):
    """``P[beta, alpha] = P(nu_alpha -> nu_beta)`` (3x3) at one energy.

    Natural units: ``energy_eV`` in eV, ``length_invEV`` in eV^-1, ``v_cc_eV`` the
    charged-current potential in eV.  An array ``energy_eV`` is evaluated
    elementwise; a 1-D one gives ``(N, 3, 3)``.
    """
    if anti:
        v_cc_eV = -v_cc_eV
    Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = _channels(
        _vacuum_invariants(params, anti), energy_eV, length_invEV, v_cc_eV)
    # NuFast indexes probs[alpha][beta] = P(alpha -> beta); stack transposed,
    # P[..., beta, alpha], with the 3x3 axes last (row-major, the layout every
    # caller vmaps into).  Stacking along the *last* axis matters: building
    # the (3, 3, ...) array and transposing it measured ~3x slower on CPU.
    P = jnp.stack([Pee, Pme, Pte, Pem, Pmm, Ptm, Pet, Pmt, Ptt], axis=-1)
    P = P.reshape(P.shape[:-1] + (3, 3))
    if P.ndim > 3:
        # Historical behaviour for >= 2-D energy arrays: the result used to be
        # ``jnp.array([[...]]).T``, which also reversed the energy axes.
        nb = P.ndim - 2
        P = jnp.transpose(P, tuple(range(nb - 1, -1, -1)) + (nb, nb + 1))
    return P


def event_probability(params, energy_GeV, baseline_km, flavor_from, flavor_to,
                      density=0.0, ye=0.5, anti=False):
    """Per-event ``P(nu_{flavor_from[i]} -> nu_{flavor_to[i]})`` at ``energy_GeV[i]``.

    The event-sample counterpart of
    ``probability_constant(params, E, L, density, ye, anti)[i, flavor_to[i],
    flavor_from[i]]`` (constant density, standard 3-flavor, same units and
    conventions; same values to rounding), evaluated in one elementwise pass:
    only the channel each event needs is selected, nothing of shape
    ``(N, 3, 3)`` is built, and there is no gather (whose reverse mode is a
    scatter).  ``flavor_from``/``flavor_to`` are integer arrays with values in
    ``{0, 1, 2}`` (not checked under tracing; out-of-range codes give an
    unspecified channel).  ``anti`` is a Python bool (whole sample) or a boolean
    array broadcastable to ``energy_GeV`` (per event, e.g. a sorted nu/antinu
    sample in one call).  ``params`` must be :func:`eligible` (3-flavor, no NSI).
    """
    # oscillator imports this module, hence the deferred import
    from .oscillator import _check_nonneg, _check_positive, n_flavors

    _check_positive(energy_GeV, "energy_GeV", "GeV, not eV")
    _check_nonneg(baseline_km, "baseline_km", "km, not m")
    _check_nonneg(density, "density", "g/cm^3")
    n = n_flavors(params)
    if not eligible(params, None, getattr(params, "n_active", n)):
        raise ValueError(
            "nufast.event_probability needs standard 3-flavor params (no "
            "steriles); use probability_constant with a matrix backend instead"
        )

    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    length_invEV = jnp.asarray(baseline_km) * C.KM_TO_INV_EV
    v_cc, _ = C.matter_potentials(density, ye)

    if isinstance(anti, (bool, int)):
        static_anti, sign = bool(anti), None
        if static_anti:
            v_cc = -v_cc
    else:
        static_anti = False
        sign = jnp.where(jnp.asarray(anti), -1.0, 1.0)
    ch = _channels(_vacuum_invariants(params, static_anti), energy_eV,
                   length_invEV, v_cc, sign)
    code = 3 * jnp.asarray(flavor_from, jnp.int32) + jnp.asarray(flavor_to, jnp.int32)
    shape = jnp.broadcast_shapes(jnp.shape(code), *(jnp.shape(c) for c in ch))
    return lax.select_n(jnp.broadcast_to(code, shape),
                        *(jnp.broadcast_to(c, shape) for c in ch))


def eligible(params, nsi, n_active):
    """True if the NuFast direct formula applies (standard 3-flavor, no NSI)."""
    return params.msquared().shape[0] == 3 and nsi is None and n_active == 3
