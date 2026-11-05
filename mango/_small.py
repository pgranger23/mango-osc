"""Private helpers for batched tiny-matrix kernels (performance only).

Two XLA:CPU pathologies dominate the cost of MANGO's 3x3 kernels once they are
``vmap``-ed over an (energy, zenith) grid:

* **Tiny ``dot_general``.**  A batched 3x3 complex ``a @ b`` (or ``einsum``) is
  lowered to a generic dot that is ~7-20x slower than the same contraction
  written as elementwise multiply-adds, which XLA fuses into one vectorised
  loop.  :func:`matmul_small` is that formulation (same contraction, so equal to
  ``a @ b`` up to floating-point summation order).  It unrolls the (static)
  inner dimension rather than using ``(a[..., :, :, None] * b[..., None, :, :])
  .sum(-2)``: the two are equally fast for complex input, but the reduce form is
  ~9x slower than the unrolled one for *real* input on XLA:CPU.

* **Fusion duplication.**  XLA:CPU happily *duplicates* a shared producer into
  every consumer fusion -- including transcendental ops.  In the Cayley
  propagator the eigenvalues (``arccos``, two ``cos``) feed three
  ``exp(-i mu)``, which feed nine output entries, which feed the nine entries of
  the layer product: the optimized HLO of one scan step contained 42 ``cos``
  instead of 5 and ran ~4x slower than necessary.  :func:`materialize` (an
  ``optimization_barrier``) pins such fan-out points so each is computed once.
  It is the identity on values and gradients; on GPU/TPU, where XLA's fusion
  trade-offs differ and kernel launches are not free, it is a no-op.

A third one concerns the transcendentals themselves:

* **Scalar libm calls.**  XLA:CPU evaluates float64 ``sin``/``cos`` (and the
  ``atan2`` that ``arccos`` lowers to) inside a fused elementwise loop as calls
  to the C math library, ~10-20 ns each and one element at a time -- the call
  also keeps the rest of the loop from being vectorised.  In the Cayley layer
  kernel (one ``arccos``, two ``cos``, three ``exp(-i mu)``) and NuFast they are
  the largest single cost.  :func:`sincos` and :func:`arccos` are the classic
  fdlibm algorithms (Cody-Waite argument reduction and the minimax polynomials
  of ``__kernel_sin``/``__kernel_cos``; ``__ieee754_acos``) written as plain
  arithmetic, which XLA fuses and vectorises: same results to ~1 ulp, several
  times faster.  CPU only (GPUs have fast vectorised ``sin``/``cos``; their
  numerics are left untouched).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def materialize(x):
    """Compute the pytree ``x`` once (no duplication into consumers) on CPU.

    Identity in value and derivative (``lax.optimization_barrier`` is
    transparent to ``jvp``/``vmap``/transposition); a no-op off-CPU.
    """
    if jax.default_backend() == "cpu":
        return jax.lax.optimization_barrier(x)
    return x


def matmul_small(a, b):
    """``a @ b`` for tiny (static) trailing matrix dims, as fused multiply-adds.

    ``C[..., i, j] = sum_k a[..., i, k] * b[..., k, j]``, accumulated in order
    ``k = 0, 1, ...``; batched over any leading dimensions (broadcast like
    ``jnp.matmul``).
    """
    acc = a[..., :, 0:1] * b[..., None, 0, :]
    for k in range(1, a.shape[-1]):
        acc = acc + a[..., :, k:k + 1] * b[..., None, k, :]
    return acc


# --- vectorisable float64 sin / cos / arccos (CPU) ---------------------------
#
# fdlibm / FreeBSD msun algorithms (public domain, Sun Microsystems), written as
# elementwise arithmetic with branch-free selects so that XLA fuses and
# vectorises them (see the module docstring).  Accuracy, measured against
# long-double references: sin/cos within 1.5 ulp (absolute <= 1.2e-16; XLA's
# libm calls: 0.5 ulp) and agreeing with ``jnp.sin``/``jnp.cos`` to <= 1 ulp
# (identical for ~85% of arguments); arccos within 1 ulp of ``jnp.arccos``.
# Derivatives are exactly the ``jnp`` rules (custom JVPs, so forward and reverse
# mode, vmap and higher derivatives all work).


def _fast_trig(x):
    """Use the arithmetic formulations: float64 on the CPU backend (trace-time
    choice, like :func:`materialize`)."""
    return (jax.default_backend() == "cpu"
            and jnp.result_type(x) == jnp.float64)


# Cody-Waite reduction x = k pi/2 + r, |r| <= pi/4: pi/2 = _PIO2_1 + _PIO2_2 +
# _PIO2_3 (+ 4e-35), with 27 and 25 significant bits in the first two parts, so
# k*_PIO2_1 and k*_PIO2_2 are exact and x - k*_PIO2_1 is exact (Sterbenz) for
# |k| < 2^26, i.e. |x| < 1.05e8 rad -- far beyond any phase a 3-flavour
# calculation produces (~1e6 for 0.1 MeV across the Earth).  r is then accurate
# to ~1e-17 absolute, and its rounding error is carried along as the tail ``y``
# of the kernels.  Beyond that range the products k*_PIO2_i round, by at most
# ~ulp(x)/2 -- the size of the rounding error x itself carries -- unless XLA
# contracts them into FMAs, as XLA:CPU does (then measured within 1.5 ulp up to
# |x| ~ 3e15).  For |x| >~ 1e16 (ulp(x) >= 2 rad: no phase information left)
# r and y are clipped to their in-range bounds, so the result stays on the unit
# circle instead of growing; inf/NaN give NaN as jnp.sin/cos do.
_PIO2_1 = float.fromhex("0x1.921fb54000000p+0")
_PIO2_2 = float.fromhex("0x1.10b4610000000p-30")
_PIO2_3 = float.fromhex("0x1.a62633145c06ep-58")
_INV_PIO2 = 6.36619772367581382433e-01
_R_MAX = 0.7854  # pi/4 = 0.78539816..., plus room for the rounding of k
_Y_MAX = 2.0 ** -52  # |y| <= ulp(r)/2 <= 2^-54 in range
# __kernel_sin / __kernel_cos minimax coefficients on [-pi/4, pi/4].
_S1, _S2, _S3, _S4, _S5, _S6 = (
    -1.66666666666666324348e-01, 8.33333333332248946124e-03,
    -1.98412698298579493134e-04, 2.75573137070700676789e-06,
    -2.50507602534068634195e-08, 1.58969099521155010221e-10)
_C1, _C2, _C3, _C4, _C5, _C6 = (
    4.16666666666666019037e-02, -1.38888888888741095749e-03,
    2.48015872894767294178e-05, -2.75573143513906633035e-07,
    2.08757232129817482790e-09, -1.13596475577881948265e-11)


@jax.custom_jvp
def _sincos_arith(x):
    k = jnp.round(x * _INV_PIO2)
    t = (x - k * _PIO2_1) - k * _PIO2_2
    w = k * _PIO2_3
    r = t - w
    y = (t - r) - w  # tail: r + y = x - k pi/2 to ~1e-17
    # no-ops unless |x| >~ 1e15 (see above); keep the result on the unit circle
    r = jnp.clip(r, -_R_MAX, _R_MAX)
    y = jnp.clip(y, -_Y_MAX, _Y_MAX)
    z = r * r
    zz = z * z
    v = z * r
    ps = _S2 + z * (_S3 + z * _S4) + z * zz * (_S5 + z * _S6)
    s = r - ((z * (0.5 * y - v * ps) - y) - v * _S1)
    pc = z * (_C1 + z * (_C2 + z * _C3)) + zz * zz * (_C4 + z * (_C5 + z * _C6))
    hz = 0.5 * z
    w1 = 1.0 - hz
    c = w1 + (((1.0 - w1) - hz) + (z * pc - r * y))
    # quadrant q = k mod 4 (exact): sin x = (s, c, -s, -c)[q], cos x = (c, -s, -c, s)[q]
    q = k - 4.0 * jnp.floor(0.25 * k)
    odd = (q == 1.0) | (q == 3.0)
    sn = jnp.where(odd, c, s)
    cs = jnp.where(odd, s, c)
    return (jnp.where(q >= 2.0, -sn, sn),
            jnp.where((q == 1.0) | (q == 2.0), -cs, cs))


@_sincos_arith.defjvp
def _sincos_arith_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    # (s, c) pinned (see sincos(pinned=True)): they are also the derivative,
    # which reverse mode would otherwise recompute in every cotangent fusion.
    z = jax.lax.optimization_barrier(jax.lax.complex(*_sincos_arith(x)[::-1]))
    s, c = jnp.imag(z), jnp.real(z)
    return (s, c), (c * t, -s * t)  # the jnp.sin / jnp.cos rules


def sincos(x, pinned=False):
    """``(sin x, cos x)`` (elementwise); the vectorisable formulation on CPU.

    ``pinned=True`` computes both once, into one buffer, behind an
    ``optimization_barrier`` (identity in value and derivative): XLA:CPU would
    otherwise recompute them inside every consumer fusion, and with a barrier
    on the *pair* it still gives each of the two its own fusion, so on CPU they
    are pinned packed as one complex ``cos + i sin``.
    """
    x = jnp.asarray(x)
    if not _fast_trig(x):
        sc = (jnp.sin(x), jnp.cos(x))
        return jax.lax.optimization_barrier(sc) if pinned else sc
    s, c = _sincos_arith(x)
    if pinned:
        z = jax.lax.optimization_barrier(jax.lax.complex(c, s))
        return jnp.imag(z), jnp.real(z)
    return s, c


# __ieee754_acos: acos(x) = pi/2 - (x + x R(x^2)) for |x| < 1/2, else
# 2 asin(sqrt((1 - |x|)/2)) mirrored, with R a rational minimax approximation.
_PI = 3.14159265358979311600e+00
_PIO2_HI = 1.57079632679489655800e+00
_PIO2_LO = 6.12323399573676603587e-17
_PS0, _PS1, _PS2, _PS3, _PS4, _PS5 = (
    1.66666666666666657415e-01, -3.25565818622400915405e-01,
    2.01212532134862925881e-01, -4.00555345006794114027e-02,
    7.91534994289814532176e-04, 3.47933107596021167570e-05)
_QS1, _QS2, _QS3, _QS4 = (
    -2.40339491173441421878e+00, 2.02094576023350569471e+00,
    -6.88283971605453293030e-01, 7.70381505559019352791e-02)


@jax.custom_jvp
def _arccos_arith(x):
    ax = jnp.abs(x)
    small = ax < 0.5
    z = jnp.where(small, x * x, (1.0 - ax) * 0.5)
    p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
    q = 1.0 + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
    r = p / q
    a_small = _PIO2_HI - (x - (_PIO2_LO - x * r))
    s = jnp.sqrt(z)  # NaN for |x| > 1, as jnp.arccos
    a_neg = _PI - 2.0 * (s + (r * s - _PIO2_LO))
    # x > 1/2: df = s with its low 32 bits cleared, so df*df is exact
    df = jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(s, jnp.uint64)
        & jnp.uint64(0xFFFFFFFF00000000), jnp.float64)
    c = (z - df * df) / (s + df)
    a_pos = 2.0 * (df + (r * s + c))
    out = jnp.where(small, a_small, jnp.where(x < 0.0, a_neg, a_pos))
    return jnp.where(x == 1.0, 0.0, jnp.where(x == -1.0, _PI, out))


@_arccos_arith.defjvp
def _arccos_arith_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    # the jnp.arccos rule: -t / sqrt(1 - x^2)
    return (_arccos_arith(x),
            t * jax.lax.neg(jax.lax.rsqrt(jax.lax.one_minus_square(x))))


def arccos(x):
    """``arccos x`` (elementwise); the vectorisable formulation on CPU."""
    x = jnp.asarray(x)
    if _fast_trig(x):
        return _arccos_arith(x)
    return jnp.arccos(x)
