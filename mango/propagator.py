"""Evolution operator ``S = exp(-i H L)`` for a constant-density segment.

Three interchangeable backends:

* ``cayley`` (default, 3x3): Cayley-Hamilton expansion ``exp(-iHL) = a0 I + a1 H
  + a2 H^2`` with coefficients from the analytic eigenvalues.  Fast, eigenvector-
  free, clean gradients.
* ``eigh``: reconstruct from ``jnp.linalg.eigh``.  Works for any N (steriles);
  used as the validation oracle.
* ``expm``: ``jax.scipy.linalg.expm`` of ``-iHL``.  Reference cross-check.

``H`` is in eV, ``L`` in eV^-1, so ``H * L`` is dimensionless.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from ._small import materialize, matmul_small, sincos
from .eigensolve import _eigvalsh3_tuple, eigvalsh3_sym


def propagator_eigh(h, length):
    """``exp(-i H L)`` via Hermitian eigendecomposition (any N).

    .. warning::
       This is the general-``N`` reference path, not the fast one, and its
       **gradient** is unsafe at degeneracy: JAX's VJP for ``eigh`` contains
       eigenvector-derivative terms in ``1/(lambda_i - lambda_j)``, which blow
       up when two eigenvalues coincide. The default ``cayley`` backend needs no
       eigenvectors and is accurate at exact degeneracy, so prefer it whenever
       derivatives are taken. Nothing here can detect the problem at run time --
       the divergence appears as a large or non-finite gradient, not as an
       error -- so this note is the only guard.
    """
    w, v = jnp.linalg.eigh(h)
    phase = jnp.exp(-1j * w * length)
    return matmul_small(v * phase, jnp.conj(v).T)


def propagator_expm(h, length):
    """``exp(-i H L)`` via matrix exponential (any N)."""
    return jsl.expm(-1j * h * length)


# Threshold below which two eigenphases are treated as confluent and the
# divided difference (f(a)-f(b))/(a-b) is replaced by its derivative form.
#
# Measured, do not raise without re-measuring: eigvalsh3 resolves an exactly
# degenerate pair only to ~5e-7 (relative to the spectral radius), because the
# trigonometric cubic solution goes through arccos(r)/3 whose derivative
# diverges as r -> +-1.  That computed splitting exceeds this threshold, so at
# exact degeneracy the ordinary divided-difference branch is what actually runs.
# That is deliberate: it is the more accurate of the two here.  Sweeping this
# constant against a scipy.linalg.expm reference (15 random Hermitian 3x3 per
# point) gives, for the propagator error at exact degeneracy / worst case over
# true gaps 1e-5..1:
#     1e-7 (this value) -> 1.2e-14 / 1.2e-14
#     1e-6              -> 1.3e-13 / 7.1e-14
#     1e-5              -> 1.4e-13 / 4.9e-11
# i.e. engaging the confluent branch earlier costs accuracy, because it drops
# O(d) terms that are not actually negligible at a splitting of ~5e-7.
_DEGEN_EPS = 1e-7


def _f(x):
    # exp(-i x) for real x, as (cos x, -sin x) without the complex exp's exp(0)
    # of the (zero) real part; sin/cos from one vectorisable sincos on CPU
    # (mango._small; elsewhere jnp.sin/cos: bit-identical to jnp.exp(-1j * x)).
    s, c = sincos(x)
    return jax.lax.complex(c, -s)


def _fp(x, fx=None):
    # f'(x) = -i exp(-i x) = -sin x - i cos x (bit-identical to -1j * _f(x)),
    # read off a precomputed fx = f(x) when given (no further transcendentals).
    if fx is None:
        fx = _f(x)
    return jax.lax.complex(jnp.imag(fx), -jnp.real(fx))


def _cdiv(z, d):
    # complex / real, componentwise (bit-identical to ``z / d`` for real ``d``).
    return jax.lax.complex(jnp.real(z) / d, jnp.imag(z) / d)


def _divdiff(a, b, fa=None, fb=None):
    """First divided difference ``(f(a)-f(b))/(a-b)`` with confluent limit.

    Uses a safe denominator so the *unused* branch never feeds NaN into the
    reverse-mode gradient (JAX ``where`` propagates NaNs from both branches).
    ``fa``/``fb`` optionally pass precomputed ``f(a)``/``f(b)``.
    """
    fa = _f(a) if fa is None else fa
    fb = _f(b) if fb is None else fb
    d = a - b
    big = jnp.abs(d) > _DEGEN_EPS
    safe = jnp.where(big, d, 1.0)
    return jnp.where(big, _cdiv(fa - fb, safe), _fp(a, fa))


def _newton_coefficients(m0, m1, m2):
    """``f[m0]``, ``f[m0,m1]``, ``f[m0,m1,m2]`` of ``f(x) = exp(-ix)`` (ascending
    eigenphases), with the confluent limits; each ``f(m_k)`` computed once."""
    f0, f1, f2 = materialize((_f(m0), _f(m1), _f(m2)))
    d01 = _divdiff(m0, m1, f0, f1)
    d12 = _divdiff(m1, m2, f1, f2)
    dd = m2 - m0
    big = jnp.abs(dd) > _DEGEN_EPS
    safe = jnp.where(big, dd, 1.0)
    d012 = jnp.where(big, _cdiv(d12 - d01, safe), -0.5 * f0)  # f''(m0)/2
    return materialize((f0, d01, d012))


def propagator_cayley(h, length):
    """``exp(-i H L)`` via Cayley-Hamilton / Newton divided differences (3x3).

    Works with the dimensionless matrix ``M = H L`` (eigenvalues O(1)) so the
    interpolation is well conditioned -- the raw Hamiltonian eigenvalues are
    ~1e-13 eV.  The Hermite (Newton) form

        S = f[m0] I + f[m0,m1] (M - m0 I) + f[m0,m1,m2] (M - m0 I)(M - m1 I)

    is exact for repeated eigenvalues, so zero-length segments (degenerate zero
    eigenvalues -> identity) and level crossings are handled without the
    singular Vandermonde solve.

    ``H`` must be Hermitian (only its real diagonal and upper triangle are
    read).  The product ``(M - m0 I)(M - m1 I)`` of two commuting Hermitian
    matrices is Hermitian, so it is formed entrywise from its diagonal and upper
    triangle (no batched ``dot``, see :mod:`mango._small`); the shared
    intermediates are materialized once.
    """
    m = h * length  # Hermitian, eigenvalues O(1)
    if not jnp.iscomplexobj(m):
        s = propagator_cayley_sym(m[0, 0], m[1, 1], m[2, 2],
                                  m[0, 1], m[0, 2], m[1, 2])
        return _sym_to_matrix(s)
    m0, m1, m2 = _eigvalsh3_tuple(m)  # real, ascending
    f0, d01, d012 = _newton_coefficients(m0, m1, m2)

    re = jnp.real(m)
    x, y, z = m[0, 1], m[0, 2], m[1, 2]
    xx = jnp.real(x) ** 2 + jnp.imag(x) ** 2
    yy = jnp.real(y) ** 2 + jnp.imag(y) ** 2
    zz = jnp.real(z) ** 2 + jnp.imag(z) ** 2
    # A = M - m0 I, B = M - m1 I (real diagonals; off-diagonals x, y, z of M).
    a0, a1, a2 = re[0, 0] - m0, re[1, 1] - m0, re[2, 2] - m0
    b0, b1, b2 = re[0, 0] - m1, re[1, 1] - m1, re[2, 2] - m1
    ab00 = a0 * b0 + xx + yy
    ab11 = xx + a1 * b1 + zz
    ab22 = yy + zz + a2 * b2
    ab01 = (a0 + b1) * x + y * jnp.conj(z)
    ab02 = (a0 + b2) * y + x * z
    ab12 = jnp.conj(x) * y + (a1 + b2) * z
    s00 = f0 + d01 * a0 + d012 * ab00
    s11 = f0 + d01 * a1 + d012 * ab11
    s22 = f0 + d01 * a2 + d012 * ab22
    s01 = d01 * x + d012 * ab01
    s02 = d01 * y + d012 * ab02
    s12 = d01 * z + d012 * ab12
    s10 = d01 * jnp.conj(x) + d012 * jnp.conj(ab01)
    s20 = d01 * jnp.conj(y) + d012 * jnp.conj(ab02)
    s21 = d01 * jnp.conj(z) + d012 * jnp.conj(ab12)
    return materialize(jnp.stack([jnp.stack([s00, s01, s02]),
                                  jnp.stack([s10, s11, s12]),
                                  jnp.stack([s20, s21, s22])]))


def propagator_cayley_sym(m00, m11, m22, m01, m02, m12):
    """``exp(-i M)`` for a *real symmetric* dimensionless 3x3 ``M`` (= H L).

    Same Newton/Cayley-Hamilton construction and degeneracy guards as
    :func:`propagator_cayley`, specialised to real input: the matrix polynomial
    is real and only the three scalar coefficients are complex, and the result
    is complex *symmetric* (``exp(-iM)^T = exp(-iM^T)``).  Takes and returns the
    six independent entries ``(00, 11, 22, 01, 02, 12)`` as separate
    (elementwise-batched) arrays; this is the per-layer kernel of the
    reduced-basis fast path in :mod:`mango.layers`.
    """
    m0, m1, m2 = eigvalsh3_sym(m00, m11, m22, m01, m02, m12)
    f0, d01, d012 = _newton_coefficients(m0, m1, m2)
    a0, a1, a2 = m00 - m0, m11 - m0, m22 - m0
    b0, b1, b2 = m00 - m1, m11 - m1, m22 - m1
    # (M - m0 I)(M - m1 I): product of commuting symmetric matrices, symmetric.
    ab00 = a0 * b0 + m01 * m01 + m02 * m02
    ab11 = m01 * m01 + a1 * b1 + m12 * m12
    ab22 = m02 * m02 + m12 * m12 + a2 * b2
    ab01 = (a0 + b1) * m01 + m02 * m12
    ab02 = (a0 + b2) * m02 + m01 * m12
    ab12 = m01 * m02 + (a1 + b2) * m12
    return materialize((f0 + d01 * a0 + d012 * ab00,
                        f0 + d01 * a1 + d012 * ab11,
                        f0 + d01 * a2 + d012 * ab22,
                        d01 * m01 + d012 * ab01,
                        d01 * m02 + d012 * ab02,
                        d01 * m12 + d012 * ab12))


def _sym_to_matrix(s):
    """``(00, 11, 22, 01, 02, 12)`` entries -> symmetric ``(..., 3, 3)`` matrix."""
    s00, s11, s22, s01, s02, s12 = s
    return jnp.stack([jnp.stack([s00, s01, s02], -1),
                      jnp.stack([s01, s11, s12], -1),
                      jnp.stack([s02, s12, s22], -1)], -2)


_BACKENDS = {
    "cayley": propagator_cayley,
    "eigh": propagator_eigh,
    "expm": propagator_expm,
}


def propagator(h, length, backend="cayley"):
    """Dispatch to a propagator backend by name."""
    try:
        fn = _BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {sorted(_BACKENDS)}"
        )
    return fn(h, length)
