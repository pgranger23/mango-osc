"""Analytic eigenvalues of a 3x3 Hermitian matrix.

Uses the stable trigonometric solution of the characteristic cubic (eigenvalues
of a Hermitian matrix are real).  Closed-form and differentiable, which makes it
preferable to a generic ``eigh`` for the small matrices appearing in 3-flavor
oscillations.  Only eigenvalues are produced; the :mod:`mango.propagator`
``cayley`` backend builds ``exp(-iHL)`` from them via Cayley-Hamilton, so no
eigenvectors are required.

Reference: standard 3x3 symmetric eigenvalue algorithm (Smith 1961), extended to
Hermitian matrices by using squared off-diagonal magnitudes.

Implementation note (performance): the solver works on the six independent
*real* quantities of a Hermitian 3x3 -- the real diagonal, the squared moduli of
the three off-diagonal entries and the real triple product
``2 Re(a01 a12 conj(a02))`` -- rather than on a complex matrix
(:func:`_eigvalsh3_parts`).  The characteristic-polynomial determinant is then a
handful of real multiplies instead of a generic complex ``det``, and the same
core serves the real-symmetric 3x3 of the reduced-basis fast path
(:func:`eigvalsh3_sym`, see :mod:`mango.layers`).  Every operation is
elementwise, so under ``vmap`` each entry is a contiguous batch array that XLA
fuses into a single vectorised loop.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._small import materialize

_EPS = 1e-12


def _sort3(d0, d1, d2):
    """Ascending sort of three arrays (elementwise), as a compare-exchange network.

    Equivalent to a *stable* ``jnp.sort`` of ``(d0, d1, d2)`` in value *and* in
    gradient: every output is selected (``where``) from exactly one input -- ties
    keep the original order, as the stable sort's permutation does -- so the
    tangent of each sorted value is that of the one input it came from (``min``/
    ``max`` would instead split the gradient 50/50 at ties).  Avoids XLA's general
    ``sort`` on a length-3 axis, which is costly on CPU.
    """
    s = d1 < d0
    a, b = jnp.where(s, d1, d0), jnp.where(s, d0, d1)
    s = d2 < b
    c, hi = jnp.where(s, d2, b), jnp.where(s, b, d2)
    s = c < a
    lo, mid = jnp.where(s, c, a), jnp.where(s, a, c)
    return lo, mid, hi


def _eigvalsh3_parts(d0, d1, d2, aa, bb, cc, t):
    """Core of the trigonometric solver on real invariants.

    ``d0, d1, d2``: the (real) diagonal; ``aa, bb, cc``: ``|a01|^2, |a02|^2,
    |a12|^2``; ``t``: ``2 Re(a01 a12 conj(a02))``.  Returns ``(lo, mid, hi)``.
    """
    # |off-diagonal|^2 sum.
    p1 = aa + bb + cc
    q = (d0 + d1 + d2) / 3.0  # mean of diagonal == trace/3

    b0, b1, b2 = d0 - q, d1 - q, d2 - q
    p2 = (b0 * b0 + b1 * b1 + b2 * b2) + 2.0 * p1
    # Guard against the diagonal/degenerate limit (p2 -> 0).  Double-`where` so the
    # degenerate branch carries no gradient: ``sqrt`` has an infinite derivative at
    # 0, which (via 0*inf in the unselected branch) would otherwise NaN the
    # gradient through a zero / repeated-eigenvalue matrix (e.g. a zero-length
    # segment).
    big = p2 > _EPS
    p2_safe = jnp.where(big, p2, 1.0)
    p = jnp.where(big, jnp.sqrt(p2_safe / 6.0), 0.0)
    p_div = jnp.where(big, p, 1.0)  # safe denominator

    # det(a - q I) of a Hermitian matrix, written in real arithmetic:
    #   b0 b1 b2 - b0 |a12|^2 - b1 |a02|^2 - b2 |a01|^2 + 2 Re(a01 a12 conj(a02)).
    detm = b0 * (b1 * b2 - cc) - b1 * bb - b2 * aa + t
    r = detm / (2.0 * p_div**3)
    # Clamp to the valid arccos domain (degenerate eigenvalues sit at r = +-1).
    r = jnp.clip(r, -1.0 + _EPS, 1.0 - _EPS)

    phi = jnp.arccos(r) / 3.0
    two_pi_3 = 2.0 * jnp.pi / 3.0

    e_hi = q + 2.0 * p * jnp.cos(phi)
    e_lo = q + 2.0 * p * jnp.cos(phi + two_pi_3)
    e_mid = 3.0 * q - e_hi - e_lo  # trace conservation

    # If the matrix was essentially diagonal, fall back to the sorted diagonal.
    s_lo, s_mid, s_hi = _sort3(d0, d1, d2)
    # Materialized: every downstream entry of exp(-iM) reads all three (see
    # mango._small -- otherwise XLA:CPU recomputes arccos/cos per consumer).
    return materialize((jnp.where(big, e_lo, s_lo), jnp.where(big, e_mid, s_mid),
                        jnp.where(big, e_hi, s_hi)))


def eigvalsh3_sym(m00, m11, m22, m01, m02, m12):
    """Eigenvalues ``(lo, mid, hi)`` of the real symmetric 3x3 given by its six
    independent entries (each may be an array; evaluated elementwise).

    Same algorithm and guards as :func:`eigvalsh3`, specialised to real input.
    """
    return _eigvalsh3_parts(m00, m11, m22, m01 * m01, m02 * m02, m12 * m12,
                            2.0 * (m01 * m12 * m02))


def eigvalsh3(a):
    """Eigenvalues of a 3x3 Hermitian matrix ``a``, sorted ascending.

    Returns a real array of shape ``(3,)``.
    """
    return jnp.stack(_eigvalsh3_tuple(jnp.asarray(a)))


def _eigvalsh3_tuple(a):
    """:func:`eigvalsh3` as a ``(lo, mid, hi)`` tuple (reads only the diagonal
    and the upper triangle of ``a``)."""
    if not jnp.iscomplexobj(a):
        return eigvalsh3_sym(a[0, 0], a[1, 1], a[2, 2], a[0, 1], a[0, 2], a[1, 2])
    re, im = jnp.real(a), jnp.imag(a)
    xr, xi = re[0, 1], im[0, 1]  # a01
    yr, yi = re[0, 2], im[0, 2]  # a02
    zr, zi = re[1, 2], im[1, 2]  # a12
    # Re(a01 a12 conj(a02)) with a01 a12 = (xr zr - xi zi) + i (xr zi + xi zr).
    t = 2.0 * ((xr * zr - xi * zi) * yr + (xr * zi + xi * zr) * yi)
    return _eigvalsh3_parts(re[0, 0], re[1, 1], re[2, 2],
                            xr * xr + xi * xi, yr * yr + yi * yi,
                            zr * zr + zi * zi, t)
