"""PMNS leptonic mixing matrix construction.

Written generically in ``N`` flavors as a product of complex plane rotations so
that sterile-neutrino extensions only need to add rotations.  The 3-flavor PDG
parametrization ``U = R23 . U13(delta) . R12`` is provided as a helper.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._small import matmul_small


def plane_rotation(n: int, i: int, j: int, theta, delta=0.0):
    """Complex rotation in the ``(i, j)`` plane of an ``n``-dim space.

    Builds the unitary matrix that is the identity except for::

        U[i, i] = U[j, j] = cos(theta)
        U[i, j] =  sin(theta) * exp(-i delta)
        U[j, i] = -sin(theta) * exp(+i delta)

    With ``delta = 0`` this is a real rotation; a non-zero ``delta`` carries the
    CP-violating phase (used on the (1, 3) plane in the standard convention).
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    phase = jnp.exp(-1j * jnp.asarray(delta, dtype=jnp.complex128))

    u = jnp.eye(n, dtype=jnp.complex128)
    u = u.at[i, i].set(c.astype(jnp.complex128))
    u = u.at[j, j].set(c.astype(jnp.complex128))
    u = u.at[i, j].set(s.astype(jnp.complex128) * phase)
    u = u.at[j, i].set(-s.astype(jnp.complex128) * jnp.conj(phase))
    return u


def pmns_3flavor(theta12, theta13, theta23, deltacp):
    """Standard PDG 3-flavor PMNS matrix ``U = R23 . U13(delta) . R12``."""
    r23 = plane_rotation(3, 1, 2, theta23, 0.0)
    u13 = plane_rotation(3, 0, 2, theta13, deltacp)
    r12 = plane_rotation(3, 0, 1, theta12, 0.0)
    return matmul_small(matmul_small(r23, u13), r12)


def pmns_3flavor_factors(theta12, theta13, theta23, deltacp):
    """Factor the PDG matrix as ``U = W . O . D^dagger`` (reduced basis).

    With ``D = diag(1, 1, e^{i delta})`` the CP-phase rotation is
    ``U13(delta) = D R13 D^dagger`` and ``D^dagger`` commutes with ``R12``, so

        U = (R23 D) (R13 R12) D^dagger = W O D^dagger,

    ``W = R23 D`` complex unitary, ``O = R13 R12`` *real* orthogonal.  ``W`` acts
    only on the (mu, tau) block, so it commutes with the standard matter term
    ``diag(V, 0, 0)``; hence ``H = W H' W^dagger`` with the **real symmetric**
    ``H' = O diag(msq/2E) O^T + diag(V, 0, 0)``, and ``exp(-iHL) = W exp(-iH'L)
    W^dagger`` (``D`` only rephases mass states and drops out of every
    probability).  Antineutrinos (``U -> U*``) use ``W*`` with the same ``O``.

    Returns ``(W, O)`` as ``(3, 3)`` complex / real arrays.
    """
    c12, s12 = jnp.cos(theta12), jnp.sin(theta12)
    c13, s13 = jnp.cos(theta13), jnp.sin(theta13)
    c23, s23 = jnp.cos(theta23), jnp.sin(theta23)
    zero, one = jnp.zeros_like(c12), jnp.ones_like(c12)
    o = jnp.stack([jnp.stack([c13 * c12, c13 * s12, s13]),
                   jnp.stack([-s12, c12, zero]),
                   jnp.stack([-s13 * c12, -s13 * s12, c13])])
    e = jnp.exp(1j * jnp.asarray(deltacp, dtype=jnp.complex128))
    z = jnp.zeros_like(e)
    w = jnp.stack([jnp.stack([one + z, z, z]),
                   jnp.stack([z, c23 + z, s23 * e]),
                   jnp.stack([z, -s23 + z, c23 * e])])
    return w, o
