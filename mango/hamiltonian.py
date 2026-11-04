"""Flavor-basis Hamiltonian construction (generic in N flavors).

All quantities are in **natural units**: energy ``E`` in eV, mass-squared values
in eV^2, matter potential ``V`` in eV.  The resulting Hamiltonian has units of eV
and is meant to be multiplied by a length in eV^-1 inside the propagator.

The builder is deliberately generic so that sterile / NSI extensions just pass a
larger ``U`` / a non-zero ``nsi`` matrix:

    H = (1/2E) U diag(msq) U^dagger  +  V * (P_e + nsi)

where ``P_e = diag(1, 0, ..., 0)`` is the standard charged-current term acting on
the electron flavor.  Antineutrinos use ``U -> U*`` and ``V -> -V``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ._small import matmul_small


def vacuum_hamiltonian(u, msq, energy_eV):
    """Vacuum Hamiltonian ``(1/2E) U diag(msq) U^dagger`` (shape ``(N, N)``)."""
    u = jnp.asarray(u, dtype=jnp.complex128)
    msq = jnp.asarray(msq)
    kin = msq / (2.0 * energy_eV)
    return matmul_small(u * kin, jnp.conj(u).T)


def matter_hamiltonian(u, msq, energy_eV, v_eV, anti=False, nsi=None,
                       v_nc_eV=0.0, n_active=None):
    """Full matter Hamiltonian in eV (active flavors + optional steriles/NSI).

    Parameters
    ----------
    u : array_like, shape (N, N)
        PMNS matrix (N = n_active + n_sterile).
    msq : array_like, shape (N,)
        Mass-squared values (eV^2), conventionally ``[0, dm21, dm31, ...]``.
    energy_eV : float or array
        Neutrino energy in eV.
    v_eV : float or array
        Charged-current matter potential ``sqrt(2) G_F N_e`` in eV.
    anti : bool
        If True, build the antineutrino Hamiltonian (``U -> U*``, ``V -> -V``,
        ``nsi -> nsi*``).
    nsi : array_like, shape (n_active, n_active), optional
        Dimensionless matter-NSI matrix ``epsilon``; the matter term on the active
        block is ``V_CC (P_e + epsilon)``.  Defaults to zero.
    v_nc_eV : float or array
        Sterile-sector potential (``(1/2) sqrt(2) G_F N_n``); added to the diagonal
        of each sterile flavor.  Zero for purely active models.
    n_active : int, optional
        Number of active flavors (default N).  Flavors ``>= n_active`` are
        sterile (CC/NC blind apart from the relative ``v_nc_eV`` term).
    """
    u = jnp.asarray(u, dtype=jnp.complex128)
    n = u.shape[0]
    if n_active is None:
        n_active = n
    if anti:
        u = jnp.conj(u)
        v_eV = -v_eV
        v_nc_eV = -v_nc_eV
        if nsi is not None:
            nsi = jnp.conj(jnp.asarray(nsi))

    return (vacuum_hamiltonian(u, msq, energy_eV)
            + matter_term(n, v_eV, v_nc_eV, nsi, n_active))


def matter_term(n, v_cc, v_nc=0.0, nsi=None, n_active=None):
    """Matter potential matrix (eV), without the vacuum part.

    ``CC`` term ``v_cc`` on nu_e (+ ``v_cc * nsi`` on the active block) and the
    sterile-sector ``v_nc`` on the diagonal of flavors ``>= n_active``.  Signs and
    NSI conjugation for antineutrinos are the caller's responsibility (see
    :func:`matter_hamiltonian`).  Cheap (no ``U diag U^dagger``), so it can be
    rebuilt per layer while the vacuum part is computed once.
    """
    if n_active is None:
        n_active = n
    # Built as potentials times constant 0/1 masks rather than with ``.at[]``
    # scatters: the same values (``x * 1 = x``; the ``x * 0`` entries are zeros
    # that add exactly), but plain elementwise ops that fuse into the caller's
    # per-layer loop instead of a (batched) scatter per segment.
    pe = np.zeros((n, n))
    pe[0, 0] = 1.0
    matter = (jnp.asarray(v_cc) * pe).astype(jnp.complex128)
    if nsi is not None:
        nsi = jnp.asarray(nsi, jnp.complex128)
        na = nsi.shape[0]
        if na < n:
            nsi = jnp.zeros((n, n), jnp.complex128).at[:na, :na].set(nsi)
        matter = matter + v_cc * nsi
    if n_active < n:
        ps = np.diag((np.arange(n) >= n_active).astype(float))
        matter = matter + (jnp.asarray(v_nc) * ps).astype(jnp.complex128)
    return matter
