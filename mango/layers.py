"""Piecewise-constant layer propagation.

Given per-segment matter potentials and (natural-unit) lengths, build each
segment evolution operator and chain them ``S = S_N ... S_1`` so that ``S`` acts
on an initial flavor state.  Two product strategies, auto-selected by device:

* ``parallel=False`` (CPU default): a sequential ``lax.scan``.  Minimal total
  work and memory -- fastest on CPU, where the parallel variant has no hardware
  to exploit.
* ``parallel=True`` (GPU/TPU default): build all per-layer propagators with
  ``vmap`` and combine with ``lax.associative_scan`` (parallel prefix, O(log N)
  depth).  Faster on accelerators; slower on CPU (it computes all prefixes).

In both, the layer-independent vacuum Hamiltonian is computed once, outside the
per-layer loop, and the 3x3 products avoid XLA's slow batched tiny ``dot`` (see
:mod:`mango._small`).  Reverse-mode autodiff flows through both paths.

**Reduced-basis fast path** (:func:`propagate_layers_pdg`): for the standard
3-flavor PDG mixing without NSI the theta23/delta rotation commutes with the
matter term (:func:`mango.pmns.pmns_3flavor_factors`), so every layer
Hamiltonian is ``W H'_k W^dagger`` with a *real symmetric* ``H'_k`` and

    S = W [prod_k exp(-i H'_k L_k)] W^dagger.

Each layer then needs only the eigenvalues of a real symmetric 3x3 and a real
matrix polynomial with three complex scalar coefficients
(:func:`mango.propagator.propagator_cayley_sym`).  Moreover ``exp(-i H' L)`` is
complex *symmetric*, so for a trajectory that is mirror-symmetric about its
midpoint (an Earth chord seen from a detector at the surface: the ascending leg
crosses the same shells as the descending one, in reverse order) the product of
the second half is the transpose of the first, ``S_half^T``, and only half of
the layers need computing (NuFast-Earth's "symmetric trajectory" trick).
:func:`mango.oscillator.probability_earth` / ``probability_profile`` use this
path automatically when it applies.
"""

from __future__ import annotations

import operator

import jax
import jax.numpy as jnp

from ._small import matmul_small
from .hamiltonian import vacuum_hamiltonian, matter_term
from .pmns import pmns_3flavor_factors
from .propagator import propagator, propagator_cayley_sym


def propagate_layers(u, msq, energy_eV, v_eV, length_invEV, anti=False,
                     backend="cayley", nsi=None, v_nc=None, n_active=None,
                     parallel=None):
    """Total evolution operator through a stack of constant-density segments.

    Parameters
    ----------
    v_eV, length_invEV : array_like, shape (n_seg,)
        Charged-current potential (eV) and length (eV^-1) of each segment, ordered
        from the *source* to the *detector* (segment 0 is applied first).
    v_nc : array_like, shape (n_seg,), optional
        Per-segment sterile-sector potential; defaults to zero (purely active).
    n_active : int, optional
        Number of active flavors (see :func:`mango.hamiltonian.matter_hamiltonian`).
    parallel : bool, optional
        Product strategy (see module docstring).  Defaults to ``False`` on CPU and
        ``True`` on GPU/TPU.
    Returns the ``(N, N)`` evolution operator ``S``.

    For standard 3-flavor PDG parameters (an :class:`mango.OscParams`) without
    NSI, :func:`propagate_layers_pdg` computes the same operator several times
    faster, and can exploit a mirror-symmetric trajectory.
    """
    u = jnp.asarray(u, dtype=jnp.complex128)
    n = u.shape[0]
    if n_active is None:
        n_active = n
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    v_eV = jnp.asarray(v_eV)
    length_invEV = jnp.asarray(length_invEV)
    v_nc = jnp.zeros_like(v_eV) if v_nc is None else jnp.asarray(v_nc)

    # Antineutrino conventions of matter_hamiltonian (U -> U*, V -> -V,
    # nsi -> nsi*), applied once; the vacuum part is layer-independent and is
    # hoisted out of the per-layer loop (per layer only the cheap matter term is
    # rebuilt -- the same sum matter_hamiltonian forms).
    if anti:
        u_eff = jnp.conj(u)
        v_eV, v_nc = -v_eV, -v_nc
        nsi_eff = jnp.conj(jnp.asarray(nsi)) if nsi is not None else None
    else:
        u_eff, nsi_eff = u, nsi
    h_vac = vacuum_hamiltonian(u_eff, msq, energy_eV)

    def make_s(v_k, vnc_k, l_k):
        h = h_vac + matter_term(n, v_k, vnc_k, nsi_eff, n_active)
        return propagator(h, l_k, backend=backend)

    if not parallel:
        @jax.checkpoint  # see _remat_note
        def step(carry, seg):
            return matmul_small(make_s(*seg), carry), None

        s0 = jnp.eye(n, dtype=jnp.complex128)
        s, _ = jax.lax.scan(step, s0, (v_eV, v_nc, length_invEV))
        return s

    # parallel (GPU/TPU): vmap-build + associative scan.
    s_all = jax.vmap(make_s)(v_eV, v_nc, length_invEV)
    prods = jax.lax.associative_scan(lambda a, b: matmul_small(b, a), s_all)
    return prods[-1]


# _remat_note: the sequential scan steps are ``jax.checkpoint``-ed.  Reverse
# mode then saves only each step's inputs (the carry and the segment) and
# recomputes the layer kernel in the backward sweep, instead of writing and
# re-reading its dozens of intermediates per layer and grid point: ~1.3-1.5x
# faster gradients on CPU (and far less memory).  Values are unaffected.


# --- reduced-basis (real symmetric) fast path --------------------------------
#
# Matrices are carried entrywise -- a symmetric one as its six entries
# (00, 11, 22, 01, 02, 12), a general one as a 3x3 tuple of tuples -- so that
# under vmap every entry is one contiguous batch array and each layer is a few
# fused elementwise loops (no tiny trailing dimensions, no dot_general).

_SYM_IDX = ((0, 3, 4), (3, 1, 5), (4, 5, 2))  # (i, j) -> position in a 6-tuple


def _sym_layer(hv, v_k, l_k):
    """``exp(-i (H'_vac + v_k e_00 e_00^T) l_k)`` as a symmetric 6-tuple."""
    h00, h11, h22, h01, h02, h12 = hv
    return propagator_cayley_sym((h00 + v_k) * l_k, h11 * l_k, h22 * l_k,
                                 h01 * l_k, h02 * l_k, h12 * l_k)


def _sym_times(s, c):
    """``S @ C`` for symmetric ``S`` (6-tuple) and general ``C`` (3x3 tuples)."""
    return tuple(tuple(s[_SYM_IDX[i][0]] * c[0][j] + s[_SYM_IDX[i][1]] * c[1][j]
                       + s[_SYM_IDX[i][2]] * c[2][j] for j in range(3))
                 for i in range(3))


def _sym_full(s):
    """Symmetric 6-tuple -> 3x3 tuple of tuples."""
    return tuple(tuple(s[_SYM_IDX[i][j]] for j in range(3)) for i in range(3))


def _gram_t(r):
    """``R^T R`` (complex symmetric) of a general 3x3 tuple, as a 6-tuple."""
    def dot(i, j):
        return r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j]
    return (dot(0, 0), dot(1, 1), dot(2, 2), dot(0, 1), dot(0, 2), dot(1, 2))


def _tuple_to_matrix(c):
    return jnp.stack([jnp.stack(row, -1) for row in c], -2)


def _check_mirror(mirror, n_seg):
    """Validate a static ``mirror=(start, n_half)`` against ``n_seg`` segments."""
    if mirror is None:
        return n_seg, 0
    start, n_half = (operator.index(x) for x in mirror)
    if start < 0 or n_half < 0 or start + 2 * n_half > n_seg:
        raise ValueError(
            f"mirror=({start}, {n_half}) does not fit {n_seg} segments: need "
            "0 <= start, 0 <= n_half and start + 2*n_half <= n_seg")
    return start, n_half


def _edge_correction(hv, v_k, delta):
    """``1 - i delta H'`` (symmetric 6-tuple), ``H' = H'_vac + v_k e_00 e_00^T``:
    ``exp(-iH'(L + delta)) = exp(-iH'L) (1 - i delta H') + O(delta^2 |H'|^2)``."""
    h00, h11, h22, h01, h02, h12 = hv
    c = -1j * delta
    return (1.0 + c * (h00 + v_k), 1.0 + c * h11, 1.0 + c * h22,
            c * h01, c * h02, c * h12)


def _reduced_chain(hv, v, length, mirror=None, parallel=False):
    """Reduced-basis evolution operator ``S'`` (``(3, 3)`` complex).

    ``hv``: six entries of the real symmetric vacuum part ``H'_vac`` (eV);
    ``v``, ``length``: per-segment CC potential (eV, sign already applied for
    antineutrinos) and length (eV^-1), source -> detector.

    ``mirror=(start, n_half)`` (static): the block ``M = [start + n_half,
    start + 2 n_half)`` repeats the block ``H = [start, start + n_half)`` in
    reverse order.  ``exp(-iH'L)`` is complex symmetric, so the product over
    ``M`` is ``R^T`` with ``R`` the product over ``H``, and ``M`` is not
    propagated.  Only the *length* of M's last segment (the partner of H's first
    one, i.e. the surface shell's second crossing) is read: it may differ from
    its partner's by a rounding-level ``delta``, which is applied exactly to
    first order (:func:`_edge_correction`; it commutes with that layer).  So the
    total is ``suffix . (1 - i delta H'_start) R^T R . prefix``.
    """
    n_seg = v.shape[0]
    start, n_half = _check_mirror(mirror, n_seg)
    rest = start + 2 * n_half
    n_suf = n_seg - rest

    if parallel:
        def make_s(v_k, l_k):
            return _tuple_to_matrix(_sym_full(_sym_layer(hv, v_k, l_k)))

        def chain(sl):
            s_all = jax.vmap(make_s)(v[sl], length[sl])
            prods = jax.lax.associative_scan(lambda a, b: matmul_small(b, a), s_all)
            return prods[-1]

        if n_half == 0:
            return chain(slice(0, n_seg))
        r = chain(slice(start, start + n_half))
        edge = _edge_correction(hv, v[start], length[rest - 1] - length[start])
        total = matmul_small(_tuple_to_matrix(_sym_full(edge)),
                             matmul_small(jnp.swapaxes(r, -1, -2), r))
        if start > 0:
            total = matmul_small(total, chain(slice(0, start)))
        if n_suf > 0:
            total = matmul_small(chain(slice(rest, n_seg)), total)
        return total

    @jax.checkpoint  # see _remat_note
    def step(carry, seg):
        return _sym_times(_sym_layer(hv, *seg), carry), None

    def scan(sl, c):
        c, _ = jax.lax.scan(step, c, (v[sl], length[sl]))
        return c

    one = jnp.ones((), jnp.complex128)
    zero = jnp.zeros((), jnp.complex128)
    ident = ((one, zero, zero), (zero, one, zero), (zero, zero, one))
    if n_half == 0:
        return _tuple_to_matrix(scan(slice(0, n_seg), ident))
    # Any unmirrored layers around the block go through their own scans.  An
    # Earth chord seen from a surface detector has none (its vacuum leg is
    # applied in closed form by the caller, and the surface shell's two
    # crossings differ only by the rounding-level ``delta`` handled by the edge
    # correction); a detector underground keeps the shells at and above its
    # depth explicit on both legs.
    c = scan(slice(0, start), ident) if start > 0 else None
    r = scan(slice(start, start + n_half), ident)
    gram = _gram_t(r)  # R^T R: mirrored half times the first
    edge = _edge_correction(hv, v[start], length[rest - 1] - length[start])
    total = _sym_times(edge, _sym_full(gram) if c is None
                       else _sym_times(gram, c))
    if n_suf > 0:
        total = scan(slice(rest, n_seg), total)
    return _tuple_to_matrix(total)


def _vacuum_sym(o, kin, length):
    """Reduced-basis vacuum propagator ``O diag(exp(-i kin L)) O^T`` (symmetric
    6-tuple), exact: the vacuum eigenvalues are ``kin = msq / 2E`` themselves."""
    ph = jax.lax.complex(jnp.cos(kin * length), -jnp.sin(kin * length))
    op = o * ph

    def e(i, j):
        return op[i, 0] * o[j, 0] + op[i, 1] * o[j, 1] + op[i, 2] * o[j, 2]
    return (e(0, 0), e(1, 1), e(2, 2), e(0, 1), e(0, 2), e(1, 2))


def _pdg_reduced_setup(params, energy_eV, anti):
    """``W`` (``W*`` for antineutrinos), ``O``, ``kin = msq/2E`` and the six
    entries of the real symmetric reduced vacuum Hamiltonian ``O diag(kin) O^T``."""
    w, o = pmns_3flavor_factors(params.theta12, params.theta13,
                                params.theta23, params.deltacp)
    if anti:
        w = jnp.conj(w)
    msq = jnp.asarray(params.msquared())
    kin = msq / (2.0 * energy_eV)
    ok = o * kin

    def h(i, j):
        return ok[i, 0] * o[j, 0] + ok[i, 1] * o[j, 1] + ok[i, 2] * o[j, 2]
    hv = (h(0, 0), h(1, 1), h(2, 2), h(0, 1), h(0, 2), h(1, 2))
    return w, o, kin, hv


def _propagate_pdg(params, energy_eV, v_eV, length_invEV, anti=False,
                   mirror=None, parallel=None, mass_states=False,
                   vacuum_first=False):
    """Reduced-basis amplitude: ``S`` (flavor basis, ``W S' W^dagger``) or, with
    ``mass_states``, ``W S' O`` -- equal to ``S U`` (``S U*`` for
    antineutrinos) up to a phase per column, i.e. the same ``|.|^2``.

    ``vacuum_first`` (static): segment 0 is vacuum (``v_eV[0] == 0``, e.g. the
    atmospheric leg of an Earth chord) and is applied in closed form,
    :func:`_vacuum_sym` (for ``mass_states`` it only rephases the columns and is
    dropped); ``mirror`` then refers to the remaining segments ``1:``.
    """
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    w, o, kin, hv = _pdg_reduced_setup(params, energy_eV, anti)
    v = jnp.asarray(v_eV)
    v = -v if anti else v
    length = jnp.asarray(length_invEV)
    if vacuum_first:
        v, l_vac, length = v[1:], length[0], length[1:]
    sp = _reduced_chain(hv, v, length, mirror=mirror, parallel=parallel)
    ws = matmul_small(w, sp)
    if mass_states:
        return matmul_small(ws, o.astype(jnp.complex128))
    if vacuum_first:
        ws = matmul_small(ws, _tuple_to_matrix(_sym_full(_vacuum_sym(o, kin, l_vac))))
    return matmul_small(ws, jnp.conj(w).T)


def _constant_pdg(params, energy_eV, v_eV, length_invEV, anti=False):
    """One constant-density segment in the reduced basis, ``W exp(-iH'L) W^dag``
    (the ``cayley`` operator of ``probability_constant`` for PDG parameters)."""
    w, _, _, hv = _pdg_reduced_setup(params, energy_eV, anti)
    v = -v_eV if anti else v_eV
    s = _tuple_to_matrix(_sym_full(_sym_layer(hv, v, length_invEV)))
    return matmul_small(matmul_small(w, s), jnp.conj(w).T)


def propagate_layers_pdg(params, energy_eV, v_eV, length_invEV, anti=False,
                         mirror=None, parallel=None):
    """:func:`propagate_layers` for standard 3-flavor PDG mixing, in the reduced
    real basis (see module docstring) -- same operator, several times faster.

    Parameters
    ----------
    params : mango.OscParams
        Standard 3-flavor parameters (its ``theta12, theta13, theta23, deltacp``
        and ``msquared()`` are used; ``U = params.pmns()``).  No NSI / steriles:
        use :func:`propagate_layers` for those.
    energy_eV : float
        Neutrino energy (eV).
    v_eV, length_invEV : array_like, shape (n_seg,)
        Charged-current potential (eV) and length (eV^-1) per segment, ordered
        source -> detector, exactly as for :func:`propagate_layers`.
    anti : bool
        Antineutrinos (``U -> U*``, ``V -> -V``).
    mirror : (int, int), optional
        Static ``(start, n_half)`` declaring the trajectory mirror-symmetric
        there: segment ``start + n_half + i`` is the same layer as segment
        ``start + n_half - 1 - i`` (``i < n_half``) -- as the ascending and
        descending crossings of the Earth's shells are, seen from a detector at
        the surface.  Only the first block is propagated (``exp(-iH'L)`` is
        complex symmetric, so the second block's product is the transpose of
        the first's), which halves the work.  The second block's entries are
        **not read**, except the length of its last segment (the second
        crossing of the shell entered first): an Earth chord's two crossings of
        its surface shell differ by rounding (<~1e-10 km; the ascending one is
        clamped at the detector), and that difference is applied exactly, to
        first order (it commutes with the layer).  Derivatives w.r.t. the unread
        entries are zero; for chord segments their geometry inputs reach the
        result through the first block's (bit-identical) copies instead.  The
        declaration is not checked: if the block is not a mirror image, the
        result is simply wrong.  Example: the Earth part ``[1:]`` of
        ``layered_chord_segments(model, cz, h_atm_km=h, det_depth_km=0.0)``
        (``n`` descending then ``n`` ascending crossings) is ``mirror=(0, n)``;
        the full list, atmospheric segment 0 included, ``mirror=(1, n)``.  Not
        for ``det_depth_km > 0``, where the shells at and above the detector are
        crossed differently by the two legs (:func:`mango.oscillator.
        probability_earth` keeps those explicit).
    parallel : bool, optional
        Product strategy, as for :func:`propagate_layers`.

    Returns the ``(3, 3)`` flavor-basis evolution operator, equal to
    ``propagate_layers(params.pmns(), params.msquared(), energy_eV, v_eV,
    length_invEV, anti=anti)`` to floating-point precision.
    """
    return _propagate_pdg(params, energy_eV, v_eV, length_invEV, anti=anti,
                          mirror=mirror, parallel=parallel)
