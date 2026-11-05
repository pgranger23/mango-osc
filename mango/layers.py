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

import numpy as np
import jax
import jax.numpy as jnp

from ._small import materialize, matmul_small, sincos
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


def _identity(shape):
    one = jnp.ones(shape, jnp.complex128)
    zero = jnp.zeros(shape, jnp.complex128)
    return ((one, zero, zero), (zero, one, zero), (zero, zero, one))


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
    sn, cs = sincos(kin * length)
    op = o * materialize(jax.lax.complex(cs, -sn))

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
    l_vac = None
    if vacuum_first:
        v, l_vac, length = v[1:], length[0], length[1:]
    sp = _reduced_chain(hv, v, length, mirror=mirror, parallel=parallel)
    return _finish_pdg(w, o, kin, sp, l_vac, mass_states)


def _finish_pdg(w, o, kin, sp, l_vac, mass_states):
    """``W S' W^dagger`` (with the closed-form vacuum leg ``l_vac`` applied first
    unless ``None``) or ``W S' O`` for ``mass_states``; one point."""
    ws = matmul_small(w, sp)
    if mass_states:
        return matmul_small(ws, o.astype(jnp.complex128))
    if l_vac is not None:
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
                         mirror=None, parallel=None, mass_states=False):
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
    mass_states : bool
        Return the amplitude from the vacuum mass states, ``W S' O`` (see
        :func:`mango.pmns.pmns_3flavor_factors`; ``W*`` for antineutrinos),
        instead of the flavor-basis operator.  It equals ``S U`` (``S U*``)
        up to the phase ``exp(-+i delta)`` of its third column (``U = W O
        D^dagger``), and saves the final product with ``W^dagger`` plus the
        caller's product with ``U``: ``|.|^2`` is ``P(nu_i -> nu_beta)``, and in
        interference sums the phase cancels against ``W O`` used in place of
        ``U`` (e.g. ``sum_k A_bk conj(U'_ak)`` with ``A = W S' O``, ``U' = W
        O``, is ``S_ba``).

    Returns the ``(3, 3)`` flavor-basis evolution operator, equal to
    ``propagate_layers(params.pmns(), params.msquared(), energy_eV, v_eV,
    length_invEV, anti=anti)`` to floating-point precision (or, with
    ``mass_states``, ``W S' O`` as above).
    """
    return _propagate_pdg(params, energy_eV, v_eV, length_invEV, anti=anti,
                          mirror=mirror, parallel=parallel,
                          mass_states=mass_states)


# --- many chords at once: unreached shells skipped, chains packed ------------
#
# On an (energy, zenith) grid most chords reach only the outer shells: shells
# below the turning point r_min = r_det sin(zenith) (and, going down, everything
# below the detector) get *exactly* zero length -- ``chord_segments`` clamps
# them -- and their layer is exactly the identity (zero M: degenerate guards,
# S = 1 entrywise, and multiplying by it is exact).  On a uniform full-sky grid
# with PREM n_sub=4 that is ~85% of all layer evaluations.  Skipping them for a
# *concrete* zenith grid (known at trace time) gives every chord its own chain
# length, which a single ``lax.scan`` (one shape for the whole batch) cannot
# express directly; the chains are therefore *packed*: each "virtual row" of
# the scan runs several chords' chains one after the other, the running product
# being reset to the identity where a new chain starts (a segmented scan), and
# each chain's product is read off at its last slot.  One scan, as before (no
# extra compile time), over ~total work / max length virtual rows; the per-chord
# product is the same sequence of layer products as before, minus identities.


def _times(a, b):
    """``A @ B`` for general 3x3 tuples (entrywise, accumulated as matmul_small)."""
    return tuple(tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
                       for j in range(3)) for i in range(3))


def _pack(lengths):
    """Best-fit-decreasing packing of chain lengths into bins of capacity
    ``max(lengths)``.  Returns ``(capacity, n_bins, place)`` with ``place[j] =
    (bin, first slot)`` (``(-1, -1)`` for empty chains).  Static (numpy)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    cap = max(int(lengths.max(initial=0)), 1)
    by_room = [[] for _ in range(cap + 1)]  # remaining room -> bin ids
    fill = []
    place = np.full((len(lengths), 2), -1, dtype=np.int64)
    for j in np.argsort(-lengths, kind="stable"):
        n = int(lengths[j])
        if n == 0:
            continue
        for room in range(n, cap + 1):
            if by_room[room]:
                b = by_room[room].pop()
                break
        else:
            b, room = len(fill), cap
            fill.append(0)
        place[j] = (b, fill[b])
        fill[b] += n
        by_room[room - n].append(b)
    return cap, len(fill), place


def _packed_products(hv, v, length, jobs, parallel):
    """Products ``S_{k_m} ... S_{k_1}`` of the layer chains ``jobs`` (a list of
    ``(row, [k_1, ..., k_m])``; segment indices into ``v[row]``,
    ``length[row]``), packed into one scan (see the section note).

    ``v``, ``length``: ``(n_rows, n_seg)``; ``hv`` entries broadcast against
    ``(n_rows, 1)`` (e.g. one per energy, shape ``(n_E,)``).  Returns a 3x3
    tuple of ``(n_jobs,) + batch`` arrays; empty chains give the identity.
    """
    n_rows, n_seg = v.shape
    lengths = np.array([len(k) for _, k in jobs], dtype=np.int64)
    cap, n_bins, place = _pack(lengths)
    if n_bins == 0:  # nothing to propagate
        shape = (len(jobs),) + jnp.broadcast_shapes(jnp.shape(hv[0]), (1, 1))[1:]
        return _identity(shape)
    pad = n_rows * n_seg  # index of the appended zero-length padding layer
    src = np.full((cap, n_bins), pad, dtype=np.int64)
    reset = np.zeros((cap, n_bins), dtype=bool)
    end_t = np.zeros(len(jobs), dtype=np.int64)
    end_b = np.zeros(len(jobs), dtype=np.int64)
    for j, (row, ks) in enumerate(jobs):
        if not ks:
            continue
        b, t0 = place[j]
        src[t0:t0 + len(ks), b] = row * n_seg + np.asarray(ks)
        reset[t0, b] = True
        end_t[j], end_b[j] = t0 + len(ks) - 1, b
    zero = jnp.zeros((1,), v.dtype)
    v_slot = jnp.concatenate([v.reshape(-1), zero])[src][..., None]  # (cap, bins, 1)
    l_slot = jnp.concatenate([length.reshape(-1), zero.astype(length.dtype)])[src][..., None]
    flags = jnp.asarray(reset)[..., None]
    batch = jnp.broadcast_shapes(jnp.shape(hv[0]), (n_bins, 1))
    if parallel:
        # segmented parallel prefix: (f, A) . (g, B) = (f | g, g ? B : B A)
        def comb(a, b):
            prod = _times(b[1], a[1])
            return (a[0] | b[0], tuple(tuple(jnp.where(b[0], b[1][i][j], prod[i][j])
                                             for j in range(3)) for i in range(3)))
        s_all = _sym_full(_sym_layer(hv, v_slot, l_slot))
        _, ys = jax.lax.associative_scan(comb, (flags, s_all), axis=0)
    else:
        ident = _identity(batch)

        @jax.checkpoint  # see _remat_note
        def step(carry, seg):
            v_t, l_t, r_t = seg
            c = tuple(tuple(jnp.where(r_t, ident[i][j], carry[i][j]) for j in range(3))
                      for i in range(3))
            new = _sym_times(_sym_layer(hv, v_t, l_t), c)
            return new, new
        _, ys = jax.lax.scan(step, ident, (v_slot, l_slot, flags))

    empty = jnp.asarray(lengths == 0).reshape((-1,) + (1,) * (len(batch) - 1))
    ident = _identity(())
    return tuple(tuple(jnp.where(empty, ident[i][j], ys[i][j][end_t, end_b])
                       for j in range(3)) for i in range(3))


def _reduced_chain_rows(hv, v, length, trim, mirror=None, parallel=False):
    """:func:`_reduced_chain` for many chords at once, skipping their unreached
    shells (see the section note).

    ``v``, ``length``: ``(n_rows, n_seg)``, each row an Earth chord's segments
    (``n = n_seg / 2`` shells descending, then the same ascending; no vacuum
    leg); ``hv`` entries broadcast against ``(n_rows, 1)``.  ``trim`` (static,
    ``(n_rows,)`` ints): the ``2 trim[i]`` segments around row ``i``'s midpoint
    (its innermost shells, both legs) have exactly zero length and are not
    propagated.  ``mirror``: as for :func:`_reduced_chain`, centred on the
    midpoint (``start + n_half == n``); ``trim[i] <= n_half`` then.  Returns the
    3x3 tuple of ``(n_rows,) + batch`` arrays.
    """
    n_rows, n_seg = v.shape
    half = n_seg // 2
    start, n_half = _check_mirror(mirror, n_seg)
    trim = np.asarray(trim, dtype=np.int64)
    if n_seg % 2 or trim.shape != (n_rows,) or np.any(trim < 0) or (
            n_half and (start + n_half != half or np.any(trim > n_half))) or (
            np.any(trim > half)):
        raise ValueError("trim/mirror do not describe chords of this shape")
    rows = range(n_rows)
    if not n_half:
        jobs = [(i, list(range(0, half - trim[i])) + list(range(half + trim[i], n_seg)))
                for i in rows]
        return _packed_products(hv, v, length, jobs, parallel)
    rest = start + 2 * n_half
    jobs = [(i, list(range(start, half - trim[i]))) for i in rows]   # H blocks
    if start:
        jobs += [(i, list(range(0, start))) for i in rows]           # prefixes
    prods = _packed_products(hv, v, length, jobs, parallel)
    r = tuple(tuple(x[:n_rows] for x in row) for row in prods)
    ex = (slice(None),) + (None,) * (len(jnp.shape(r[0][0])) - 1)   # (n_rows, 1, ...)
    edge = _edge_correction(hv, v[:, start][ex],
                            (length[:, rest - 1] - length[:, start])[ex])
    gram = _gram_t(r)  # R^T R: mirrored half times the first
    if start:
        c = tuple(tuple(x[n_rows:] for x in row) for row in prods)
        total = _sym_times(edge, _sym_times(gram, c))
    else:
        total = _sym_times(edge, _sym_full(gram))
    if rest < n_seg:  # suffix (detector underground): applied to the total
        seg = (v[:, rest:].T[(slice(None),) + ex], length[:, rest:].T[(slice(None),) + ex])
        if parallel:
            def make_s(v_k, l_k):
                return _sym_full(_sym_layer(hv, v_k, l_k))
            s_all = make_s(*seg)
            suf = jax.lax.associative_scan(lambda a, b: _times(b, a), s_all, axis=0)
            total = _times(tuple(tuple(x[-1] for x in row) for row in suf), total)
        else:
            @jax.checkpoint  # see _remat_note
            def step(carry, sg):
                return _sym_times(_sym_layer(hv, *sg), carry), None
            total, _ = jax.lax.scan(step, total, seg)
    return total


def _propagate_pdg_rows(params, energy_eV, v_eV, length_invEV, trim, anti=False,
                        mirror=None, parallel=None, mass_states=False,
                        vacuum_first=False):
    """:func:`_propagate_pdg` for ``n_rows`` chords (``v_eV``, ``length_invEV``:
    ``(n_rows, n_seg)``) times energies (scalar or 1-D), skipping each chord's
    ``trim`` innermost (zero-length) shells (static; see
    :func:`_reduced_chain_rows`).  Returns ``(n_rows, n_E, 3, 3)`` (``(n_rows,
    3, 3)`` for a scalar energy) -- the values of ``_propagate_pdg`` per (row,
    energy).
    """
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    e = jnp.asarray(energy_eV)
    w, o = pmns_3flavor_factors(params.theta12, params.theta13,
                                params.theta23, params.deltacp)
    w = jnp.conj(w) if anti else w  # as _pdg_reduced_setup
    kin, hv = jax.vmap(lambda x: _pdg_reduced_setup(params, x, anti)[2:])(e.reshape(-1))
    v = jnp.asarray(v_eV)
    v = -v if anti else v
    length = jnp.asarray(length_invEV)
    l_vac = None
    if vacuum_first:
        v, l_vac, length = v[:, 1:], length[:, 0], length[:, 1:]
    sp = _tuple_to_matrix(_reduced_chain_rows(hv, v, length, trim, mirror=mirror,
                                              parallel=parallel))  # (rows, n_E, 3, 3)

    def row(sp_r, lv):
        return jax.vmap(lambda s, k: _finish_pdg(w, o, k, s, lv, mass_states))(sp_r, kin)
    out = (jax.vmap(row)(sp, l_vac) if l_vac is not None
           else jax.vmap(lambda sp_r: row(sp_r, None))(sp))
    return out if e.ndim else out[:, 0]
