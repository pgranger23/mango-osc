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
"""

from __future__ import annotations

import jax


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
