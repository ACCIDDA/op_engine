"""Steady states of autonomous ODE right-hand sides.

:func:`steady_state` finds ``y*`` with ``rhs(t, y*) = 0`` by pseudo-transient
continuation (PTC). Each iteration solves the implicit-Euler step

    (I / dt - J) delta = rhs(t, y)

and grows ``dt`` by the ratio of successive residuals, so early iterations
follow the dynamics and later ones become Newton steps. Three features of
epidemic models need handling (issue #192):

- **Absorbing states.** Cumulative counters (deaths, incidence) never reach a
  zero derivative. Pass them as ``fixed``: they keep their initial value and
  are excluded from the convergence norms. :func:`sink_states` finds states
  that no derivative depends on.
- **Conserved totals.** When ``w @ rhs(t, y) = 0`` for every ``y``, the total
  ``w @ y`` is conserved, the steady states form a family indexed by it, and
  ``J`` is singular. As ``dt`` grows, ``I / dt - J`` becomes ill conditioned
  and the total drifts. Pass the conserved directions as ``invariants``: the
  step solves a bordered system that holds ``W @ y`` at its initial value
  exactly and stays nonsingular as ``dt`` grows without bound.
  :func:`conserved_quantities` gives the structural invariants of a
  stoichiometry matrix; :func:`linear_invariants` also finds totals that are
  conserved because rates balance (births ``mu * N``), which the
  stoichiometry alone misses.
- **Slow modes.** A small residual can hide a large state error along a slowly
  decaying mode (error ~ residual / rate). Convergence therefore also requires
  a small Newton correction ``-J^{-1} rhs``, which measures that error, and the
  final correction is applied to the returned state.

The iteration count is static and every diagnostic stays an array, so the
solver runs under ``jax.jit`` and ``jax.vmap``. Pass
``loop=jax.lax.fori_loop`` there, so the iteration is compiled once rather
than unrolled.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

from ._array import array_namespace as _namespace_of
from ._typing import Array

SteadyStateRhs = Callable[[float, Array], Array]
SteadyStateJacobian = Callable[[float, Array], Array]
_Carry = tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]
#: ``loop(lower, upper, body, carry)``, with ``body(index, carry) -> carry``;
#: ``jax.lax.fori_loop`` has this signature.
SteadyStateLoop = Callable[[int, int, Callable[[int, _Carry], _Carry], _Carry], _Carry]


@dataclass(slots=True, frozen=True)
class SteadyStateConfig:
    """Controls for :func:`steady_state`.

    Attributes:
        dt0: Initial pseudo-time step, in the RHS's time units.
        dt_max: Largest pseudo-time step.
        min_growth: Smallest factor by which ``dt`` grows when the unscaled
            residual did not increase. Without it, a slowly decaying mode
            barely shrinks the residual and ``dt`` stalls. ``dt`` is held
            while the residual grows by less than this factor per step, as
            it does while an epidemic takes off, and shrinks beyond that.
        max_growth: Largest factor by which ``dt`` grows, or shrinks, in one
            iteration. Between the two bounds ``dt`` follows the ratio of
            successive unscaled residual norms (switched evolution
            relaxation).
        max_iterations: Iteration budget; the loop is unrolled this many times
            when ``early_exit`` is false.
        residual_tol: Bound on the RMS of ``rhs / (atol + |y|)`` over the
            unknowns, in inverse time units.
        step_tol: Bound on the RMS of the Newton correction
            ``(I / dt_max - J)^{-1} rhs / (atol + |y|)``, a relative
            state-error estimate.
        atol: Absolute floor of the per-state scale ``atol + |y|``.
        early_exit: With the default Python loop, stop as soon as the solve
            converges. This converts the convergence flag to a host boolean,
            so under ``jax.jit`` or ``jax.vmap`` either set it to ``False`` or,
            better, pass ``loop=jax.lax.fori_loop``.
    """

    dt0: float = 1.0
    dt_max: float = 1e12
    min_growth: float = 2.0
    max_growth: float = 10.0
    max_iterations: int = 100
    residual_tol: float = 1e-9
    step_tol: float = 1e-9
    atol: float = 1e-8
    early_exit: bool = True

    def __post_init__(self) -> None:
        """Validate the controls.

        Raises:
            ValueError: If a control is outside its valid range.
        """
        for name in ("dt0", "dt_max", "residual_tol", "step_tol", "atol"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                msg = f"{name} must be finite and positive"
                raise ValueError(msg)
        if self.dt_max < self.dt0:
            msg = "dt_max must be at least dt0"
            raise ValueError(msg)
        for name in ("min_growth", "max_growth"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 1.0
            ):
                msg = f"{name} must be finite and at least one"
                raise ValueError(msg)
        if self.max_growth < self.min_growth:
            msg = "max_growth must be at least min_growth"
            raise ValueError(msg)
        if (
            not isinstance(self.max_iterations, Integral)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 1
        ):
            msg = "max_iterations must be a positive integer"
            raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class SteadyStateResult:
    """Candidate steady state and array-valued diagnostics.

    Every diagnostic is a zero-dimensional array in the state's namespace, so
    a traced solve returns them without a host conversion; check them
    afterwards, or call :func:`require_steady_state` at an eager boundary.

    Attributes:
        state: The candidate steady state, including fixed states.
        residual: ``rhs(t, state)``, zero at fixed states.
        converged: Whether both the residual and the Newton correction met
            their tolerances.
        iterations: PTC iterations performed before convergence.
        residual_norm: Scaled residual RMS at ``state``.
        step_norm: Scaled RMS of the last Newton correction.
        dt: Final pseudo-time step.
        invariant_drift: Largest ``|W @ state - W @ y0|`` relative to
            ``1 + |W @ y0|``; zero without invariants.
    """

    state: Array
    residual: Array
    converged: Array
    iterations: Array
    residual_norm: Array
    step_norm: Array
    dt: Array
    invariant_drift: Array


class SteadyStateConvergenceError(RuntimeError):
    """Raised at an eager boundary when a steady-state solve did not converge."""

    def __init__(self, result: SteadyStateResult) -> None:
        """Store the failed result."""
        self.result = result
        super().__init__(
            "Steady-state solve did not converge after "
            f"{int(result.iterations.item())} iterations; scaled residual "
            f"{float(result.residual_norm.item())!r}, Newton correction "
            f"{float(result.step_norm.item())!r}"
        )


def require_steady_state(result: SteadyStateResult) -> SteadyStateResult:
    """Return a converged result or raise.

    This converts the convergence flag to a Python boolean; do not call it
    inside ``jax.jit``.

    Args:
        result: Result to check.

    Returns:
        ``result`` unchanged.

    Raises:
        SteadyStateConvergenceError: If the solve did not converge.
    """
    if not bool(result.converged.item()):
        raise SteadyStateConvergenceError(result)
    return result


def _finite_difference_jacobian(rhs: SteadyStateRhs) -> SteadyStateJacobian:
    """Build a forward-difference dense Jacobian of ``rhs``.

    Returns:
        A callback costing one RHS evaluation per state.
    """

    def jacobian(t: float, y: Array) -> Array:
        xp = _namespace_of(y)
        n = y.shape[0]
        base = rhs(t, y)
        eps = math.sqrt(float(xp.finfo(y.dtype).eps))
        steps = xp.multiply(eps, xp.maximum(xp.abs(y), xp.ones_like(y)))
        eye = xp.eye(n, dtype=y.dtype)
        columns = [
            xp.divide(
                xp.subtract(rhs(t, xp.add(y, xp.multiply(steps[j], eye[j]))), base),
                steps[j],
            )
            for j in range(n)
        ]
        return cast("Array", xp.stack(columns, axis=1))

    return jacobian


def _central_difference_jacobian(rhs: SteadyStateRhs) -> SteadyStateJacobian:
    """Build a central-difference dense Jacobian of ``rhs`` on the host.

    It costs two RHS evaluations per state but is about a thousand times
    more accurate than forward differences, which matters for separating a
    truly conserved direction from a slowly decaying one.

    Returns:
        A callback returning a NumPy array.
    """

    def jacobian(t: float, y: Array) -> Array:
        values = np.asarray(y, dtype=np.float64)
        steps = np.cbrt(np.finfo(np.float64).eps) * np.maximum(np.abs(values), 1.0)
        eye = np.eye(values.shape[0])
        columns = [
            (
                np.asarray(rhs(t, cast("Array", values + h * e)), dtype=np.float64)
                - np.asarray(rhs(t, cast("Array", values - h * e)), dtype=np.float64)
            )
            / (2.0 * h)
            for h, e in zip(steps, eye, strict=True)
        ]
        return cast("Array", np.stack(columns, axis=1))

    return jacobian


def _free_mask(fixed: object, n: int) -> np.ndarray:
    """Return a host boolean mask of the unknowns.

    Raises:
        ValueError: If ``fixed`` has the wrong length or out-of-range indices.
    """
    if fixed is None:
        return np.ones(n, dtype=np.bool_)
    values = np.asarray(fixed)
    if values.dtype == np.bool_:
        if values.shape != (n,):
            msg = f"fixed mask shape {values.shape} does not match ({n},)"
            raise ValueError(msg)
        mask = values
    else:
        indices = values.astype(np.int64).ravel()
        if np.any((indices < 0) | (indices >= n)):
            msg = "fixed indices must lie in [0, n_state)"
            raise ValueError(msg)
        mask = np.zeros(n, dtype=np.bool_)
        mask[indices] = True
    if mask.all():
        msg = "fixed must leave at least one unknown"
        raise ValueError(msg)
    # Annotated so NumPy < 2.5 stubs, where logical_not returns Any, type-check.
    free: np.ndarray = np.logical_not(mask)
    return free


def _invariant_matrix(invariants: object, n: int, xp: Any, dtype: object) -> Any:  # noqa: ANN401
    """Return invariant rows as a ``(k, n)`` array, or ``None``.

    Raises:
        ValueError: If the rows have the wrong width or are linearly dependent.
    """
    if invariants is None:
        return None
    rows = np.atleast_2d(np.asarray(invariants, dtype=np.float64))
    if rows.shape[0] == 0:
        return None
    if rows.ndim != 2 or rows.shape[1] != n:
        msg = f"invariants must have shape (k, {n}); got {rows.shape}"
        raise ValueError(msg)
    if np.linalg.matrix_rank(rows) < rows.shape[0]:
        msg = "invariants must be linearly independent"
        raise ValueError(msg)
    return xp.asarray(rows, dtype=dtype)


def _scaled_rms(values: Any, scale: Any, free: Any, n_free: int, xp: Any) -> Any:  # noqa: ANN401
    """Return the RMS of ``values / scale`` over the unknowns.

    Returns:
        Scalar array.
    """
    ratio = xp.divide(values, scale)
    squared = xp.where(free, xp.multiply(ratio, ratio), xp.zeros_like(ratio))
    return xp.sqrt(xp.divide(xp.sum(squared), n_free))


def steady_state(  # noqa: PLR0913, PLR0914, PLR0915
    rhs: SteadyStateRhs,
    y0: Array,
    *,
    t: float = 0.0,
    jacobian: SteadyStateJacobian | None = None,
    fixed: Sequence[int] | Array | None = None,
    invariants: Array | Sequence[Sequence[float]] | None = None,
    config: SteadyStateConfig | None = None,
    loop: SteadyStateLoop | None = None,
) -> SteadyStateResult:
    """Find a steady state of ``rhs`` near ``y0`` by pseudo-transient continuation.

    Args:
        rhs: ``rhs(t, y) -> dy/dt`` for a one-dimensional state ``y``, such as
            ``lambda t, y: compiled.eval_fn(t, y, **params)``.
        y0: Starting state. It also fixes the absorbing states' values and the
            invariant totals.
        t: Time at which the autonomous RHS is evaluated.
        jacobian: ``jacobian(t, y) -> (n, n)`` dense array. Defaults to forward
            differences (``n`` RHS evaluations per iteration); pass
            ``lambda t, y: jax.jacfwd(lambda z: rhs(t, z))(y)`` under JAX.
        fixed: Indices, or a boolean mask, of states held at ``y0``.
        invariants: ``(k, n)`` linearly independent rows ``W`` whose totals
            ``W @ y`` the dynamics conserve; held at ``W @ y0``. They are
            setup data, checked on the host, so they must be concrete values
            (not traced inside ``jax.jit``).
        config: Iteration controls.
        loop: Optional ``loop(lower, upper, body, carry)`` driver for the
            iterations, such as ``jax.lax.fori_loop``. Under ``jax.jit`` it
            compiles the iteration once instead of unrolling
            ``max_iterations`` copies; it runs every iteration, ignoring
            ``early_exit``. ``None`` uses a Python loop.

    Returns:
        The candidate steady state and diagnostics.

    Raises:
        ValueError: If shapes are inconsistent.
        TypeError: If ``y0`` is not a floating-point vector.
    """
    config = config or SteadyStateConfig()
    xp = _namespace_of(y0)
    if len(y0.shape) != 1 or y0.shape[0] < 1:
        msg = f"y0 must be a non-empty vector; got shape {y0.shape}"
        raise ValueError(msg)
    if not xp.isdtype(y0.dtype, "real floating"):
        msg = "y0 must have a real floating-point dtype"
        raise TypeError(msg)
    n = int(y0.shape[0])
    dtype = y0.dtype
    jacobian = jacobian or _finite_difference_jacobian(rhs)
    free_host = _free_mask(fixed, n)
    n_free = int(free_host.sum())
    free = xp.asarray(free_host)
    w = _invariant_matrix(invariants, n, xp, dtype)
    k = 0 if w is None else int(w.shape[0])
    totals = None if w is None else xp.matmul(w, y0)

    eye = xp.eye(n, dtype=dtype)
    free_pair = xp.logical_and(xp.reshape(free, (n, 1)), xp.reshape(free, (1, n)))
    inv_dt_max = xp.asarray(1.0 / config.dt_max, dtype=dtype)

    def masked_rhs(y: Any) -> Any:  # noqa: ANN401
        return xp.where(free, rhs(t, y), xp.zeros_like(y))

    def step(y: Any, value: Any, jac: Any, inv_dt: Any) -> Any:  # noqa: ANN401
        """Solve ``(I/dt - J) delta = rhs`` with fixed rows and invariants.

        Returns:
            The step ``delta`` for the state.
        """
        matrix = xp.where(free_pair, xp.subtract(xp.multiply(eye, inv_dt), jac), eye)
        if w is None:
            return xp.linalg.solve(matrix, value)
        bordered = xp.concat(
            (
                xp.concat((matrix, xp.matrix_transpose(w)), axis=1),
                xp.concat((w, xp.zeros((k, k), dtype=dtype)), axis=1),
            ),
            axis=0,
        )
        target = xp.concat((value, xp.subtract(totals, xp.matmul(w, y))))
        return xp.linalg.solve(bordered, target)[:n]

    value0 = masked_rhs(y0)
    scale0 = xp.add(config.atol, xp.abs(y0))
    initial: _Carry = (
        y0,
        value0,
        scale0,
        _scaled_rms(value0, scale0, free, n_free, xp),
        xp.sqrt(xp.mean(xp.multiply(value0, value0))),
        xp.asarray(np.inf, dtype=dtype),
        xp.asarray(config.dt0, dtype=dtype),
        xp.zeros((), dtype=xp.bool),
        xp.zeros((), dtype=xp.int32),
    )

    def iterate(_index: int, carry: _Carry) -> _Carry:  # noqa: PLR0914
        """Advance one PTC iteration; a converged carry passes through.

        Returns:
            The next carry.
        """
        (y, value, scale, residual_norm, raw_norm, _step_norm, dt, converged,
         iterations) = carry  # fmt: skip
        jac = jacobian(t, y)
        # The Newton correction is taken at dt_max rather than an infinite
        # step: identical for every mode faster than 1/dt_max, and finite
        # (but huge, so unconverged) along a conserved direction that was
        # not declared, where pure Newton would be singular.
        newton = step(y, value, jac, inv_dt_max)
        ptc = step(y, value, jac, xp.divide(1.0, dt))
        newton_norm = _scaled_rms(newton, scale, free, n_free, xp)
        done_now = xp.logical_and(
            xp.logical_and(
                xp.less_equal(residual_norm, config.residual_tol),
                xp.less_equal(newton_norm, config.step_tol),
            ),
            xp.all(xp.isfinite(newton)),
        )
        # A converged iterate takes its Newton correction as a final polish.
        candidate = xp.add(y, xp.where(done_now, newton, ptc))
        candidate_value = masked_rhs(candidate)
        candidate_scale = xp.add(config.atol, xp.abs(candidate))
        candidate_raw = xp.sqrt(xp.mean(xp.multiply(candidate_value, candidate_value)))
        positive = xp.greater(candidate_raw, 0)
        ratio = xp.where(
            positive,
            # Both branches are evaluated, so keep the denominator nonzero.
            xp.divide(raw_norm, xp.where(positive, candidate_raw, 1.0)),
            xp.asarray(config.max_growth, dtype=dtype),
        )
        # Grow while the residual falls; hold through moderate rises, which
        # ordinary dynamics produce (an epidemic taking off); shrink only when
        # the residual jumps by more than min_growth in one step.
        factor = xp.where(
            xp.greater_equal(ratio, 1.0),
            xp.clip(ratio, config.min_growth, config.max_growth),
            xp.where(
                xp.greater_equal(ratio, 1.0 / config.min_growth),
                xp.ones_like(ratio),
                xp.maximum(ratio, 1.0 / config.max_growth),
            ),
        )
        updated: _Carry = (
            candidate,
            candidate_value,
            candidate_scale,
            _scaled_rms(candidate_value, candidate_scale, free, n_free, xp),
            candidate_raw,
            newton_norm,
            xp.minimum(xp.multiply(dt, factor), config.dt_max),
            xp.logical_or(converged, done_now),
            xp.add(iterations, xp.ones_like(iterations)),
        )
        held = (
            xp.where(converged, old, new)
            for old, new in zip(carry[:-2], updated[:-2], strict=True)
        )
        return cast(
            "_Carry",
            (*held, updated[-2], xp.where(converged, iterations, updated[-1])),
        )

    carry = initial
    if loop is not None:
        carry = loop(0, config.max_iterations, iterate, carry)
    else:
        for index in range(config.max_iterations):
            carry = iterate(index, carry)
            if config.early_exit and bool(carry[7]):
                break
    (y, value, _, residual_norm, _, step_norm, dt, converged, iterations) = carry

    drift = (
        xp.asarray(0.0, dtype=dtype)
        if w is None
        else xp.max(
            xp.divide(
                xp.abs(xp.subtract(xp.matmul(w, y), totals)),
                xp.add(1.0, xp.abs(totals)),
            )
        )
    )
    return SteadyStateResult(
        state=y,
        residual=cast("Array", value),
        converged=cast("Array", converged),
        iterations=cast("Array", iterations),
        residual_norm=cast("Array", residual_norm),
        step_norm=cast("Array", step_norm),
        dt=cast("Array", dt),
        invariant_drift=cast("Array", drift),
    )


def _left_null_space(matrix: np.ndarray, *, tol: float) -> np.ndarray:
    """Return orthonormal rows spanning ``{w : w @ matrix = 0}``.

    Returns:
        ``(k, n)`` array.
    """
    if matrix.size == 0:
        return np.eye(matrix.shape[0])
    u, singular, _ = np.linalg.svd(matrix, full_matrices=True)
    threshold = tol * max(1.0, float(singular[0])) if singular.size else tol
    rank = int(np.sum(singular > threshold))
    return u[:, rank:].T.copy()


def conserved_quantities(stoichiometry: object, *, tol: float = 1e-10) -> np.ndarray:
    """Return the structural conservation laws of a reaction network.

    Rows ``w`` satisfy ``w @ stoichiometry = 0``, so ``w @ y`` is unchanged by
    every reaction, whatever the rates. Totals conserved only because rates
    balance (births ``mu * N`` against deaths) are not structural; use
    :func:`linear_invariants` for those.

    Args:
        stoichiometry: ``(n_state, n_reactions)`` net change matrix, such as
            ``CompiledReactionNetwork.stoichiometry``.
        tol: Relative singular-value threshold.

    Returns:
        ``(k, n_state)`` orthonormal rows, possibly empty.
    """
    return _left_null_space(np.asarray(stoichiometry, dtype=np.float64), tol=tol)


def linear_invariants(  # noqa: PLR0913
    rhs: SteadyStateRhs,
    states: Sequence[Array],
    *,
    t: float = 0.0,
    jacobian: SteadyStateJacobian | None = None,
    fixed: Sequence[int] | Array | None = None,
    tol: float = 1e-6,
) -> np.ndarray:
    """Find totals ``w @ y`` that the dynamics conserve at sampled states.

    A row ``w`` is returned when ``w @ J(y) = 0`` and ``w @ rhs(t, y) = 0`` at
    every sampled ``y``, which includes rate-balanced totals that
    :func:`conserved_quantities` misses. Sample a few varied states away from
    special points such as the disease-free state.

    The test is numerical. A mode decaying more slowly than ``tol`` times the
    fastest rate looks conserved, and each row's accuracy is limited by the
    Jacobian's error relative to the slowest real rate: a slow mode tilts the
    rows toward itself. The default Jacobian uses central differences for
    that reason; an exact one (``jax.jacfwd``, for example) is better still.
    When you know an invariant, such as a row of ones over the living
    compartments, pass that exact row to :func:`steady_state` and use this
    function to confirm that nothing else is conserved.

    Args:
        rhs: ``rhs(t, y) -> dy/dt``.
        states: Sample states, each a one-dimensional array.
        t: Evaluation time.
        jacobian: Dense Jacobian callback; central differences by default.
        fixed: Indices, or a boolean mask, of states that
            :func:`steady_state` will hold fixed. They are left out, so the
            rows have zeros there.
        tol: Singular-value threshold relative to the fastest rate.

    Returns:
        ``(k, n_state)`` orthonormal rows, possibly empty.

    Raises:
        ValueError: If ``states`` is empty.
    """
    jacobian = jacobian or _central_difference_jacobian(rhs)
    blocks: list[np.ndarray] = []
    free: np.ndarray | None = None
    for y in states:
        values = np.asarray(y, dtype=np.float64)
        if free is None:
            free = _free_mask(fixed, values.shape[0])
        jac = np.asarray(jacobian(t, y), dtype=np.float64)[np.ix_(free, free)]
        # Divide by the state's size so both blocks are rates, comparable
        # with one threshold.
        rate = np.asarray(rhs(t, y), dtype=np.float64)[free] / max(
            float(np.abs(values).max()), 1.0
        )
        blocks.extend((jac, rate.reshape(-1, 1)))
    if free is None:
        msg = "linear_invariants needs at least one sample state"
        raise ValueError(msg)
    rows = _left_null_space(np.concatenate(blocks, axis=1), tol=tol)
    embedded = np.zeros((rows.shape[0], free.shape[0]))
    embedded[:, free] = rows
    return embedded


def sink_states(
    rhs: SteadyStateRhs,
    states: Sequence[Array],
    *,
    t: float = 0.0,
    jacobian: SteadyStateJacobian | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return a mask of states no derivative depends on.

    Cumulative counters and absorbing compartments that feed nothing back have
    a zero Jacobian column. They never reach a zero derivative while their
    inflow is positive, so hold them fixed in :func:`steady_state`.

    Args:
        rhs: ``rhs(t, y) -> dy/dt``.
        states: Sample states.
        t: Evaluation time.
        jacobian: Dense Jacobian callback; central differences by default.
        tol: Largest column entry, relative to the largest Jacobian entry,
            still treated as zero.

    Returns:
        Boolean ``(n_state,)`` mask.
    """
    jacobian = jacobian or _central_difference_jacobian(rhs)
    columns = np.zeros(0)
    for y in states:
        magnitude = np.abs(np.asarray(jacobian(t, y), dtype=np.float64)).max(axis=0)
        columns = magnitude if columns.size == 0 else np.maximum(columns, magnitude)
    return columns <= tol * max(float(columns.max(initial=0.0)), 1e-300)


__all__ = [
    "SteadyStateConfig",
    "SteadyStateConvergenceError",
    "SteadyStateJacobian",
    "SteadyStateLoop",
    "SteadyStateResult",
    "SteadyStateRhs",
    "conserved_quantities",
    "linear_invariants",
    "require_steady_state",
    "sink_states",
    "steady_state",
]
