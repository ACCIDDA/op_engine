"""Steady states by pseudo-transient continuation (#192)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_engine import (
    SteadyStateConfig,
    SteadyStateConvergenceError,
    conserved_quantities,
    linear_invariants,
    require_steady_state,
    sink_states,
    steady_state,
)

if TYPE_CHECKING:
    import contextlib

    from numpy.typing import NDArray

BETA, GAMMA, MU, OMEGA = 0.4, 0.1, 1 / (70 * 365), 1 / 365
TOTAL = 5000.0


def _sirs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """SIRS with births ``mu * N`` and a cumulative death counter ``D``.

    Returns:
        ``(S, I, R, D)`` derivatives. ``S + I + R`` is conserved because
        births balance deaths, not by stoichiometry, and ``D`` never stops
        growing while ``I > 0``.
    """
    s, i, r, _ = y
    n = s + i + r
    infection = BETA * s * i / n
    return np.array([
        MU * n - infection - MU * s + OMEGA * r,
        infection - (GAMMA + MU) * i,
        GAMMA * i - (MU + OMEGA) * r,
        MU * i,
    ])


def _endemic() -> NDArray[np.float64]:
    s = TOTAL * (GAMMA + MU) / BETA
    i = (TOTAL - s) / (1 + GAMMA / (MU + OMEGA))
    return np.array([s, i, GAMMA * i / (MU + OMEGA)])


Y0 = np.array([4990.0, 10.0, 0.0, 0.0])
SINK = np.array([False, False, False, True])
TOTAL_ROW = np.array([[1.0, 1.0, 1.0, 0.0]])


def test_linear_system_reaches_its_unique_root() -> None:
    """``A y + b = 0`` has the root ``-A^{-1} b``."""
    a = np.array([[-2.0, 1.0, 0.0], [0.5, -1.0, 0.2], [0.0, 0.3, -0.5]])
    b = np.array([1.0, 2.0, 0.5])
    result = require_steady_state(steady_state(lambda _t, y: a @ y + b, np.zeros(3)))
    np.testing.assert_allclose(result.state, np.linalg.solve(a, -b), rtol=1e-12)
    assert float(result.residual_norm) <= 1e-9


def test_endemic_equilibrium_with_sink_and_conserved_total() -> None:
    """From near the disease-free state to the analytic endemic state."""
    result = require_steady_state(
        steady_state(_sirs, Y0, fixed=SINK, invariants=TOTAL_ROW)
    )
    np.testing.assert_allclose(result.state[:3], _endemic(), rtol=1e-12)
    assert result.state[3] == 0.0  # the counter is held at y0
    assert float(result.invariant_drift) < 1e-14
    assert int(result.iterations) < 60


def test_fixed_states_accept_indices_or_a_mask() -> None:
    """``fixed=[3]`` and the boolean mask hold the same state."""
    by_index = steady_state(_sirs, Y0, fixed=[3], invariants=TOTAL_ROW)
    by_mask = steady_state(_sirs, Y0, fixed=SINK, invariants=TOTAL_ROW)
    np.testing.assert_array_equal(by_index.state, by_mask.state)


def test_an_unfixed_sink_is_reported_unconverged() -> None:
    """A counter with positive inflow has no steady state; the solve says so."""
    result = steady_state(_sirs, Y0, invariants=TOTAL_ROW)
    assert not bool(result.converged)
    with pytest.raises(SteadyStateConvergenceError, match="did not converge"):
        require_steady_state(result)


def test_newton_correction_catches_a_slow_mode() -> None:
    """A small residual can hide a large error along a slow mode.

    ``y0' = -1e-7 (y0 - 3)``: a residual of 1e-9 still leaves ~1e-2 of
    error. The Newton criterion drives it out; a residual-only criterion
    stops early.
    """

    def slow(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([-1e-7 * (y[0] - 3.0), -(y[1] - 1.0)])

    strict = require_steady_state(steady_state(slow, np.zeros(2)))
    np.testing.assert_allclose(strict.state, [3.0, 1.0], rtol=1e-12)
    loose = steady_state(slow, np.zeros(2), config=SteadyStateConfig(step_tol=1e30))
    assert abs(float(loose.state[0]) - 3.0) > 1e-7


def test_structural_conservation_from_stoichiometry() -> None:
    """S -> I -> R conserves S + I + R; births and deaths break it."""
    closed = np.array([[-1, 0], [1, -1], [0, 1]])
    (row,) = conserved_quantities(closed)
    np.testing.assert_allclose(np.abs(row), np.full(3, 1 / np.sqrt(3)))
    with_births = np.concatenate([closed, [[1], [0], [0]]], axis=1)
    assert conserved_quantities(with_births).shape == (0, 3)


def test_rate_balanced_total_and_sink_are_detected() -> None:
    """``S + I + R`` is conserved by balanced rates; ``D`` feeds nothing back."""
    rng = np.random.default_rng(0)
    samples = [rng.uniform(10.0, 3000.0, 4) for _ in range(3)]
    np.testing.assert_array_equal(sink_states(_sirs, samples), SINK)
    (row,) = linear_invariants(_sirs, samples, fixed=SINK)
    np.testing.assert_allclose(np.abs(row), TOTAL_ROW[0] / np.sqrt(3.0), atol=1e-8)


def test_slow_decay_is_not_mistaken_for_an_invariant() -> None:
    """A difference decaying at a slow rate is not reported as conserved.

    Two groups ``(X_a, Y_a)`` exchange at rate 1, receive equal shares of
    births ``mu * N``, and die at ``mu``. Only the total is conserved;
    ``N_a - N_b`` decays at ``mu``, about 4e-5 of the fastest rate.
    """

    def two_groups(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        births = MU * y.sum() / 2
        x, z = y[0::2], y[1::2]
        exchange = z - x
        out = np.empty_like(y)
        out[0::2] = births + exchange - MU * x
        out[1::2] = -exchange - MU * z
        return out

    rng = np.random.default_rng(1)
    samples = [rng.uniform(10.0, 3000.0, 4) for _ in range(3)]
    (row,) = linear_invariants(two_groups, samples)
    np.testing.assert_allclose(np.abs(row), np.full(4, 0.5), atol=1e-6)


def test_two_age_demography_has_one_conserved_total() -> None:
    """Rates and RHS values are compared in the same units.

    With births shared equally and deaths at ``mu``, ``N_young - N_old``
    decays at ``mu``. Mixing the RHS (population times rate) with the
    Jacobian (rates) in one threshold used to report it as conserved.
    """

    def two_age(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        s, i, r = y[0:2], y[2:4], y[4:6]
        n = (s + i + r).sum()
        infection = BETA * s * i.sum() / n
        return np.concatenate([
            MU * n / 2 - infection - MU * s + OMEGA * r,
            infection - (GAMMA + MU) * i,
            GAMMA * i - (MU + OMEGA) * r,
        ])

    rng = np.random.default_rng(1)
    samples = [rng.uniform(10.0, 3000.0, 6) for _ in range(3)]
    (row,) = linear_invariants(two_age, samples)
    np.testing.assert_allclose(np.abs(row), np.full(6, 1 / np.sqrt(6.0)), atol=1e-7)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"fixed": [9]}, ValueError, "fixed indices", id="fixed-index"),
        pytest.param(
            {"fixed": [0, 1, 2, 3]}, ValueError, "at least one", id="all-fixed"
        ),
        pytest.param(
            {"invariants": np.ones((1, 3))}, ValueError, "shape", id="invariant-width"
        ),
        pytest.param(
            {"invariants": np.ones((2, 4))}, ValueError, "independent", id="dependent"
        ),
    ],
)
def test_malformed_arguments_fail_visibly(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    """Masks and invariants are checked before iterating."""
    with pytest.raises(error, match=match):
        steady_state(_sirs, Y0, **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt0", 0.0),
        ("dt_max", 0.5),
        ("max_iterations", 0),
        ("residual_tol", -1.0),
        ("min_growth", 0.5),
        ("max_growth", 1.5),
    ],
)
def test_config_rejects_invalid_controls(field: str, value: float) -> None:
    """Pseudo-time and tolerance controls are validated."""
    with pytest.raises(ValueError, match=field):
        SteadyStateConfig(**{field: value})


def _vector_host(b: Any, xp: Any = np) -> Any:  # noqa: ANN401
    """Return the conserved vector-host SIRS reproduction from issue #209."""

    def rhs(_t: float, y: Any) -> Any:  # noqa: ANN401
        s, i, r, sv, iv = y
        mu, gamma, omega, mu_v = 1 / 3650, 1 / 7, 1 / 300, 1 / 700
        lam_h, lam_v = b * iv / 1000, b * i / 1000
        return xp.stack([
            mu * 1000 - lam_h * s - mu * s + omega * r,
            lam_h * s - (gamma + mu) * i,
            gamma * i - (omega + mu) * r,
            mu_v * 100_000 - lam_v * sv - mu_v * sv,
            lam_v * sv - mu_v * iv,
        ])

    return rhs


def _vector_host_equilibrium(b: float) -> NDArray[np.float64]:
    """Return the analytic endemic root, independently of the PTC iteration."""
    mu, gamma, omega, mu_v = 1 / 3650, 1 / 7, 1 / 300, 1 / 700
    recovery_ratio = gamma / (omega + mu)
    i = (1000 - (gamma + mu) * 1000**2 * mu_v / (b**2 * 100_000)) / (
        1 + recovery_ratio + (gamma + mu) * 1000 / (b * 100_000)
    )
    denominator = 1000 * mu_v + b * i
    iv = 100_000 * b * i / denominator
    sv = 100_000 * 1000 * mu_v / denominator
    s = (gamma + mu) * i * 1000 / (b * iv)
    return np.array([s, i, recovery_ratio * i, sv, iv])


VECTOR_HOST_INITIAL = np.array([1000.0, 0.0, 0.0, 99_990.0, 10.0])
VECTOR_HOST_TOTALS = np.array([[1.0, 1, 1, 0, 0], [0.0, 0, 0, 1, 1]])


@pytest.mark.parametrize("b", [0.02, 0.05, 0.2])
def test_fast_takeoff_reaches_endemic_equilibrium(b: float) -> None:
    """Seeded take-off converges within the default budget without losing mass."""
    result = require_steady_state(
        steady_state(
            _vector_host(b),
            VECTOR_HOST_INITIAL,
            invariants=VECTOR_HOST_TOTALS,
            nonnegative=True,
        )
    )
    np.testing.assert_allclose(result.state, _vector_host_equilibrium(b), rtol=1e-10)
    assert np.all(result.state >= 0)
    assert float(result.invariant_drift) < 1e-14
    assert int(result.iterations) < 100


def test_moderate_takeoff_does_not_need_a_nonnegativity_constraint() -> None:
    """Ordinary transient growth no longer shrinks the step after acceptance."""
    result = require_steady_state(
        steady_state(
            _vector_host(0.05), VECTOR_HOST_INITIAL, invariants=VECTOR_HOST_TOTALS
        )
    )
    np.testing.assert_allclose(result.state, _vector_host_equilibrium(0.05), rtol=1e-10)


def test_excessive_residual_growth_rejects_without_changing_state() -> None:
    """A finite overshoot is held; a smaller next attempt can recover."""
    result = steady_state(
        lambda _t, y: 1 - y**3,
        np.zeros(1),
        config=SteadyStateConfig(dt0=10, max_iterations=1),
    )
    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.residual, [1.0])
    assert not bool(result.converged)
    assert float(result.dt) == 5
    recovered = require_steady_state(
        steady_state(
            lambda _t, y: 1 - y**3, np.zeros(1), config=SteadyStateConfig(dt0=10)
        )
    )
    np.testing.assert_allclose(recovered.state, [1.0])


@pytest.mark.parametrize("bad_value", [np.inf, np.nan])
def test_nonfinite_candidate_residual_is_rejected(bad_value: float) -> None:
    """Non-finite residuals never replace a valid state or its diagnostics."""
    result = steady_state(
        lambda _t, y: np.where(y <= 0.5, 1.0, bad_value),
        np.zeros(1),
        jacobian=lambda _t, _y: np.zeros((1, 1)),
        config=SteadyStateConfig(max_iterations=1),
    )
    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.residual, [1.0])
    assert np.isfinite(result.residual_norm)
    assert float(result.dt) == 0.5
    assert not bool(result.converged)


def test_singular_ptc_step_is_rejected_and_retried() -> None:
    """At a singular shift NumPy raises; halving the step restores the solve."""
    rhs = lambda _t, y: y * (1 - y)  # noqa: E731
    jacobian = lambda _t, y: np.diag(1 - 2 * y)  # noqa: E731
    config = SteadyStateConfig(dt0=1.25, max_iterations=1)
    y0 = np.array([0.1])
    result = steady_state(rhs, y0, jacobian=jacobian, config=config)
    np.testing.assert_array_equal(result.state, y0)
    assert float(result.dt) == 0.625
    recovered = require_steady_state(
        steady_state(
            rhs,
            y0,
            jacobian=jacobian,
            nonnegative=True,
            config=SteadyStateConfig(dt0=1.25),
        )
    )
    np.testing.assert_allclose(recovered.state, [1.0])


def test_nonnegative_mask_allows_unconstrained_signed_states() -> None:
    """Only selected entries are constrained, and fixed states stay fixed."""
    result = require_steady_state(
        steady_state(
            lambda _t, y: np.array([1.0, -2.0, 0.0]) - y,
            np.array([0.0, 0.0, -3.0]),
            fixed=[2],
            nonnegative=[True, False, False],
        )
    )
    np.testing.assert_allclose(result.state, [1.0, -2.0, -3.0])


def test_nonnegative_rejection_keeps_rhs_evaluations_in_bounds() -> None:
    """A callback with a non-negative domain never sees a rejected overshoot."""

    def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        assert np.all(np.isfinite(y))
        assert np.all(y >= 0)
        return y * (1 - y)

    y0 = np.array([0.1])
    result = steady_state(
        rhs,
        y0,
        jacobian=lambda _t, y: np.diag(1 - 2 * y),
        nonnegative=True,
        config=SteadyStateConfig(dt0=2, max_iterations=1),
    )
    np.testing.assert_array_equal(result.state, y0)
    assert not bool(result.converged)
    assert float(result.dt) == 1


@pytest.mark.parametrize("mask", [[True], [0, 1, 0, 0], np.ones((2, 2), dtype=bool)])
def test_nonnegative_rejects_malformed_masks(mask: Any) -> None:  # noqa: ANN401
    """Bound masks must have one boolean for each state."""
    with pytest.raises(ValueError, match="nonnegative"):
        steady_state(_sirs, Y0, nonnegative=mask)


def test_nonnegative_constraint_checks_the_final_newton_correction() -> None:
    """A small residual and correction cannot mark an infeasible root converged."""
    result = steady_state(
        lambda _t, y: -y - 1e-12,
        np.zeros(1),
        jacobian=lambda _t, _y: -np.eye(1),
        nonnegative=True,
        config=SteadyStateConfig(residual_tol=1, step_tol=1, max_iterations=1),
    )
    assert not bool(result.converged)
    np.testing.assert_array_equal(result.state, [0.0])
    assert float(result.dt) == 0.5


def _enable_x64() -> contextlib.AbstractContextManager[None]:
    jax = pytest.importorskip("jax")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    return enable()


def test_fast_takeoff_with_rejection_under_jit_and_vmap() -> None:
    """Array-valued accept/reject decisions remain valid in a batched JAX loop."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with _enable_x64():

        def solve(b: Any) -> tuple[Any, Any, Any]:  # noqa: ANN401
            rhs = _vector_host(b, jnp)
            result = steady_state(
                rhs,
                jnp.asarray(VECTOR_HOST_INITIAL),
                jacobian=lambda t, y: jax.jacfwd(lambda z: rhs(t, z))(y),
                invariants=VECTOR_HOST_TOTALS,
                nonnegative=True,
                loop=jax.lax.fori_loop,
            )
            return result.state, result.converged, result.invariant_drift

        parameters = np.array([0.02, 0.05, 0.2])
        states, converged, drifts = jax.jit(jax.vmap(solve))(jnp.asarray(parameters))
        assert bool(jnp.all(converged))
        assert bool(jnp.all(states >= 0))
        assert bool(jnp.all(drifts < 1e-14))
        for b, state in zip(parameters, np.asarray(states), strict=True):
            np.testing.assert_allclose(state, _vector_host_equilibrium(b), rtol=1e-10)


def test_jit_and_vmap_over_parameters_with_fori_loop() -> None:
    """A compiled, batched solve matches the analytic equilibria."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with _enable_x64():

        def solve(beta: Any, y0: Any) -> tuple[Any, Any]:  # noqa: ANN401
            def rhs(_t: float, y: Any) -> Any:  # noqa: ANN401
                s, i, r = y[0], y[1], y[2]
                n = s + i + r
                return jnp.stack([
                    MU * n - beta * s * i / n - MU * s + OMEGA * r,
                    beta * s * i / n - (GAMMA + MU) * i,
                    GAMMA * i - (MU + OMEGA) * r,
                ])

            result = steady_state(
                rhs,
                y0,
                jacobian=lambda t, y: jax.jacfwd(lambda z: rhs(t, z))(y),
                invariants=np.ones((1, 3)),  # setup data, not traced
                config=SteadyStateConfig(max_iterations=60),
                loop=jax.lax.fori_loop,
            )
            return result.state, result.converged

        betas = jnp.array([0.3, 0.4, 0.5])
        states, converged = jax.jit(jax.vmap(solve, in_axes=(0, None)))(
            betas, jnp.array([4990.0, 10.0, 0.0])
        )
        assert bool(jnp.all(converged))
        for beta, state in zip(np.asarray(betas), np.asarray(states), strict=True):
            s = TOTAL * (GAMMA + MU) / beta
            i = (TOTAL - s) / (1 + GAMMA / (MU + OMEGA))
            np.testing.assert_allclose(state[:2], [s, i], rtol=1e-10)
