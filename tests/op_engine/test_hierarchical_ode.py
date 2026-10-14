"""Independent trajectory and higher-order AD evidence on hierarchical states."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.hierarchical_ode import (
    HierarchicalProblem,
    initial_state,
    loss_function,
    nominal_parameters,
    numpy_replay_trajectory,
    observations,
    record_schedule,
    reference_initial,
    reference_objective,
    reference_rhs,
    reference_trajectory,
    refined_schedule,
    replay_trajectory,
    rhs,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from op_engine import Array

TIMES = np.asarray([0.0, 0.2, 0.5, 1.0])
REGIMES = [(False, False), (False, True), (True, False), (True, True)]


@pytest.fixture
def jax_x64() -> Iterator[None]:
    """Scope double precision to independent derivative evidence."""
    jax = pytest.importorskip("jax")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        yield


@pytest.mark.parametrize(("stiff", "coupled"), REGIMES)
def test_numpy_hierarchical_equations_and_trajectories(
    *, stiff: bool, coupled: bool
) -> None:
    """Cell-loop equations and an accurate independent solve agree for all regimes."""
    problem = HierarchicalProblem(
        groups=3, local_dimension=2, stiff=stiff, coupled=coupled
    )
    parameters = nominal_parameters(problem)
    initial = np.asarray(initial_state(problem, parameters))
    np.testing.assert_allclose(
        initial, reference_initial(problem, parameters), rtol=1e-14
    )
    compiled = np.asarray(rhs(problem, parameters)(0.0, initial))
    expected = reference_rhs(problem, parameters, initial)
    np.testing.assert_allclose(compiled, expected, rtol=1e-14)
    np.testing.assert_allclose(compiled.sum(axis=0), np.zeros((3, 2)), atol=1e-14)
    schedule, config = record_schedule(problem, parameters, TIMES)
    trajectory = numpy_replay_trajectory(problem, parameters, schedule, config)
    independent = reference_trajectory(problem, parameters, TIMES)
    np.testing.assert_allclose(trajectory, independent, rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(
        trajectory.sum(axis=1), np.ones((len(TIMES), 3, 2)), atol=2e-13
    )
    refined = numpy_replay_trajectory(
        problem, parameters, refined_schedule(schedule), config
    )
    assert (
        np.max(abs(refined - independent))
        <= np.max(abs(trajectory - independent)) + 2e-13
    )


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize(("stiff", "coupled"), REGIMES)
def test_jax_hierarchical_forward_reverse_hessian_and_logdet(  # noqa: PLR0914
    *, stiff: bool, coupled: bool
) -> None:
    """Shared/local/IC derivatives use a nonzero-residual hierarchical objective."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    problem = HierarchicalProblem(stiff=stiff, coupled=coupled)
    nominal = nominal_parameters(problem)
    parameters = jnp.asarray(nominal)
    data_np = observations(problem, TIMES)
    data = jnp.asarray(data_np)
    schedule, config = record_schedule(problem, nominal, TIMES)
    trajectory = jax.jit(
        lambda values: replay_trajectory(problem, values, schedule, config)
    )(parameters)
    external = jax.jit(
        lambda values: replay_trajectory(
            problem, values, schedule, config, external=True
        )
    )(parameters)
    reference = reference_trajectory(problem, nominal, TIMES)
    np.testing.assert_allclose(trajectory, reference, rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(external, trajectory, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        trajectory,
        numpy_replay_trajectory(problem, nominal, schedule, config),
        rtol=1e-12,
        atol=1e-13,
    )
    assert np.max(abs(reference[1:, 0] - data_np)) > 0.001
    loss = loss_function(problem, schedule, config, data)
    external_loss = loss_function(problem, schedule, config, data, external=True)
    value, reverse = jax.jit(jax.value_and_grad(loss))(parameters)
    forward = jax.jit(jax.jacfwd(loss))(parameters)
    np.testing.assert_allclose(reverse, forward, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        value,
        reference_objective(problem, nominal, TIMES, data_np),
        rtol=1e-6,
        atol=1e-7,
    )
    perturbations = 1e-4 * np.eye(problem.parameter_count)
    reference_gradient = np.asarray([
        (
            reference_objective(problem, nominal + shift, TIMES, data_np)
            - reference_objective(problem, nominal - shift, TIMES, data_np)
        )
        / 2e-4
        for shift in perturbations
    ])
    np.testing.assert_allclose(reverse, reference_gradient, rtol=2e-5, atol=2e-6)
    hessian = jax.jit(jax.hessian(loss))(parameters)
    gradient = jax.jit(jax.grad(loss))
    hessian_fd = np.column_stack([
        (
            np.asarray(gradient(parameters + shift))
            - np.asarray(gradient(parameters - shift))
        )
        / 2e-4
        for shift in perturbations
    ])
    np.testing.assert_allclose(hessian, hessian_fd, rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(hessian, np.asarray(hessian).T, rtol=1e-12, atol=1e-12)
    regularization = 25.0
    regularized = hessian + regularization * jnp.eye(problem.parameter_count)
    assert np.min(np.linalg.eigvalsh(np.asarray(regularized))) > 0

    def logdet(values: Array) -> Array:
        matrix = jax.hessian(loss)(values) + regularization * jnp.eye(
            problem.parameter_count
        )
        return cast("Array", 0.5 * jnp.linalg.slogdet(matrix)[1])

    laplace_gradient = jax.jit(jax.grad(logdet))(parameters)
    evaluate = jax.jit(logdet)
    logdet_fd = np.asarray([
        (evaluate(parameters + shift) - evaluate(parameters - shift)) / 2e-4
        for shift in perturbations
    ])
    assert np.linalg.norm(np.asarray(laplace_gradient)) > 1e-5
    np.testing.assert_allclose(laplace_gradient, logdet_fd, rtol=2e-5, atol=2e-6)
    external_gradient = jax.jit(jax.grad(external_loss))(parameters)
    np.testing.assert_allclose(external_gradient, reverse, rtol=1e-10, atol=1e-10)
    refined = loss_function(problem, refined_schedule(schedule), config, data)
    refined_gradient = jax.jit(jax.grad(refined))(parameters)
    np.testing.assert_allclose(refined_gradient, reverse, rtol=2e-5, atol=2e-6)
