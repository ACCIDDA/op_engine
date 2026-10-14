"""Per-run stream replay, ownership and independent synthetic process moments."""

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import pytest

from op_engine import (
    AdaptiveTauLeapingConfig,
    AdaptiveTauLeapingSolver,
    Array,
    DirectSSASolver,
    ModelCore,
    NumpyPoissonSampler,
    NumpySSASampler,
    NumpyThinningSampler,
    TauLeapingSolver,
    ThinningSSASolver,
)
from op_engine.model_core import ModelCoreOptions

METHODS = ("tau", "ssa", "adaptive", "thinning")


def _case(method: str, *, dtype: str = "float64") -> tuple[ModelCore, Any]:
    core = ModelCore(
        1,
        1,
        np.array([0.0, 0.5, 1.0]),
        options=ModelCoreOptions(dtype=np.dtype(dtype)),
    )
    core.set_initial_state(np.zeros((1, 1)))
    stoichiometry = np.array([[1.0]])
    if method == "tau":
        solver = TauLeapingSolver(core, stoichiometry)
    elif method == "ssa":
        solver = DirectSSASolver(core, stoichiometry)
    elif method == "adaptive":
        solver = AdaptiveTauLeapingSolver(core, stoichiometry, np.zeros((1, 1)))
    else:
        solver = ThinningSSASolver(core, stoichiometry)
    return core, solver


def _propensity(_time: float, state: Array) -> Array:
    return cast("Array", np.full(state.shape, 12.0))


def _run(solver: Any, method: str, **kwargs: object) -> None:  # noqa: ANN401
    if method == "thinning":
        solver.run(_propensity, rate_bound=12.0, **kwargs)
    elif method == "adaptive":
        solver.run(
            _propensity,
            config=AdaptiveTauLeapingConfig(
                critical_threshold=0, exact_fallback_multiplier=0.0
            ),
            **kwargs,
        )
    else:
        solver.run(_propensity, **kwargs)


@pytest.mark.parametrize("method", METHODS)
def test_numpy_seedsequence_replays_same_solver_after_state_reset(method: str) -> None:
    """A seed sequence initializes a fresh stream after an explicit state reset."""
    core, solver = _case(method)
    stream = np.random.SeedSequence([3, 4, 5, 2026])
    _run(solver, method, rng=stream)
    first = np.array(core.state_array, copy=True)
    core.set_initial_state(np.zeros((1, 1)))
    _run(solver, method, rng=stream)
    np.testing.assert_array_equal(core.state_array, first)
    assert stream.n_children_spawned == 0


@pytest.mark.parametrize("method", METHODS)
def test_numpy_generator_advances_and_replays_from_fresh_initialization(
    method: str,
) -> None:
    """Caller-owned generator consumption is visible and reproducible across runs."""
    core, solver = _case(method)
    stream = np.random.default_rng(991)
    start = copy.deepcopy(stream.bit_generator.state)
    _run(solver, method, rng=stream)
    middle = copy.deepcopy(stream.bit_generator.state)
    first = np.array(core.state_array, copy=True)
    core.set_initial_state(np.zeros((1, 1)))
    _run(solver, method, rng=stream)
    second = np.array(core.state_array, copy=True)
    assert middle != start
    assert stream.bit_generator.state != middle
    replay = np.random.default_rng(991)
    core.set_initial_state(np.zeros((1, 1)))
    _run(solver, method, rng=replay)
    np.testing.assert_array_equal(core.state_array, first)
    core.set_initial_state(np.zeros((1, 1)))
    _run(solver, method, rng=replay)
    np.testing.assert_array_equal(core.state_array, second)


@pytest.mark.parametrize("method", METHODS)
def test_numpy_indexed_replicates_do_not_depend_on_launch_order(method: str) -> None:
    """Stable replicate/unit/stage identities support reordered execution."""
    core, solver = _case(method)

    def evaluate(order: list[int]) -> dict[int, np.ndarray]:
        results = {}
        for replicate in order:
            core.set_initial_state(np.zeros((1, 1)))
            _run(solver, method, rng=np.random.SeedSequence([replicate, 7, 2, 101]))
            results[replicate] = np.array(core.state_array, copy=True)
        return results

    forward = evaluate([0, 1, 2, 3])
    reverse = evaluate([3, 2, 1, 0])
    for replicate in forward:
        np.testing.assert_array_equal(forward[replicate], reverse[replicate])


@pytest.mark.parametrize("method", METHODS)
def test_numpy_explicit_run_stream_preserves_existing_sampler(method: str) -> None:
    """Call-local streams preserve an independently held sampler's state."""
    core, solver = _case(method)

    def samplers() -> dict[str, object]:
        if method == "tau":
            return {"poisson_sampler": NumpyPoissonSampler(seed=16)}
        if method == "ssa":
            return {"ssa_sampler": NumpySSASampler(seed=16)}
        if method == "thinning":
            return {"thinning_sampler": NumpyThinningSampler(seed=16)}
        return {
            "poisson_sampler": NumpyPoissonSampler(seed=16),
            "ssa_sampler": NumpySSASampler(seed=17),
        }

    original = samplers()
    _run(solver, method, rng=np.random.SeedSequence(442))
    core.set_initial_state(np.zeros((1, 1)))
    _run(solver, method, **original)
    actual = np.array(core.state_array, copy=True)
    core.set_initial_state(np.zeros((1, 1)))
    _run(solver, method, **samplers())
    np.testing.assert_array_equal(core.state_array, actual)


@pytest.mark.parametrize("method", METHODS)
def test_numpy_invalid_or_conflicting_run_streams_fail_before_advancement(
    method: str,
) -> None:
    """Neither missing sources nor native/functional sampler conflicts are hidden."""
    core, solver = _case(method)
    with pytest.raises(ValueError, match="provide"):
        _run(solver, method)
    with pytest.raises(TypeError, match="Generator or SeedSequence"):
        _run(solver, method, rng=123)
    name = (
        "poisson_sampler"
        if method in {"tau", "adaptive"}
        else "ssa_sampler"
        if method == "ssa"
        else "thinning_sampler"
    )
    sampler = (
        NumpyPoissonSampler(seed=1)
        if name == "poisson_sampler"
        else NumpySSASampler(seed=1)
        if name == "ssa_sampler"
        else NumpyThinningSampler(seed=1)
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(solver, method, rng=np.random.default_rng(1), **{name: sampler})
    assert core.current_step == 0
    np.testing.assert_array_equal(core.current_state, np.zeros((1, 1)))


@pytest.mark.parametrize(
    "sampler_type", [NumpyPoissonSampler, NumpySSASampler, NumpyThinningSampler]
)
def test_numpy_sampler_stream_validation(sampler_type: Any) -> None:  # noqa: ANN401
    """Legacy seed construction remains separate from explicit stream initialization."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        sampler_type(seed=1, rng=np.random.SeedSequence(1))
    with pytest.raises(TypeError, match="Generator or SeedSequence"):
        sampler_type(rng=1)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_state_requires_native_sampler_even_when_absorbing(
    method: str, backend: str
) -> None:
    """A NumPy convenience stream never silently crosses a native state boundary."""
    module = pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    core, solver = _case(method, dtype="float32")
    core.set_initial_state(module.zeros((1, 1)))
    generator = np.random.default_rng(42)
    before = copy.deepcopy(generator.bit_generator.state)
    with pytest.raises(TypeError, match="requires NumPy state"):
        _run(solver, method, rng=generator)
    assert generator.bit_generator.state == before
    assert core.current_step == 0


def test_numpy_adaptive_stream_matches_manual_shared_sampler_injection() -> None:
    """Critical death and noncritical birth draw from one advancing stream."""
    core = ModelCore(2, 1, np.array([0.0, 0.5, 1.0]))
    initial = np.array([[3.0], [0.0]])
    core.set_initial_state(initial)
    solver = AdaptiveTauLeapingSolver(
        core, np.array([[-1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 0.0]])
    )

    def propensity(_time: float, y: Array) -> Array:
        value = np.asarray(y)
        return cast("Array", np.array([[2.0 * value[0, 0]], [12.0]]))

    config = AdaptiveTauLeapingConfig(exact_fallback_multiplier=0.0)
    solver.run(propensity, rng=np.random.SeedSequence(772), config=config)
    actual = np.array(core.state_array, copy=True)
    assert np.asarray(core.current_state)[0, 0] < 3.0
    assert np.asarray(core.current_state)[1, 0] > 0.0
    core.set_initial_state(initial)
    reference = np.random.default_rng(np.random.SeedSequence(772))
    solver.run(
        propensity,
        NumpyPoissonSampler(rng=reference),
        NumpySSASampler(rng=reference),
        config=config,
    )
    np.testing.assert_array_equal(core.state_array, actual)


@pytest.mark.parametrize("method", METHODS)
def test_numpy_indexed_stream_birth_process_has_analytic_moments(method: str) -> None:
    """Independent constant births have Poisson(12) endpoint mean and variance."""
    core, solver = _case(method)
    counts = []
    repetitions = 256
    for replicate in range(repetitions):
        core.set_initial_state(np.zeros((1, 1)))
        _run(solver, method, rng=np.random.SeedSequence([replicate, 3, 0, 73019]))
        counts.append(core.current_state.item())
    # Predetermined broad Monte Carlo bounds; not an empirical precision claim.
    assert abs(np.mean(counts) - 12.0) < 5 * np.sqrt(12.0 / repetitions)
    assert abs(np.var(counts, ddof=1) - 12.0) < 5 * 12.0 * np.sqrt(
        2.0 / (repetitions - 1)
    )
