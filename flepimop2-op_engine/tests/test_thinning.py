# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""End-to-end thinning through real producer reaction artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum
from op_engine import NumpyThinningSampler, ThinningSample, TotalRateBound

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine

if TYPE_CHECKING:
    from flepimop2.typing import Array
    from numpy.typing import NDArray


def _birth_system(n_cells: int = 1) -> OpSystemSystem:
    """Return a generic source-only network with smoothly increasing rates."""
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [{"name": "group", "coords": [str(i) for i in range(n_cells)]}],
            "state": ["X[group]"],
            "transitions": [
                {
                    "name": "birth",
                    "from": None,
                    "to": "X[group]",
                    "rate": "2 * t",
                    "reactants": [],
                }
            ],
        }
    )


def _engine(**controls: object) -> OpEngineFlepimop2Engine:
    """Return a pure stochastic thinning provider."""
    return OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig.model_validate(
            {
                "mode": "stochastic",
                "stochastic_method": "thinning-ssa",
            }
            | controls
        ),
    )


class _Candidates:
    """Record inputs and return prescribed native exponential/uniform draws."""

    def __init__(self, *draws: tuple[float, float]) -> None:
        self.draws = iter(draws)
        self.indices: list[int] = []
        self.rates: list[float] = []

    def __call__(self, rate: Array, index: int, /) -> ThinningSample:
        """Return the next namespace-preserving candidate."""
        self.indices.append(index)
        self.rates.append(float(rate.item()))
        xp = rate.__array_namespace__()
        wait, uniform = next(self.draws)
        return ThinningSample(
            xp.asarray(wait, dtype=rate.dtype),
            xp.asarray(uniform, dtype=rate.dtype),
        )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("bound_source", ["config", "run"])
def test_provider_smooth_births_resume_reject_and_preserve_pending_draws(
    backend: str,
    bound_source: str,
) -> None:
    """Candidate-time rates wake an initially dormant producer on both backends."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = _birth_system()
    engine = _engine(**({"thinning_rate_bound": 2} if bound_source == "config" else {}))
    initial = {
        str(name): ParameterValue(xp.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    sampler = _Candidates((0.25, 0.5), (0.25, 0.1), (0.25, 0.1), (0.25, 0))
    result = engine.run(
        system,
        np.asarray([0, 0.1, 0.25, 0.5, 0.75, 1]),
        initial,
        {},
        thinning_sampler=sampler,
        **({"rate_bound": 2} if bound_source == "run" else {}),
    )
    np.testing.assert_array_equal(np.asarray(result)[:, 1], [0, 0, 0, 1, 2, 2])
    assert result.__array_namespace__() is xp
    assert sampler.indices == [0, 1, 2, 3]
    assert sampler.rates == [2, 2, 2, 2]
    assert engine.validate_system(system) is None


def test_seeded_numpy_provider_paths_ignore_added_observations() -> None:
    """Seeded NumPy paths are reproducible and independent of observations."""
    system = _birth_system(16)
    initial = {
        str(name): ParameterValue(np.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    engine = _engine(thinning_rate_bound=32, random_seed=173)
    coarse = np.asarray([0, 0.5, 1])
    dense = np.asarray([0, 0.125, 0.25, 0.5, 0.75, 1])
    first = engine.run(system, coarse, initial, {})
    repeat = engine.run(system, coarse, initial, {})
    extra = engine.run(system, dense, initial, {})
    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(first, np.asarray(extra)[[0, 3, 5]])
    assert np.sum(np.asarray(first)[-1, 1:]) > 0


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_callback_bounds_and_functional_indices_ignore_observations(
    backend: str,
) -> None:
    """Bounds receive native core state and forcing limits across expiry and events."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = _birth_system(16)
    initial = {
        str(name): ParameterValue(xp.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    engine = _engine(forcing_breakpoints=(0.75,))
    if backend == "jax":
        jax = pytest.importorskip("jax")
        key = jax.random.key(173)

    def solve(
        times: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], list[int], list[tuple[float, float, float]]]:
        """Return shared observations and the sampler/bound call histories."""
        indices: list[int] = []
        bounds: list[tuple[float, float, float]] = []
        numpy_sampler = NumpyThinningSampler(173)

        def bound(t: float, state: Array, limit: float) -> TotalRateBound:
            assert state.shape == (16, 1)
            assert state.__array_namespace__() is xp
            bounds.append((t, float(np.sum(np.asarray(state))), limit))
            return TotalRateBound(2 * limit * 16, min(0.5, limit) if t < 0.5 else limit)

        def sampler(rate: Array, index: int, /) -> ThinningSample:
            indices.append(index)
            assert rate.__array_namespace__() is xp
            if backend == "numpy":
                return numpy_sampler(rate, index)
            wait_key, uniform_key = jax.random.split(jax.random.fold_in(key, index))
            return ThinningSample(
                jax.random.exponential(wait_key, dtype=rate.dtype) / rate,
                jax.random.uniform(uniform_key, dtype=rate.dtype),
            )

        result = engine.run(
            system, times, initial, {}, rate_bound=bound, thinning_sampler=sampler
        )
        assert result.__array_namespace__() is xp
        assert indices == list(range(len(indices)))
        shared = np.asarray(result)[np.isin(times, [0, 0.5, 1])]
        assert shared[-1, 1:].sum() > 0
        return shared, indices, bounds

    coarse = solve(np.asarray([0.0, 0.5, 1.0]))
    dense = solve(np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_array_equal(coarse[0], dense[0])
    assert coarse[1:] == dense[1:]


def _table_system(policy: str) -> OpSystemSystem:
    """Return a real source-only producer with time-indexed rates."""
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "time_axis": "day",
            "time_interpolation": policy,
            "axes": [
                {"name": "group", "coords": ["a"]},
                {"name": "day", "type": "continuous", "coords": [0, 0.5, 0.75]},
            ],
            "state": ["X[group]"],
            "transitions": [
                {
                    "name": "birth",
                    "from": None,
                    "to": "X[group]",
                    "rate": "rate[day]",
                    "reactants": [],
                }
            ],
        }
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("policy", ["linear", "previous"])
def test_real_time_tables_use_candidate_rates_and_declared_boundaries(
    backend: str, policy: str
) -> None:
    """Linear tables stay smooth; hold tables combine declared forcing changes."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = _table_system(policy)
    initial = {
        str(name): ParameterValue(xp.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    params = {
        "rate": ParameterValue(
            xp.asarray([0.0, 1.0, 0.0]), ResolvedShape(("day",), (3,))
        )
    }
    limits: list[float] = []

    def bound(_t: float, _state: Array, limit: float) -> TotalRateBound:
        limits.append(limit)
        return TotalRateBound(1, limit)

    if policy == "linear":
        engine = _engine()
        times = np.asarray([0, 0.25, 0.5, 0.625, 0.75, 1])
        sampler = _Candidates(
            (0.25, 0.9), (0.25, 0.1), (0.125, 0.1), (0.125, 0.1), (0.25, 0)
        )
        expected = [0, 0, 1, 2, 2, 2]
    else:
        engine = _engine(forcing_breakpoints=(0.625,))
        times = np.asarray([0, 0.5, 0.5625, 0.625, 0.75, 1])
        sampler = _Candidates((0.5, 0), (0.0625, 0), (0.0625, 0), (0.125, 0), (0.25, 0))
        expected = [0, 0, 1, 1, 1, 1]
    result = engine.run(
        system, times, initial, params, rate_bound=bound, thinning_sampler=sampler
    )
    np.testing.assert_array_equal(np.asarray(result)[:, 1], expected)
    assert result.__array_namespace__() is xp
    assert sampler.indices == [0, 1, 2, 3, 4]
    assert limits == ([1, 1, 1] if policy == "linear" else [0.5, 0.625, 0.625, 0.75, 1])
    assert engine.config.forcing_breakpoints == (() if policy == "linear" else (0.625,))


def test_real_producer_birth_distribution_matches_integrated_intensity() -> None:
    """A real producer reproduces the analytic Poisson mean and variance."""
    system = _birth_system(1200)
    initial = {
        str(name): ParameterValue(np.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    result = _engine(thinning_rate_bound=2400, random_seed=173).run(
        system,
        np.asarray([0.0, 1.0]),
        initial,
        {},
    )
    counts = np.asarray(result)[-1, 1:]
    assert counts.mean() == pytest.approx(1, rel=0.08)
    assert counts.var() == pytest.approx(1, rel=0.15)


def test_provider_candidate_guard_counts_rejections_across_observations() -> None:
    """Provider limits cover the whole run and include rejected draws."""
    system = _birth_system()
    initial = {
        str(name): ParameterValue(np.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    sampler = _Candidates(*[(0.125, 0.99)] * 3)
    with pytest.raises(RuntimeError, match="max_candidates"):
        _engine(thinning_rate_bound=2, thinning_max_candidates=3).run(
            system,
            np.asarray([0, 0.1, 0.2, 0.3, 1]),
            initial,
            {},
            thinning_sampler=sampler,
        )
    assert sampler.indices == [0, 1, 2]


@pytest.mark.parametrize("bound", [-1, np.inf, np.nan, True, "2", [2]])
def test_invalid_run_time_bounds_fail_before_sampling(bound: object) -> None:
    """The run-time scalar interface retains the core's strict bound contract."""
    sampler = _Candidates()
    with pytest.raises(ValueError, match="rate_bound"):
        _engine().run(
            _birth_system(),
            np.asarray([0.0, 1.0]),
            {},
            {},
            rate_bound=bound,
            thinning_sampler=sampler,
        )
    assert sampler.indices == []


def test_missing_and_conflicting_bound_sources_fail_before_sampling() -> None:
    """A missing bound or two simultaneous bound sources cannot fall back silently."""
    system = _birth_system()
    sampler = _Candidates()
    with pytest.raises(ValueError, match="requires thinning_rate_bound"):
        _engine().run(system, np.asarray([0.0, 1.0]), {}, {}, thinning_sampler=sampler)
    with pytest.raises(ValueError, match="not both"):
        _engine(thinning_rate_bound=2).run(
            system,
            np.asarray([0.0, 1.0]),
            {},
            {},
            rate_bound=2,
            thinning_sampler=sampler,
        )
    assert sampler.indices == []


def test_violated_bound_and_invalid_callback_expiry_fail_visibly() -> None:
    """The provider forwards the bound contract without clipping or extending it."""
    system = _birth_system()
    sampler = _Candidates((0.75, 0.99))
    with pytest.raises(ValueError, match="exceeds bound rate"):
        _engine(thinning_rate_bound=1).run(
            system,
            np.asarray([0.0, 1.0]),
            {},
            {},
            thinning_sampler=sampler,
        )
    assert sampler.indices == [0]
    with pytest.raises(ValueError, match="valid_until"):
        _engine().run(
            system,
            np.asarray([0.0, 1.0]),
            {},
            {},
            rate_bound=lambda _t, _y, limit: TotalRateBound(2, limit + 1),
            thinning_sampler=_Candidates(),
        )


def test_invalid_sampler_and_missing_jax_sampler_fail() -> None:
    """A sampler must be callable, and non-NumPy runs must inject one."""
    system = _birth_system()
    with pytest.raises(TypeError, match="thinning_sampler must be callable"):
        _engine(thinning_rate_bound=2).run(
            system,
            np.asarray([0.0, 1.0]),
            {},
            {},
            thinning_sampler=123,
        )
    jnp = pytest.importorskip("jax.numpy")
    initial = {
        str(name): ParameterValue(jnp.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    with pytest.raises(TypeError, match="required for non-NumPy"):
        _engine(thinning_rate_bound=2).run(system, np.asarray([0.0, 1.0]), initial, {})
    with pytest.raises(TypeError, match="preserve the state array namespace"):
        _engine(thinning_rate_bound=2).run(
            system,
            np.asarray([0.0, 1.0]),
            initial,
            {},
            thinning_sampler=lambda _r, _i: ThinningSample(
                np.asarray(0.25), np.asarray(0.5)
            ),
        )


@pytest.mark.parametrize("mode", ["deterministic", "stochastic", "hybrid"])
@pytest.mark.parametrize("input_name", ["rate_bound", "thinning_sampler"])
def test_thinning_inputs_cannot_be_ignored_by_other_methods(
    mode: str, input_name: str
) -> None:
    """Reserved thinning inputs require an explicit selection of the method."""
    engine = _engine(
        mode=mode,
        stochastic_method="tau-leaping",
        stochastic_reactions=("birth",) if mode == "hybrid" else (),
    )
    with pytest.raises(ValueError, match="require stochastic_method"):
        engine.run(
            _birth_system(),
            np.asarray([0.0, 1.0]),
            {},
            {},
            **{input_name: 2 if input_name == "rate_bound" else _Candidates()},
        )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_zero_bound_hold_intervals_resume_without_sampling(backend: str) -> None:
    """A dormant producer advances to its next change and resumes under a new bound."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = _table_system("previous")
    initial = {
        str(name): ParameterValue(xp.asarray(0.0), ResolvedShape())
        for name in system.option("state_names")
    }
    params = {
        "rate": ParameterValue(
            xp.asarray([0.0, 1.0, 0.0]), ResolvedShape(("day",), (3,))
        )
    }
    sampler = _Candidates((0.125, 0), (0.125, 0))
    result = _engine().run(
        system,
        np.asarray([0, 0.25, 0.5, 0.625, 0.75, 1]),
        initial,
        params,
        rate_bound=lambda t, _y, limit: TotalRateBound(
            1 if 0.5 <= t < 0.75 else 0, limit
        ),
        thinning_sampler=sampler,
    )
    np.testing.assert_array_equal(np.asarray(result)[:, 1], [0, 0, 0, 1, 1, 1])
    assert sampler.indices == [0, 1]


def _transfer_system(rate: str) -> OpSystemSystem:
    """Return a generic one-cell A-to-B network."""
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [{"name": "group", "coords": ["a"]}],
            "state": ["A[group]", "B[group]"],
            "transitions": [
                {
                    "name": "transfer",
                    "from": "A[group]",
                    "to": "B[group]",
                    "rate": rate,
                    "reactants": [{"state": "A[group]", "order": 1}],
                }
            ],
        }
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_accepted_events_refresh_state_dependent_provider_bounds(backend: str) -> None:
    """Bounds use updated native populations and become zero after exhaustion."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = _transfer_system("2 * t")
    initial = {
        str(name): ParameterValue(xp.asarray(value), ResolvedShape())
        for name, value in zip(system.option("state_names"), (2.0, 0.0), strict=True)
    }
    bounds: list[float] = []

    def bound(_t: float, state: Array, limit: float) -> TotalRateBound:
        assert state.__array_namespace__() is xp
        rate = 2 * limit * float(np.asarray(state)[0, 0])
        bounds.append(rate)
        return TotalRateBound(rate, limit)

    sampler = _Candidates((0.25, 0.1), (0.25, 0.1))
    result = _engine().run(
        system,
        np.asarray([0, 0.1, 0.25, 0.5, 1]),
        initial,
        {},
        rate_bound=bound,
        thinning_sampler=sampler,
    )
    np.testing.assert_array_equal(
        np.asarray(result)[:, 1:], [[2, 0], [2, 0], [1, 1], [0, 2], [0, 2]]
    )
    assert bounds == [4, 2, 0]
    assert sampler.rates == [4, 2]


@pytest.mark.parametrize(
    "method", ["tau-leaping", "adaptive-tau-leaping", "direct-ssa"]
)
def test_legacy_provider_seeded_histories_remain_unchanged(method: str) -> None:
    """Histories recorded from merged main retain the pre-thinning random streams."""
    system = _transfer_system("0.1")
    initial = {
        str(name): ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in zip(system.option("state_names"), (100.0, 0.0), strict=True)
    }
    engine = _engine(
        stochastic_method=method,
        random_seed=173,
        tau_max_step=0.1 if method == "tau-leaping" else None,
    )
    result = engine.run(system, np.asarray([0, 0.2, 0.5, 1]), initial, {})
    expected = (
        [[100, 0], [96, 4], [92, 8], [81, 19]]
        if method == "tau-leaping"
        else [[100, 0], [100, 0], [95, 5], [91, 9]]
    )
    np.testing.assert_array_equal(np.asarray(result)[:, 1:], expected)
