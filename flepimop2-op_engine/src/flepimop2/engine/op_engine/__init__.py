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

"""flepimop2 Engine integration for op_engine (thin, single-file)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from flepimop2.engine.abc import EngineABC
from flepimop2.exceptions import ValidationIssue
from flepimop2.typing import IdentifierString, StateChangeEnum  # noqa: TC002
from pydantic import Field, PrivateAttr

from op_engine.core_solver import (
    AdaptiveStepSchedule,
    CoreSolver,
    NonlinearIntegrationDiagnostics,
    NonlinearMethodConfig,
    fixed_step_sizes,
)
from op_engine.model_core import ModelCore, ModelCoreOptions
from op_engine.stochastic_solver import (
    DirectSSAConfig,
    DirectSSASolver,
    NumpyPoissonSampler,
    NumpySSASampler,
    TauLeapingConfig,
    TauLeapingSolver,
)

from .config import (
    ExecutionMode,
    OpEngineEngineConfig,
    SolverMethod,
    StochasticMethod,
    _coerce_operator_specs,
    _has_operator_specs,
)
from .operators import (
    compile_operator_descriptors,
    typed_operator_descriptors,
)
from .reactions import CompiledReactionNetwork, compile_reaction_network

if TYPE_CHECKING:
    from collections.abc import Callable

    from flepimop2.parameter.abc import ModelStateSpecification, ParameterValue
    from flepimop2.system.abc import SystemABC, SystemProtocol
    from flepimop2.typing import Array, Float64NDArray
    from numpy.typing import DTypeLike

    from op_engine._typing import Scalar
    from op_engine.core_solver import CoreOperators, RunConfig, StageOperatorFactory
    from op_engine.nonlinear_solver import NonlinearSolver
    from op_engine.stochastic_solver import PoissonSampler, SSASampler


class _IndexableArray(Protocol):
    """Internal indexing surface kept out of the public Array protocol."""

    def __getitem__(self, key: object) -> Array:
        """Return one array item or slice."""
        ...


_ARRAY_API_ERROR = (
    "op_engine provider inputs must implement __array_namespace__(); got {type_name}."
)


@dataclass(frozen=True, slots=True)
class AdaptiveSchedule:
    """Provider replay artifact for one accepted adaptive mesh.

    The accepted steps are conditional on the model, parameters, tolerances,
    and output grid used during discovery. The provider can validate the
    recorded method and controller configuration, but callers must refresh the
    schedule after material model or parameter changes.
    """

    method: SolverMethod
    step_schedule: AdaptiveStepSchedule
    rtol: float
    atol: float
    dt_init: float | None
    max_reject: int
    max_steps: int
    dt_min: float
    dt_max: float
    safety: float
    fac_min: float
    fac_max: float
    gamma: float | None
    strict: bool

    @classmethod
    def from_core(
        cls,
        step_schedule: AdaptiveStepSchedule,
        config: OpEngineEngineConfig,
    ) -> AdaptiveSchedule:
        """Build an artifact from a completed provider discovery run.

        Returns:
            Schedule artifact bound to the active numerical configuration.
        """
        return cls(
            method=config.method,
            step_schedule=step_schedule,
            rtol=config.rtol,
            atol=config.atol,
            dt_init=config.dt_init,
            max_reject=config.max_reject,
            max_steps=config.max_steps,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            safety=config.safety,
            fac_min=config.fac_min,
            fac_max=config.fac_max,
            gamma=config.gamma,
            strict=config.strict,
        )

    def validate_config(self, config: OpEngineEngineConfig) -> None:
        """Validate that a provider configuration can replay this artifact.

        Raises:
            ValueError: If the solver method or adaptive controls changed.
        """
        if config.method is not self.method:
            msg = (
                f"Adaptive schedule method '{self.method.value}' does not match "
                f"configured method '{config.method.value}'."
            )
            raise ValueError(msg)

        recorded = (
            self.rtol,
            self.atol,
            self.dt_init,
            self.max_reject,
            self.max_steps,
            self.dt_min,
            self.dt_max,
            self.safety,
            self.fac_min,
            self.fac_max,
            self.gamma,
            self.strict,
        )
        configured = (
            config.rtol,
            config.atol,
            config.dt_init,
            config.max_reject,
            config.max_steps,
            config.dt_min,
            config.dt_max,
            config.safety,
            config.fac_min,
            config.fac_max,
            config.gamma,
            config.strict,
        )
        if recorded != configured:
            msg = (
                "Adaptive schedule controller settings do not match the active "
                "engine configuration; discover a fresh schedule."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AdaptiveRunResult:
    """Trajectory, frozen schedule, and optional nonlinear diagnostics."""

    trajectory: Array
    schedule: AdaptiveSchedule
    diagnostics: NonlinearIntegrationDiagnostics | None = None

    def require_converged(self) -> AdaptiveRunResult:
        """Validate any nonlinear diagnostics attached to this result.

        Returns:
            This unchanged result after successful validation.
        """
        if self.diagnostics is not None:
            self.diagnostics.require_converged()
        return self


@dataclass(frozen=True, slots=True)
class _ExecutionResult:
    """Internal result shared by ordinary and schedule-aware entry points."""

    trajectory: Array
    schedule: AdaptiveSchedule | None = None
    diagnostics: NonlinearIntegrationDiagnostics | None = None


def _namespace_of(value: object) -> Any:  # noqa: ANN401
    """Return the Array-API namespace advertised by ``value``."""
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        raise TypeError(_ARRAY_API_ERROR.format(type_name=type(value).__name__))
    return namespace()


def _is_jax_namespace(namespace: object) -> bool:
    """Return whether an Array-API namespace is JAX's NumPy namespace."""
    return getattr(namespace, "__name__", "") == "jax.numpy"


def _array_item(value: Array, key: object) -> Array:
    """Index an Array while keeping the shared public protocol minimal."""
    return cast("_IndexableArray", value)[key]


def _state_reference(  # noqa: PLR0911
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
) -> object | None:
    """Return the first payload that contributes to the initial state."""
    state_names = system.option("state_names", None)
    init_map = system.option("initial_state", None)
    if isinstance(state_names, tuple | list) and isinstance(init_map, Mapping):
        for state_name_obj in state_names:
            state_name = str(state_name_obj)
            if state_name in initial_state:
                return initial_state[state_name].value

            entry = init_map.get(state_name)
            if isinstance(entry, str) and entry in params:
                return params[entry].value
            if isinstance(entry, Mapping):
                shaped_name = entry.get("shaped")
                if isinstance(shaped_name, str) and shaped_name in params:
                    return params[shaped_name].value
            elif getattr(entry, "__array_namespace__", None) is not None:
                return entry

    if initial_state:
        return next(iter(initial_state.values())).value
    if params:
        return next(iter(params.values())).value
    return None


def _numerical_context(
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
) -> tuple[Any, object]:
    """Select the state namespace and floating dtype for a run.

    Returns:
        Array namespace and result dtype selected from incoming payloads.
    """
    reference = _state_reference(system, initial_state, params)
    if reference is None:
        xp = np
        return xp, xp.asarray(0.0).dtype

    reference_array = cast("Array", reference)
    xp = _namespace_of(reference_array)
    default_float_dtype = xp.asarray(0.0).dtype
    dtype = xp.result_type(
        default_float_dtype, cast("DTypeLike", reference_array.dtype)
    )
    return xp, dtype


def _as_float64_1d(x: object, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"{name} must be a 1D array"
        raise ValueError(msg)
    return np.ascontiguousarray(arr)


def _ensure_strictly_increasing(times: np.ndarray, *, name: str) -> None:
    if times.size <= 1:
        return
    if np.any(np.diff(times) <= 0.0):
        msg = f"{name} must be strictly increasing"
        raise ValueError(msg)


def _rhs_from_stepper(
    stepper: SystemProtocol,
    *,
    n_state: int,
) -> Callable[[Scalar, Array], Array]:
    def rhs(time: Scalar, state: Array) -> Array:
        xp = _namespace_of(state)
        state_arr = state
        if state_arr.shape == (n_state,):
            state_arr = cast("Array", xp.reshape(state_arr, (n_state, 1)))
        expected_shape = (n_state, 1)
        if state_arr.shape != expected_shape:
            msg = (
                f"RHS received unexpected state shape {state_arr.shape}; "
                f"expected {expected_shape}."
            )
            raise ValueError(msg)

        flat_state = _array_item(state_arr, (slice(None), 0))
        out = stepper(cast("Any", time), cast("Any", flat_state))
        out_xp = _namespace_of(out)
        if out_xp is not xp:
            msg = "System stepper must preserve the state array namespace."
            raise TypeError(msg)
        out_arr = cast("Array", xp.asarray(out, dtype=state_arr.dtype))
        if out_arr.shape != (n_state,):
            msg = f"Stepper returned shape {out_arr.shape}; expected {(n_state,)}."
            raise ValueError(msg)
        return cast("Array", xp.reshape(out_arr, expected_shape))

    return rhs


def _extract_states_2d(core: ModelCore, *, n_state: int) -> Array:
    state_array = getattr(core, "state_array", None)
    if state_array is None:
        msg = "ModelCore does not expose state_array; store_history must be enabled."
        raise RuntimeError(msg)
    arr = cast("Array", state_array)
    if len(arr.shape) == 3 and arr.shape[1] == n_state and arr.shape[2] == 1:
        return _array_item(arr, (slice(None), slice(None), 0))
    if len(arr.shape) == 2 and arr.shape[1] == n_state:
        return arr
    msg = (
        f"Unexpected state shape {arr.shape}; "
        f"expected (T, {n_state}, 1) or (T, {n_state})."
    )
    raise RuntimeError(msg)


def _make_core(times: np.ndarray, y0: Array) -> ModelCore:
    if len(y0.shape) != 1:
        msg = f"Initial state must be one-dimensional; got {y0.shape}."
        raise ValueError(msg)
    n_states = int(y0.shape[0])
    core = ModelCore(
        n_states,
        1,
        np.asarray(times, dtype=np.float64),
        options=ModelCoreOptions(
            other_axes=(),
            store_history=True,
            dtype=cast("DTypeLike", y0.dtype),
        ),
    )
    xp = _namespace_of(y0)
    core.set_initial_state(cast("Array", xp.reshape(y0, (n_states, 1))))
    return core


def _run_jax_fixed_explicit_trajectory(
    solver: CoreSolver,
    rhs: Callable[[Scalar, Array], Array],
    *,
    method: SolverMethod,
    times: np.ndarray,
    fixed_max_step: float | None,
) -> None:
    """Run a fixed explicit solve as one compact JAX scan."""
    import jax  # noqa: PLC0415

    initial_state = solver.core.get_current_state()
    xp = _namespace_of(initial_state)

    if times.size == 1:
        solver.core.apply_trajectory(
            cast("Array", xp.expand_dims(initial_state, axis=0))
        )
        return

    interval_steps = tuple(
        fixed_step_sizes(float(start), float(end), fixed_max_step)
        for start, end in itertools.pairwise(times)
    )
    flat_step_sizes: list[float] = []
    flat_step_times: list[float] = []
    for start, steps in zip(times[:-1], interval_steps, strict=True):
        step_time = float(start)
        for step_size in steps:
            flat_step_times.append(step_time)
            flat_step_sizes.append(step_size)
            step_time += step_size
    output_indices = (
        np.cumsum(
            np.asarray(tuple(len(steps) for steps in interval_steps), dtype=np.int64)
        )
        - 1
    )
    step_times = cast(
        "Array",
        xp.asarray(tuple(flat_step_times), dtype=initial_state.dtype),
    )
    step_sizes = cast(
        "Array",
        xp.asarray(tuple(flat_step_sizes), dtype=initial_state.dtype),
    )

    if method is SolverMethod.DOPRI5:
        first_state, first_stage = solver.fixed_explicit_step(
            rhs,
            method=method.value,
            t=_array_item(step_times, 0),
            dt=_array_item(step_sizes, 0),
            y=initial_state,
        )
        if first_stage is None:
            msg = "Dormand--Prince fixed steps must return an FSAL stage."
            raise RuntimeError(msg)

        if len(flat_step_sizes) == 1:
            internal_tail = cast("Array", xp.expand_dims(first_state, axis=0))
        else:

            def advance_dopri5(
                carry: tuple[Array, Array],
                step: tuple[Array, Array],
            ) -> tuple[tuple[Array, Array], Array]:
                state, stage = carry
                time, dt = step
                next_state, next_stage = solver.fixed_explicit_step(
                    rhs,
                    method=method.value,
                    t=time,
                    dt=dt,
                    y=state,
                    first_stage=stage,
                )
                if next_stage is None:
                    msg = "Dormand--Prince fixed steps must return an FSAL stage."
                    raise RuntimeError(msg)
                return (next_state, next_stage), next_state

            (_final_state, _final_stage), remaining_tail = jax.lax.scan(
                advance_dopri5,
                (first_state, first_stage),
                (
                    _array_item(step_times, slice(1, None)),
                    _array_item(step_sizes, slice(1, None)),
                ),
            )
            internal_tail = cast(
                "Array",
                xp.concat(
                    (xp.expand_dims(first_state, axis=0), remaining_tail),
                    axis=0,
                ),
            )
    else:

        def advance(state: Array, step: tuple[Array, Array]) -> tuple[Array, Array]:
            time, dt = step
            next_state, _next_stage = solver.fixed_explicit_step(
                rhs,
                method=method.value,
                t=time,
                dt=dt,
                y=state,
            )
            return next_state, next_state

        _final_state, internal_tail = jax.lax.scan(
            advance,
            initial_state,
            (step_times, step_sizes),
        )

    saved_tail = _array_item(internal_tail, output_indices)
    trajectory = cast(
        "Array",
        xp.concat((xp.expand_dims(initial_state, axis=0), saved_tail), axis=0),
    )
    solver.core.apply_trajectory(trajectory)


def _last_state(core: ModelCore, *, n_state: int) -> Array:
    """Return the final flat state stored by one solver core."""
    states = _extract_states_2d(core, n_state=n_state)
    return _array_item(states, (-1, slice(None)))


def _format_result(times: np.ndarray, states: Array) -> Array:
    """Prepend the evaluation-time column to provider state history."""
    xp = _namespace_of(states)
    time_values = cast("Array", xp.asarray(times, dtype=states.dtype))
    time_column = cast("Array", xp.expand_dims(time_values, axis=1))
    return cast("Array", xp.concat((time_column, states), axis=1))


def _split_numpy_seeds(seed: int | None) -> tuple[int | None, int | None]:
    """Derive independent reproducible Poisson and SSA seeds."""
    if seed is None:
        return None, None
    poisson_sequence, ssa_sequence = np.random.SeedSequence(seed).spawn(2)
    poisson_seed = int(poisson_sequence.generate_state(1, dtype=np.uint64)[0])
    ssa_seed = int(ssa_sequence.generate_state(1, dtype=np.uint64)[0])
    return poisson_seed, ssa_seed


def _resolve_samplers(
    y0: Array,
    config: OpEngineEngineConfig,
    kwargs: Mapping[str, object],
) -> tuple[PoissonSampler | None, SSASampler | None]:
    """Resolve injected samplers or seeded NumPy conveniences.

    Returns:
        Poisson and SSA samplers required by the configured method.
    """
    poisson_obj = kwargs.get("poisson_sampler")
    ssa_obj = kwargs.get("ssa_sampler")
    if poisson_obj is not None and not callable(poisson_obj):
        msg = "poisson_sampler must be callable."
        raise TypeError(msg)
    if ssa_obj is not None and not callable(ssa_obj):
        msg = "ssa_sampler must be callable."
        raise TypeError(msg)
    poisson_sampler = cast("PoissonSampler | None", poisson_obj)
    ssa_sampler = cast("SSASampler | None", ssa_obj)

    xp = _namespace_of(y0)
    poisson_seed, ssa_seed = _split_numpy_seeds(config.random_seed)
    if config.stochastic_method is StochasticMethod.TAU_LEAPING:
        if poisson_sampler is None and xp is np:
            poisson_sampler = NumpyPoissonSampler(poisson_seed)
        if poisson_sampler is None:
            msg = (
                "A poisson_sampler preserving the state array namespace is "
                "required for non-NumPy stochastic runs."
            )
            raise TypeError(msg)
    elif config.stochastic_method is StochasticMethod.DIRECT_SSA:
        if ssa_sampler is None and xp is np:
            ssa_sampler = NumpySSASampler(ssa_seed)
        if ssa_sampler is None:
            msg = (
                "An ssa_sampler preserving the state array namespace is required "
                "for non-NumPy stochastic runs."
            )
            raise TypeError(msg)
    return poisson_sampler, ssa_sampler


class _MonotonicPoissonSampler:
    """Keep functional sampler indices monotonic across hybrid subproblems."""

    def __init__(self, sampler: PoissonSampler) -> None:
        self._sampler = sampler
        self._draw_index = 0

    def __call__(self, mean: Array, _local_index: int, /) -> Array:
        """Forward one draw with a run-global index."""
        draw_index = self._draw_index
        self._draw_index += 1
        return self._sampler(mean, draw_index)


class _MonotonicSSASampler:
    """Keep exact-event sampler indices monotonic across hybrid subproblems."""

    def __init__(self, sampler: SSASampler) -> None:
        self._sampler = sampler
        self._draw_index = 0

    def __call__(
        self,
        total_rate: Array,
        probabilities: Array,
        _local_index: int,
        /,
    ) -> Any:  # noqa: ANN401
        """Forward one draw with a run-global index."""
        draw_index = self._draw_index
        self._draw_index += 1
        return self._sampler(total_rate, probabilities, draw_index)


def _run_stochastic_core(
    core: ModelCore,
    network: CompiledReactionNetwork,
    config: OpEngineEngineConfig,
    *,
    poisson_sampler: PoissonSampler | None,
    ssa_sampler: SSASampler | None,
) -> None:
    """Run the configured discrete method on one prepared core."""
    state = core.get_current_state()
    xp = _namespace_of(state)
    stoichiometry = cast(
        "Array",
        xp.asarray(network.stoichiometry, dtype=state.dtype),
    )
    if config.stochastic_method is StochasticMethod.TAU_LEAPING:
        if poisson_sampler is None:
            msg = "Tau-leaping sampler resolution is inconsistent."
            raise RuntimeError(msg)
        solver = TauLeapingSolver(core, stoichiometry)
        solver.run(
            network.propensity,
            poisson_sampler,
            config=TauLeapingConfig(
                max_step=config.tau_max_step,
                max_steps=config.stochastic_max_steps,
            ),
        )
        return

    if ssa_sampler is None:
        msg = "Direct-SSA sampler resolution is inconsistent."
        raise RuntimeError(msg)
    exact_solver = DirectSSASolver(core, stoichiometry)
    exact_solver.run(
        network.propensity,
        ssa_sampler,
        config=DirectSSAConfig(max_events=config.ssa_max_events),
    )


def _run_pure_stochastic(
    times: np.ndarray,
    y0: Array,
    network: CompiledReactionNetwork,
    config: OpEngineEngineConfig,
    kwargs: Mapping[str, object],
) -> Array:
    """Run all typed reactions as one discrete process."""
    poisson_sampler, ssa_sampler = _resolve_samplers(y0, config, kwargs)
    core = _make_core(times, y0)
    _run_stochastic_core(
        core,
        network,
        config,
        poisson_sampler=poisson_sampler,
        ssa_sampler=ssa_sampler,
    )
    return _extract_states_2d(core, n_state=network.n_state)


def _run_hybrid(
    times: np.ndarray,
    y0: Array,
    *,
    network: CompiledReactionNetwork,
    config: OpEngineEngineConfig,
    kwargs: Mapping[str, object],
    rhs: Callable[[float, Array], Array],
    run_config: RunConfig,
    operators: CoreOperators | StageOperatorFactory | None,
    operator_axis: str | int,
) -> Array:
    """Run deterministic residual then selected jump channels per interval.

    Returns:
        Flat state history from first-order deterministic-then-stochastic Lie
        splitting on the requested output intervals.
    """
    poisson_sampler, ssa_sampler = _resolve_samplers(y0, config, kwargs)
    indexed_poisson = (
        None if poisson_sampler is None else _MonotonicPoissonSampler(poisson_sampler)
    )
    indexed_ssa = None if ssa_sampler is None else _MonotonicSSASampler(ssa_sampler)

    def residual_rhs(time: float, state: Array) -> Array:
        xp = _namespace_of(state)
        return cast(
            "Array",
            xp.subtract(rhs(time, state), network.mean_drift(time, state)),
        )

    n_state = network.n_state
    state = y0
    history = [state]
    for start, stop in itertools.pairwise(times):
        interval = np.asarray([start, stop], dtype=np.float64)
        deterministic_core = _make_core(interval, state)
        deterministic_solver = CoreSolver(
            deterministic_core,
            operators=operators,
            operator_axis=operator_axis,
        )
        deterministic_solver.run(residual_rhs, config=run_config)
        state = _last_state(deterministic_core, n_state=n_state)

        stochastic_core = _make_core(interval, state)
        _run_stochastic_core(
            stochastic_core,
            network,
            config,
            poisson_sampler=indexed_poisson,
            ssa_sampler=indexed_ssa,
        )
        state = _last_state(stochastic_core, n_state=n_state)
        history.append(state)

    xp = _namespace_of(y0)
    return cast("Array", xp.stack(tuple(history), axis=0))


def _unwrap_parameter_values(
    values: Mapping[IdentifierString, ParameterValue],
) -> dict[IdentifierString, object]:
    """Return raw parameter payloads for binding to a system stepper."""
    return {name: value.value for name, value in values.items()}


def _as_initial_scalar(
    value: object,
    *,
    name: str,
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array:
    """Coerce one state-cell value without leaving the run namespace."""
    arr = cast("Array", xp.asarray(value, dtype=dtype))
    if arr.shape != ():
        msg = f"Initial state value for {name!r} must be scalar; got {arr.shape}."
        raise ValueError(msg)
    return arr


def _shaped_initial_scalar(
    *,
    state_name: str,
    entry: Mapping[str, object],
    params: Mapping[IdentifierString, ParameterValue],
    raw_params: Mapping[IdentifierString, object],
    axis_labels: Mapping[str, object],
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array:
    """Resolve one expanded state cell from a shaped parameter value."""
    shaped_name = entry.get("shaped")
    if not isinstance(shaped_name, str):
        msg = f"Initial state entry for {state_name!r} has no shaped parameter name."
        raise TypeError(msg)
    if shaped_name not in params:
        msg = (
            f"Initial state for {state_name!r} references shaped parameter "
            f"{shaped_name!r}, which is not present in params."
        )
        raise KeyError(msg)
    coords = entry.get("coords")
    if not isinstance(coords, Mapping):
        msg = f"Initial state entry for {state_name!r} has no coordinate mapping."
        raise TypeError(msg)

    pv = params[shaped_name]
    indices: list[int] = []
    for axis_name in pv.shape.axis_names:
        coord = coords.get(axis_name)
        labels = axis_labels.get(axis_name)
        if not isinstance(coord, str) or not isinstance(labels, tuple | list):
            msg = (
                f"Initial state for {state_name!r} cannot resolve coordinate "
                f"{coord!r} on axis {axis_name!r}."
            )
            raise KeyError(msg)
        string_labels = tuple(str(label) for label in labels)
        if coord not in string_labels:
            msg = (
                f"Initial state for {state_name!r} references unknown coordinate "
                f"{coord!r} on axis {axis_name!r}."
            )
            raise KeyError(msg)
        indices.append(string_labels.index(coord))

    shaped_value = cast("Array", raw_params[shaped_name])
    value = _array_item(shaped_value, tuple(indices))
    return _as_initial_scalar(value, name=state_name, xp=xp, dtype=dtype)


def _assemble_option_initial_state(
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array | None:
    """Assemble state from the metadata contract published by op_system."""
    state_names = system.option("state_names", None)
    if not isinstance(state_names, tuple | list):
        return None

    init_map = system.option("initial_state", None) or {}
    if not isinstance(init_map, Mapping):
        msg = "system option 'initial_state' must be a mapping when provided."
        raise TypeError(msg)
    raw_initial_state = _unwrap_parameter_values(initial_state)
    raw_params = _unwrap_parameter_values(params)
    axis_labels = system.option("axis_labels", None) or {}
    if not isinstance(axis_labels, Mapping):
        msg = "system option 'axis_labels' must be a mapping when provided."
        raise TypeError(msg)

    values: list[Array] = []
    for state_name_obj in state_names:
        state_name = str(state_name_obj)
        if state_name in raw_initial_state:
            values.append(
                _as_initial_scalar(
                    raw_initial_state[state_name],
                    name=state_name,
                    xp=xp,
                    dtype=dtype,
                )
            )
            continue

        entry = init_map.get(state_name)
        if entry is None:
            values.append(_as_initial_scalar(0.0, name=state_name, xp=xp, dtype=dtype))
        elif isinstance(entry, str):
            if entry not in raw_params:
                msg = (
                    f"Initial state for {state_name!r} references parameter "
                    f"{entry!r}, which is not present in params."
                )
                raise KeyError(msg)
            values.append(
                _as_initial_scalar(
                    raw_params[entry],
                    name=state_name,
                    xp=xp,
                    dtype=dtype,
                )
            )
        elif isinstance(entry, Mapping):
            values.append(
                _shaped_initial_scalar(
                    state_name=state_name,
                    entry=entry,
                    params=params,
                    raw_params=raw_params,
                    axis_labels=axis_labels,
                    xp=xp,
                    dtype=dtype,
                )
            )
        else:
            values.append(
                _as_initial_scalar(entry, name=state_name, xp=xp, dtype=dtype)
            )

    return cast("Array", xp.stack(tuple(values), axis=0))


def _assemble_initial_state(
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
    model_state: ModelStateSpecification | None,
) -> Array:
    """Assemble flepimop2 state entries in their declared semantic order."""
    xp, dtype = _numerical_context(system, initial_state, params)
    option_state = _assemble_option_initial_state(
        system,
        initial_state,
        params,
        xp=xp,
        dtype=dtype,
    )
    if option_state is not None:
        return option_state
    if model_state is None:
        msg = "model_state must be provided to assemble the initial state."
        raise ValueError(msg)
    values = [
        cast(
            "Array",
            xp.reshape(xp.asarray(initial_state[name].value, dtype=dtype), (-1,)),
        )
        for name in model_state.parameter_names
    ]
    return cast("Array", xp.concat(tuple(values), axis=0))


class OpEngineFlepimop2Engine(EngineABC):
    """flepimop2 engine adapter backed by op_engine.CoreSolver."""

    module: Literal["flepimop2.engine.op_engine"] = "flepimop2.engine.op_engine"
    state_change: StateChangeEnum
    config: OpEngineEngineConfig = Field(default_factory=OpEngineEngineConfig)
    _last_adaptive_schedule: AdaptiveSchedule | None = PrivateAttr(default=None)

    @property
    def last_adaptive_schedule(self) -> AdaptiveSchedule | None:
        """Return the schedule discovered or replayed by the latest run."""
        return self._last_adaptive_schedule

    def validate_system(self, system: SystemABC) -> list[ValidationIssue] | None:
        """Validate system compatibility with engine config."""
        issues: list[ValidationIssue] = []

        if system.state_change != self.state_change:
            issues.append(
                ValidationIssue(
                    msg=(
                        f"Engine state change type, '{self.state_change}', is not "
                        "compatible with system state change type "
                        f"'{system.state_change}'."
                    ),
                    kind="incompatible_system",
                ),
            )

        mode = self.config.mode
        if mode is not ExecutionMode.DETERMINISTIC:
            reactions = system.option("reactions", None)
            if not isinstance(reactions, tuple | list) or not reactions:
                issues.append(
                    ValidationIssue(
                        msg=(
                            f"{mode.value.capitalize()} mode requires typed "
                            "system.option('reactions') artifacts."
                        ),
                        kind="missing_reactions",
                    ),
                )
            elif mode is ExecutionMode.HYBRID:
                available = {
                    name
                    for reaction in reactions
                    if isinstance((name := getattr(reaction, "name", None)), str)
                }
                missing = sorted(set(self.config.stochastic_reactions) - available)
                if missing:
                    issues.append(
                        ValidationIssue(
                            msg=(
                                "Unknown stochastic reaction names: "
                                f"{', '.join(missing)}."
                            ),
                            kind="unknown_reactions",
                        ),
                    )
            if not isinstance(system.option("template_shapes", None), Mapping):
                issues.append(
                    ValidationIssue(
                        msg=(
                            f"{mode.value.capitalize()} mode requires typed "
                            "system.option('template_shapes') metadata."
                        ),
                        kind="missing_reaction_layout",
                    ),
                )
            if mode is ExecutionMode.STOCHASTIC:
                return issues or None

        method = self.config.method
        is_imex = method.is_imex

        if is_imex and not _has_operator_specs(
            _coerce_operator_specs(self.config.operators),
        ):
            sys_ops = system.option("operators", None)
            descriptors = typed_operator_descriptors(sys_ops)
            has_system_operators = bool(descriptors) or _has_operator_specs(
                _coerce_operator_specs(sys_ops)
            )
            if not has_system_operators:
                issues.append(
                    ValidationIssue(
                        msg=(
                            f"IMEX method '{method}' requires operator matrices, "
                            "but neither the engine config nor "
                            "system.option('operators') provides them."
                        ),
                        kind="missing_operators",
                    ),
                )

        if method.is_implicit:
            jac = system.option("jacobian", None)
            if jac is None:
                issues.append(
                    ValidationIssue(
                        msg=(
                            f"Implicit/Rosenbrock method '{method}' requires a "
                            "Jacobian callable, but system.option('jacobian') "
                            "is not provided."
                        ),
                        kind="missing_jacobian",
                    ),
                )

        if method.is_nonlinear:
            rhs_jacobian = system.option("rhs_jacobian", None)
            if not callable(rhs_jacobian):
                issues.append(
                    ValidationIssue(
                        msg=(
                            f"Nonlinear method '{method}' requires a full RHS "
                            "Jacobian callable from "
                            "system.option('rhs_jacobian')."
                        ),
                        kind="missing_rhs_jacobian",
                    ),
                )

        return issues or None

    def run(
        self,
        system: SystemABC,
        eval_times: Float64NDArray,
        initial_state: dict[IdentifierString, ParameterValue],
        params: Mapping[IdentifierString, ParameterValue],
        model_state: ModelStateSpecification | None = None,
        *,
        adaptive_schedule: AdaptiveSchedule | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Float64NDArray:
        """Execute a simulation, optionally replaying a frozen adaptive mesh."""
        self._last_adaptive_schedule = None
        result = self._execute(
            system,
            eval_times,
            initial_state,
            params,
            model_state=model_state,
            adaptive_schedule=adaptive_schedule,
            **kwargs,
        )
        self._last_adaptive_schedule = result.schedule
        return cast("Float64NDArray", result.trajectory)

    def run_adaptive(
        self,
        system: SystemABC,
        eval_times: Float64NDArray,
        initial_state: dict[IdentifierString, ParameterValue],
        params: Mapping[IdentifierString, ParameterValue],
        model_state: ModelStateSpecification | None = None,
        *,
        schedule: AdaptiveSchedule | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> AdaptiveRunResult:
        """Discover or replay an adaptive schedule with explicit metadata.

        Pass no schedule for eager controller discovery. Pass the returned
        schedule to replay the frozen accepted mesh under JIT or automatic
        differentiation. Resulting gradients are conditional on that mesh.

        Returns:
            Trajectory, provider schedule artifact, and optional nonlinear
            diagnostics.

        Raises:
            ValueError: If the engine is not deterministic and adaptive.
        """
        if self.config.mode is not ExecutionMode.DETERMINISTIC:
            msg = "Adaptive schedule discovery and replay require deterministic mode."
            raise ValueError(msg)
        if not self.config.adaptive:
            msg = "Adaptive schedule discovery and replay require adaptive=True."
            raise ValueError(msg)

        self._last_adaptive_schedule = None
        result = self._execute(
            system,
            eval_times,
            initial_state,
            params,
            model_state=model_state,
            adaptive_schedule=schedule,
            **kwargs,
        )
        if result.schedule is None:
            msg = "Adaptive execution completed without recording a schedule."
            raise RuntimeError(msg)
        self._last_adaptive_schedule = result.schedule
        return AdaptiveRunResult(
            trajectory=result.trajectory,
            schedule=result.schedule,
            diagnostics=result.diagnostics,
        )

    def _execute(
        self,
        system: SystemABC,
        eval_times: Float64NDArray,
        initial_state: dict[IdentifierString, ParameterValue],
        params: Mapping[IdentifierString, ParameterValue],
        model_state: ModelStateSpecification | None = None,
        *,
        adaptive_schedule: AdaptiveSchedule | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> _ExecutionResult:
        """Execute one provider run and retain schedule-aware metadata."""
        times = _as_float64_1d(eval_times, name="eval_times")
        _ensure_strictly_increasing(times, name="eval_times")
        mode = self.config.mode
        if adaptive_schedule is not None:
            if not isinstance(adaptive_schedule, AdaptiveSchedule):
                msg = "adaptive_schedule must be an AdaptiveSchedule"
                raise TypeError(msg)
            if mode is not ExecutionMode.DETERMINISTIC:
                msg = "Adaptive schedule replay requires deterministic mode."
                raise ValueError(msg)
            if not self.config.adaptive:
                msg = "Adaptive schedule replay requires adaptive=True."
                raise ValueError(msg)
            adaptive_schedule.validate_config(self.config)

        raw_params = _unwrap_parameter_values(params)
        y0 = _assemble_initial_state(system, initial_state, params, model_state)
        n_state = int(y0.shape[0])
        method = self.config.method

        if mode is ExecutionMode.STOCHASTIC:
            stochastic_network = compile_reaction_network(
                system,
                raw_params,
                n_state=n_state,
            )
            stochastic_states = _run_pure_stochastic(
                times,
                y0,
                stochastic_network,
                self.config,
                kwargs,
            )
            return _ExecutionResult(
                trajectory=_format_result(times, stochastic_states),
            )

        nonlinear: NonlinearMethodConfig | None = None
        if method.is_nonlinear:
            rhs_jacobian = system.option("rhs_jacobian", None)
            if not callable(rhs_jacobian):
                msg = (
                    f"Nonlinear method '{method}' requires callable system option "
                    "'rhs_jacobian'."
                )
                raise ValueError(msg)
            nonlinear_solver = system.option("nonlinear_solver", None)
            if nonlinear_solver is None:
                nonlinear = NonlinearMethodConfig(rhs_jacobian=rhs_jacobian)
            else:
                nonlinear = NonlinearMethodConfig(
                    rhs_jacobian=rhs_jacobian,
                    solver=cast("NonlinearSolver", nonlinear_solver),
                )

        run_cfg = self.config.to_run_config(nonlinear=nonlinear)
        is_imex = method.is_imex
        operators = run_cfg.operators
        compiled_system_operators = False

        if is_imex and not _has_operator_specs(operators):
            system_operators = system.option("operators", None)
            descriptors = typed_operator_descriptors(system_operators)
            if descriptors:
                operators = compile_operator_descriptors(
                    descriptors,
                    method=method.value,
                    state_names=system.option("state_names", None),
                    axis_order=system.option("axis_order", None),
                    axis_labels=system.option("axis_labels", None),
                    axis_coords=system.option("axis_coords", None),
                    params=raw_params,
                    reference=y0,
                )
                compiled_system_operators = True
            else:
                operators = _coerce_operator_specs(system_operators) or operators
        run_cfg = replace(run_cfg, operators=operators)

        if is_imex and not _has_operator_specs(operators):
            msg = (
                f"IMEX method '{run_cfg.method}' requires operators from engine config "
                "or system option 'operators'."
            )
            raise ValueError(msg)

        if method.is_implicit:
            jacobian = system.option("jacobian", None)
            if callable(jacobian):
                run_cfg = replace(run_cfg, jacobian=jacobian)

        operator_axis: str | int = (
            "state" if compiled_system_operators else self.config.operator_axis
        )
        if operator_axis == "state":
            system_axis = system.option("operator_axis", None)
            if not compiled_system_operators and isinstance(system_axis, str | int):
                operator_axis = system_axis

        # Bind raw payloads once. Systems such as op_system merge their own
        # static mixing kernels and should never receive ParameterValue wrappers.
        stepper: SystemProtocol = system.bind(params=raw_params)
        rhs = _rhs_from_stepper(stepper, n_state=n_state)

        if mode is ExecutionMode.HYBRID:
            hybrid_network = compile_reaction_network(
                system,
                raw_params,
                n_state=n_state,
                reaction_names=self.config.stochastic_reactions,
            )
            hybrid_states = _run_hybrid(
                times,
                y0,
                network=hybrid_network,
                config=self.config,
                kwargs=kwargs,
                rhs=rhs,
                run_config=run_cfg,
                operators=operators.default if is_imex else None,
                operator_axis=operator_axis,
            )
            return _ExecutionResult(
                trajectory=_format_result(times, hybrid_states),
            )

        core = _make_core(times, y0)

        solver = CoreSolver(
            core,
            operators=operators.default if is_imex else None,
            operator_axis=operator_axis,
        )
        state_namespace = _namespace_of(y0)
        use_jax_scan = (
            _is_jax_namespace(state_namespace)
            and not run_cfg.adaptive
            and method
            in {
                SolverMethod.EULER,
                SolverMethod.HEUN,
                SolverMethod.RK4,
                SolverMethod.DOPRI5,
            }
        )
        diagnostics: NonlinearIntegrationDiagnostics | None = None
        if adaptive_schedule is not None:
            diagnostics = solver.replay_adaptive_schedule(
                rhs,
                adaptive_schedule.step_schedule,
                config=run_cfg,
            )
        elif use_jax_scan:
            _run_jax_fixed_explicit_trajectory(
                solver,
                rhs,
                method=method,
                times=times,
                fixed_max_step=run_cfg.fixed_max_step,
            )
        else:
            diagnostics = solver.run(rhs, config=run_cfg)

        states = _extract_states_2d(core, n_state=n_state)
        step_schedule = solver.last_adaptive_schedule
        schedule = adaptive_schedule
        if schedule is None and step_schedule is not None:
            schedule = AdaptiveSchedule.from_core(step_schedule, self.config)
        return _ExecutionResult(
            trajectory=_format_result(times, states),
            schedule=schedule,
            diagnostics=diagnostics,
        )


__all__ = [
    "AdaptiveRunResult",
    "AdaptiveSchedule",
    "ExecutionMode",
    "OpEngineEngineConfig",
    "OpEngineFlepimop2Engine",
    "SolverMethod",
    "StochasticMethod",
]
