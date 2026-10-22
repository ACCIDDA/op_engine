"""Current compiler/native Diffrax boundary and separately qualified transforms."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.compiled_hierarchical import compiled_hierarchical_term
from benchmarks.hierarchical_ode import (
    HierarchicalProblem,
    initial_state,
    nominal_parameters,
    reference_rhs,
)
from op_engine.diffrax import compiled_diffrax_term

if TYPE_CHECKING:
    from collections.abc import Iterator

    from op_engine import Array

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
d = pytest.importorskip("diffrax")
compiler = pytest.importorskip("op_system")


@pytest.fixture(autouse=True)
def _native_precision() -> Iterator[None]:
    """Enable float64 inside each test and restore the caller's JAX settings."""
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        yield


def _compiled(multiplier: int = 1) -> Any:  # noqa: ANN401 - optional compiler artifact
    """Build a current shaped reversible transfer with a scalar return rate.

    Returns:
        Two-template compiler artifact with shaped rates absent from param_names.
    """
    net = f"{multiplier}*rate[group]*X[group] - back*Z[group]"
    return compiler.compile_spec({
        "kind": "expr",
        "axes": [{"name": "group", "coords": ["left", "right"]}],
        "state": ["X[group]", "Z[group]"],
        "aliases": {"net[group]": net},
        "equations": {"X[group]": "-net[group]", "Z[group]": "net[group]"},
    })


def _functions(layout: str, dtype: str, *, adaptive: bool = False) -> tuple[Any, Any]:
    """Create the real solve and an independently derived closed-form solution.

    Returns:
        Solve and analytic functions in X-left/right, Z-left/right output order.
    """
    compiled = _compiled()
    assert compiled.param_names == ("back",)
    assert dict(compiled.meta["shaped_params"]) == {"rate": ("group",)}
    term = compiled_diffrax_term(compiled, state_layout=layout)
    times = jnp.asarray([0.0, 0.25, 0.5, 1.0], dtype=getattr(jnp, dtype))
    mesh = jnp.linspace(0, 1, 17, dtype=getattr(jnp, dtype))

    def solve(parameters: Array, adjoint: object) -> Array:
        initial = {"X": parameters[3:], "Z": jnp.zeros(2, dtype=parameters.dtype)}
        y0 = initial if layout == "pytree" else jnp.concatenate(tuple(initial.values()))
        controller = (
            d.PIDController(
                rtol=2e-6 if dtype == "float32" else 2e-9,
                atol=1e-8 if dtype == "float32" else 1e-11,
            )
            if adaptive
            else d.StepTo(mesh)
        )
        solution = d.diffeqsolve(
            term,
            d.Dopri5(scan_kind="bounded"),
            t0=times[0],
            t1=times[-1],
            dt0=jnp.asarray(0.01, dtype=parameters.dtype) if adaptive else None,
            y0=y0,
            args={"rate": parameters[:2], "back": parameters[2]},
            saveat=d.SaveAt(ts=times),
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=128 if adaptive else 16,
        )
        result = solution.ys
        return cast(
            "Array",
            (
                jnp.concatenate((result["X"], result["Z"]), axis=1)
                if layout == "pytree"
                else result
            ),
        )

    def reference(parameters: Array) -> Array:
        forward, back, amplitude = parameters[:2], parameters[2], parameters[3:]
        total = forward + back
        first = (
            amplitude[None, :]
            * (back + forward[None, :] * jnp.exp(-times[:, None] * total[None, :]))
            / total[None, :]
        )
        return cast(
            "Array", jnp.concatenate((first, amplitude[None, :] - first), axis=1)
        )

    return solve, reference


@pytest.mark.parametrize("layout", ["flat", "pytree"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_current_compiler_values_changed_args_and_initial_gradients(
    layout: str, dtype: str, *, adaptive: bool
) -> None:
    """Closed-form trajectories and reverse derivatives include runtime IC effects."""
    solve, reference = _functions(layout, dtype, adaptive=adaptive)
    parameters = jnp.asarray([0.7, 1.1, 0.2, 2.0, 3.0], dtype=getattr(jnp, dtype))
    adjoint = d.RecursiveCheckpointAdjoint()
    forward = jax.jit(lambda p: solve(p, adjoint))
    tolerance = 8e-6 if dtype == "float32" else 2e-8
    for values in [parameters, parameters * 1.2]:
        np.testing.assert_allclose(
            forward(values), reference(values), rtol=tolerance, atol=tolerance
        )

    def objective(p: Array) -> Array:
        return cast(
            "Array",
            jnp.sum(solve(p, adjoint)[-1] * jnp.asarray([1, 2, 3, 5], dtype=p.dtype)),
        )

    def analytic(p: Array) -> Array:
        return cast(
            "Array",
            jnp.sum(reference(p)[-1] * jnp.asarray([1, 2, 3, 5], dtype=p.dtype)),
        )

    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(parameters),
        jax.grad(analytic)(parameters),
        rtol=tolerance * 4,
        atol=tolerance * 4,
    )
    assert forward(parameters).dtype == parameters.dtype


@pytest.mark.parametrize("layout", ["flat", "pytree"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_forward_mode_and_independent_parameter_batches(
    layout: str, dtype: str
) -> None:
    """Explicit ForwardMode JVP and vmap agree with independent analytic derivatives."""
    solve, reference = _functions(layout, dtype)
    p = jnp.asarray([0.7, 1.1, 0.2, 2.0, 3.0], dtype=getattr(jnp, dtype))

    def evaluate(values: Array) -> Array:
        return cast("Array", solve(values, d.ForwardMode()))

    tolerance = 1e-5 if dtype == "float32" else 3e-8
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(evaluate))(p),
        jax.jacfwd(reference)(p),
        rtol=tolerance,
        atol=tolerance,
    )
    batch = jnp.stack((p, p * 1.2))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(batch),
        jax.vmap(reference)(batch),
        rtol=tolerance,
        atol=tolerance,
    )


def test_bounded_direct_adjoint_hessian_is_separately_qualified() -> None:
    """Check a float64 PyTree fixed-mesh Hessian against closed-form derivatives."""
    solve, reference = _functions("pytree", "float64")
    p = jnp.asarray([0.7, 1.1, 0.2, 2.0, 3.0], dtype=jnp.float64)

    def objective(values: Array) -> Array:
        return cast("Array", jnp.sum(solve(values, d.DirectAdjoint())[-1] ** 2))

    def analytic(values: Array) -> Array:
        return cast("Array", jnp.sum(reference(values)[-1] ** 2))

    np.testing.assert_allclose(
        jax.jit(jax.hessian(objective))(p),
        jax.hessian(analytic)(p),
        rtol=2e-7,
        atol=2e-7,
    )


def test_equal_layouts_from_different_compiled_models_keep_their_own_rhs() -> None:
    """Same shape/name models and changed values cannot alias a global solve cache."""
    first, second = (
        compiled_diffrax_term(_compiled()),
        compiled_diffrax_term(_compiled(2)),
    )
    y = {"X": jnp.asarray([2.0, 3.0]), "Z": jnp.zeros(2)}
    args = {"rate": jnp.asarray([0.7, 1.1]), "back": jnp.asarray(0.2)}
    np.testing.assert_allclose(first.vf(0.0, y, args)["Z"], [1.4, 3.3])
    np.testing.assert_allclose(second.vf(0.0, y, args)["Z"], [2.8, 6.6])
    other = {**args, "rate": jnp.asarray([1.0, 2.0])}
    np.testing.assert_allclose(first.vf(0.0, y, other)["Z"], [2.0, 6.0])
    np.testing.assert_allclose(first.vf(0.0, y, args)["Z"], [1.4, 3.3])


@pytest.mark.parametrize(
    "fault",
    [
        "host",
        "keys",
        "shape",
        "dtype",
        "parameter_keys",
        "parameter_shape",
        "parameter_dtype",
    ],
)
def test_native_state_parameter_contract_fails_explicitly(fault: str) -> None:
    """The adapter never transfers host arrays, reshapes inputs or drops parameters."""
    term = compiled_diffrax_term(_compiled())
    y: Any = {"X": jnp.asarray([2.0, 3.0]), "Z": jnp.zeros(2)}
    args = {"rate": jnp.asarray([0.7, 1.1]), "back": jnp.asarray(0.2)}
    expected: type[Exception] = ValueError
    if fault == "host":
        y["X"] = np.asarray([2.0, 3.0])
        expected = TypeError
    elif fault == "keys":
        y.pop("Z")
    elif fault == "shape":
        y["X"] = jnp.ones(3)
    elif fault == "dtype":
        y["X"] = jnp.asarray([2, 3])
        expected = TypeError
    elif fault == "parameter_keys":
        args["extra"] = jnp.asarray(1.0)
    elif fault == "parameter_shape":
        args["rate"] = jnp.asarray(0.7)
    else:
        args["rate"] = jnp.asarray([0.7, 1.1], dtype=jnp.float32)
        expected = TypeError
    with pytest.raises(expected, match="Diffrax"):
        term.vf(0.0, y, args)


@pytest.mark.parametrize("case", ["operators", "history", "table", "missing_pytree"])
def test_unsupported_compiler_contributions_are_not_silently_dropped(case: str) -> None:
    """Actual history/table metadata and declared operators require another driver."""
    compiled = _compiled()
    if case == "operators":
        compiled = replace(compiled, operators=(object(),))
    elif case == "history":
        compiled = compiler.compile_spec({
            "kind": "expr",
            "state": ["x"],
            "equations": {
                "x": (
                    "convolve_history(x, kernel=gamma, window=14, "
                    "kernel_shape=3, kernel_scale=2)"
                )
            },
        })
    elif case == "table":
        compiled = compiler.compile_spec({
            "kind": "expr",
            "state": ["x"],
            "axes": [{"name": "time", "type": "continuous", "coords": [0.0, 0.5, 1.0]}],
            "time_interpolation": "previous",
            "equations": {"x": "rate[time]*x"},
        })
    else:
        compiled = replace(compiled, pytree_eval_fn=None)
    with pytest.raises(NotImplementedError, match="Diffrax"):
        compiled_diffrax_term(compiled)


def test_bad_rhs_output_is_rejected_without_casting() -> None:
    """A mismatched producer output cannot silently change the numerical dtype."""
    compiled = replace(
        _compiled(),
        pytree_eval_fn=lambda _t, y, **_p: {
            k: value.astype(jnp.float32) for k, value in y.items()
        },
    )
    term = compiled_diffrax_term(compiled)
    with pytest.raises(TypeError, match="output dtype"):
        term.vf(
            0.0,
            {"X": jnp.ones(2), "Z": jnp.zeros(2)},
            {"rate": jnp.ones(2), "back": jnp.asarray(0.2)},
        )


def test_solver_failure_result_remains_available_to_the_caller() -> None:
    """A failed native solve retains Diffrax's explicit result and throw policy."""
    term = compiled_diffrax_term(_compiled())
    arguments = {
        "terms": term,
        "solver": d.Dopri5(scan_kind="bounded"),
        "t0": 0.0,
        "t1": 1.0,
        "dt0": None,
        "y0": {"X": jnp.ones(2), "Z": jnp.zeros(2)},
        "args": {"rate": jnp.ones(2), "back": jnp.asarray(0.2)},
        "stepsize_controller": d.StepTo(jnp.linspace(0, 1, 17)),
        "max_steps": 1,
    }
    result = d.diffeqsolve(**arguments, throw=False)
    assert result.result == d.RESULTS.max_steps_reached
    with pytest.raises(Exception, match="maximum number of solver steps"):
        d.diffeqsolve(**arguments, throw=True)


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nonlinear_hierarchical_compiler_rhs_matches_independent_cells(
    dtype: str, *, coupled: bool
) -> None:
    """Current aliases, shaped rates and global reductions match scalar balances."""
    problem = HierarchicalProblem(groups=2, local_dimension=2, coupled=coupled)
    term, pack = compiled_hierarchical_term(problem)
    values = jnp.asarray(nominal_parameters(problem), dtype=getattr(jnp, dtype))
    initial = initial_state(problem, values)
    result = jax.jit(term.vf)(0.0, {"A": initial[0], "B": initial[1]}, pack(values))
    expected = reference_rhs(problem, np.asarray(values), np.asarray(initial))
    tolerance = 2e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        jnp.stack((result["A"], result["B"])), expected, rtol=tolerance, atol=tolerance
    )
