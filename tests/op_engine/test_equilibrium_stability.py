"""Independent local-equilibrium stability references for issue #256."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never

import numpy as np
import pytest

from op_engine import (
    EquilibriumDomain,
    LocalSpectrum,
    LocalStabilityConfig,
    LocalStabilityStatus,
    StabilityJacobianSource,
    StabilityTimescaleKind,
    SteadyStateResult,
    local_stability,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _config(
    *,
    residual: float = 1e-9,
    spectral_absolute: float = 1e-10,
    spectral_relative: float = 1e-8,
    boundary: float = 1e-10,
) -> LocalStabilityConfig:
    return LocalStabilityConfig(
        residual_tolerance=residual,
        spectral_absolute_tolerance=spectral_absolute,
        spectral_relative_tolerance=spectral_relative,
        boundary_tolerance=boundary,
    )


def _linear(
    matrix: NDArray[np.float64],
) -> tuple[Any, Any]:
    def rhs(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return matrix @ state

    def jacobian(_time: float, _state: NDArray[np.float64]) -> NDArray[np.float64]:
        return matrix

    return rhs, jacobian


def test_stable_focus_reports_strict_decay_timescale() -> None:
    """The known pair ``-1 +/- 2i`` has an e-folding decay time of one."""
    matrix = np.array([[-1.0, -2.0], [2.0, -1.0]])
    rhs, jacobian = _linear(matrix)
    result = local_stability(
        rhs,
        np.zeros(2),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="analytic stable focus",
        coordinates=("x", "y"),
        time_unit="day",
    )
    assert result.status is LocalStabilityStatus.ANALYZED
    assert result.classification is LocalSpectrum.ASYMPTOTICALLY_STABLE
    assert result.timescale_kind is StabilityTimescaleKind.DECAY
    assert result.leading_timescale == pytest.approx(1.0)
    assert result.spectral_abscissa == pytest.approx(-1.0)
    np.testing.assert_allclose(
        np.sort_complex(result.tangent_eigenvalues), np.array([-1 - 2j, -1 + 2j])
    )


def test_unstable_saddle_reports_growth_timescale() -> None:
    """The positive eigenvalue two implies an e-folding growth time of one half."""
    matrix = np.diag(np.array([2.0, -1.0]))
    rhs, jacobian = _linear(matrix)
    result = local_stability(
        rhs,
        np.zeros(2),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="analytic saddle",
    )
    assert result.classification is LocalSpectrum.UNSTABLE
    assert result.timescale_kind is StabilityTimescaleKind.GROWTH
    assert result.leading_timescale == pytest.approx(0.5)
    assert result.leading_eigenvalue == pytest.approx(2 + 0j)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.array([[0.0, -1.0], [1.0, 0.0]]), id="rotation"),
        pytest.param(np.array([[0.0, 1.0], [0.0, 0.0]]), id="jordan"),
        pytest.param(np.zeros((1, 1)), id="zero"),
    ],
)
def test_neutral_spectra_are_not_stability_certificates(
    matrix: NDArray[np.float64],
) -> None:
    """Imaginary and defective zero spectra only describe eigenvalue locations."""
    rhs, jacobian = _linear(matrix)
    result = local_stability(
        rhs,
        np.zeros(matrix.shape[0]),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="analytic critical spectrum",
    )
    assert result.status is LocalStabilityStatus.ANALYZED
    assert result.classification is LocalSpectrum.NEUTRAL_SPECTRUM
    assert result.leading_timescale is None


def test_zero_linearization_does_not_certify_nonlinear_attraction() -> None:
    """The attracting cubic root has a neutral linear spectrum and no timescale."""
    equilibrium = 1.0

    def rhs(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return -((state - equilibrium) ** 3)

    result = local_stability(
        rhs,
        np.array([equilibrium]),
        config=_config(),
        jacobian=lambda _t, state: np.zeros((state.size, state.size)),
        jacobian_reference="analytic derivative at the cubic root",
    )
    assert result.classification is LocalSpectrum.NEUTRAL_SPECTRUM
    assert result.leading_timescale is None


def test_mixed_decaying_and_critical_modes_are_nonhyperbolic() -> None:
    """A zero mode mixed with decay is distinct from a wholly neutral spectrum."""
    matrix = np.diag(np.array([-1.0, 0.0]))
    rhs, jacobian = _linear(matrix)
    result = local_stability(
        rhs,
        np.zeros(2),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="analytic mixed spectrum",
    )
    assert result.classification is LocalSpectrum.NONHYPERBOLIC


def test_conservation_mode_is_retained_then_explicitly_restricted() -> None:
    """Exchange has a full zero mode and decay rate ``-2k`` at fixed total."""
    rate = 0.4
    matrix = np.array([[-rate, rate], [rate, -rate]])
    rhs, jacobian = _linear(matrix)
    unrestricted = local_stability(
        rhs,
        np.array([3.0, 3.0]),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="two-state exchange",
        coordinates=("x", "y"),
    )
    assert unrestricted.classification is LocalSpectrum.NONHYPERBOLIC

    restricted = local_stability(
        rhs,
        np.array([3.0, 3.0]),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="two-state exchange",
        coordinates=("x", "y"),
        invariants=np.array([[1.0, 1.0]]),
        time_unit="day",
    )
    assert restricted.status is LocalStabilityStatus.ANALYZED
    assert restricted.classification is LocalSpectrum.ASYMPTOTICALLY_STABLE
    assert restricted.expected_conservation_neutral_modes == 1
    assert restricted.observed_active_neutral_modes == 1
    assert restricted.leading_timescale == pytest.approx(1 / (2 * rate))
    np.testing.assert_allclose(restricted.tangent_eigenvalues, [-2 * rate])
    projector = restricted.tangent_basis @ restricted.tangent_basis.T
    np.testing.assert_allclose(projector, [[0.5, -0.5], [-0.5, 0.5]], atol=1e-14)


def test_fixed_sink_residual_is_reported_but_excluded() -> None:
    """A cumulative counter may have nonzero flow while the active root is stable."""
    state = np.zeros(2)

    def rhs(_time: float, value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([-value[0], 1.0])

    result = local_stability(
        rhs,
        state,
        config=_config(),
        jacobian=lambda _t, _y: np.array([[-1.0, 0.0], [0.0, 0.0]]),
        jacobian_reference="analytic decay plus cumulative counter",
        fixed=[1],
        coordinates=("population", "cumulative"),
    )
    assert result.status is LocalStabilityStatus.ANALYZED
    assert result.classification is LocalSpectrum.ASYMPTOTICALLY_STABLE
    np.testing.assert_array_equal(result.raw_residual, [0.0, 1.0])
    np.testing.assert_array_equal(result.active_residual, [0.0])


def test_boundary_suppresses_interior_classification_but_retains_spectrum() -> None:
    """A smooth extension at zero is descriptive until a tangent cone is analyzed."""
    rhs, jacobian = _linear(np.array([[-1.0]]))
    result = local_stability(
        rhs,
        np.zeros(1),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="smooth extension",
        nonnegative=True,
        coordinates=("population",),
    )
    assert result.status is LocalStabilityStatus.BOUNDARY
    assert result.domain is EquilibriumDomain.BOUNDARY
    assert result.classification is None
    assert result.boundary_coordinates == ("population",)
    np.testing.assert_allclose(result.tangent_eigenvalues, [-1.0])


def test_boundary_refuses_central_difference_across_domain() -> None:
    """The implicit finite-difference route never probes a negative population."""
    result = local_stability(
        lambda _t, value: -value,
        np.zeros(1),
        config=_config(),
        nonnegative=True,
    )
    assert result.status is LocalStabilityStatus.BOUNDARY
    assert result.jacobian_source is StabilityJacobianSource.FINITE_DIFFERENCE
    assert result.jacobian is None


def test_infeasible_and_off_root_candidates_fail_before_spectrum() -> None:
    """Domain and equilibrium gates remain distinct explicit outcomes."""
    rhs, jacobian = _linear(np.array([[-1.0]]))
    infeasible = local_stability(
        rhs,
        np.array([-0.2]),
        config=_config(residual=1.0, boundary=0.01),
        jacobian=jacobian,
        jacobian_reference="analytic",
        nonnegative=True,
    )
    assert infeasible.status is LocalStabilityStatus.INFEASIBLE
    assert infeasible.domain is EquilibriumDomain.INFEASIBLE

    off_root = local_stability(
        rhs,
        np.array([1.0]),
        config=_config(residual=0.1),
        jacobian=jacobian,
        jacobian_reference="analytic",
    )
    assert off_root.status is LocalStabilityStatus.NOT_EQUILIBRIUM
    assert off_root.jacobian is None


def test_infeasible_state_never_calls_rhs_outside_declared_domain() -> None:
    """The domain gate precedes any scientific callback, including fixed states."""
    calls = 0

    def guarded_rhs(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        if np.any(state < 0):
            message = "outside model domain"
            raise ValueError(message)
        return -state

    result = local_stability(
        guarded_rhs,
        np.array([-0.2]),
        config=_config(boundary=0.01),
        fixed=[0],
        nonnegative=True,
    )
    assert result.status is LocalStabilityStatus.INFEASIBLE
    assert result.domain is EquilibriumDomain.INFEASIBLE
    assert result.raw_residual is None
    assert calls == 0


def test_unconverged_solver_result_is_not_relabelled() -> None:
    """A looser local residual threshold cannot erase solver failure evidence."""
    state = np.zeros(1)
    candidate = SteadyStateResult(
        state=state,
        residual=state,
        converged=np.zeros((), dtype=np.bool_),
        iterations=np.asarray(1),
        residual_norm=np.asarray(0.0),
        step_norm=np.asarray(0.0),
        dt=np.asarray(1.0),
        invariant_drift=np.asarray(0.0),
    )
    result = local_stability(lambda _t, value: value, candidate, config=_config())
    assert result.status is LocalStabilityStatus.SOLVE_FAILURE
    np.testing.assert_array_equal(result.raw_residual, [0.0])


@pytest.mark.parametrize(
    ("state", "rhs", "status"),
    [
        pytest.param(
            np.array([np.nan]),
            lambda _t, value: value,
            LocalStabilityStatus.NONFINITE_STATE,
            id="state",
        ),
        pytest.param(
            np.zeros(1),
            lambda _t, value: np.full_like(value, np.nan),
            LocalStabilityStatus.NONFINITE_RESIDUAL,
            id="residual",
        ),
    ],
)
def test_nonfinite_evidence_has_explicit_status(
    state: NDArray[np.float64], rhs: object, status: LocalStabilityStatus
) -> None:
    """Non-finite inputs and residuals do not reach an eigensolver."""
    result = local_stability(rhs, state, config=_config())
    assert result.status is status
    assert result.classification is None


def test_callback_failures_and_nonfinite_jacobian_remain_distinct() -> None:
    """Callback exceptions and successful non-finite outputs retain their boundary."""
    state = np.zeros(1)

    def broken(_time: float, _state: NDArray[np.float64]) -> NDArray[np.float64]:
        msg = "broken analytic callback"
        raise RuntimeError(msg)

    failed = local_stability(
        lambda _t, value: value,
        state,
        config=_config(),
        jacobian=broken,
        jacobian_reference="broken fixture",
    )
    assert failed.status is LocalStabilityStatus.JACOBIAN_FAILURE
    assert "broken analytic callback" in failed.message

    nonfinite = local_stability(
        lambda _t, value: value,
        state,
        config=_config(),
        jacobian=lambda _t, _y: np.array([[np.inf]]),
        jacobian_reference="non-finite fixture",
    )
    assert nonfinite.status is LocalStabilityStatus.NONFINITE_JACOBIAN


def test_inconsistent_and_redundant_invariants_fail_visibly() -> None:
    """Only declared, locally valid independent constraints remove modes."""
    matrix = np.diag(np.array([-1.0, -2.0]))
    rhs, jacobian = _linear(matrix)
    inconsistent = local_stability(
        rhs,
        np.zeros(2),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="diagonal",
        invariants=np.array([[1.0, 1.0]]),
    )
    assert inconsistent.status is LocalStabilityStatus.INCONSISTENT_INVARIANTS
    assert inconsistent.conservation_error > inconsistent.conservation_tolerance

    redundant = local_stability(
        rhs,
        np.zeros(2),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="diagonal",
        fixed=[0],
        invariants=np.array([[1.0, 0.0]]),
    )
    assert redundant.status is LocalStabilityStatus.INCONSISTENT_INVARIANTS


def test_all_fixed_coordinates_report_no_dynamic_directions() -> None:
    """An empty tangent space is not vacuously called stable."""
    rhs, jacobian = _linear(np.diag(np.array([-1.0, -2.0])))
    result = local_stability(
        rhs,
        np.zeros(2),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="diagonal",
        fixed=np.ones(2, dtype=np.bool_),
    )
    assert result.status is LocalStabilityStatus.NO_DYNAMIC_DIRECTIONS
    assert result.classification is None


def test_central_difference_matches_independent_nonlinear_jacobian() -> None:
    """Recorded central differences recover a hand-derived nonlinear Jacobian."""
    state = np.array([1.0, 0.0])

    def rhs(_time: float, value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([value[0] ** 2 - 1, -2 * value[1]])

    result = local_stability(rhs, state, config=_config(residual=1e-8))
    assert result.status is LocalStabilityStatus.ANALYZED
    assert result.jacobian_source is StabilityJacobianSource.FINITE_DIFFERENCE
    assert result.finite_difference_steps is not None
    np.testing.assert_allclose(result.jacobian, [[2.0, 0.0], [0.0, -2.0]], rtol=1e-9)


def test_reused_rhs_buffer_cannot_mutate_evidence_or_difference_columns() -> None:
    """Owned callback results prevent aliases from changing the classification."""
    matrix = np.diag(np.array([-1.0, -2.0]))
    buffer = np.empty(2)

    def rhs(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        np.matmul(matrix, state, out=buffer)
        return buffer

    result = local_stability(rhs, np.zeros(2), config=_config())
    assert result.status is LocalStabilityStatus.ANALYZED
    assert result.classification is LocalSpectrum.ASYMPTOTICALLY_STABLE
    np.testing.assert_array_equal(result.raw_residual, np.zeros(2))
    np.testing.assert_allclose(result.jacobian, matrix, rtol=1e-10, atol=1e-12)


def test_spectral_relative_band_is_similarity_invariant_for_known_pair() -> None:
    """Similar triangular matrices retain one spectral tolerance and conclusion."""
    diagonal = np.diag(np.array([-1.0, -2.0]))
    rescaled = np.array([[-1.0, -1e10], [0.0, -2.0]])
    outcomes = []
    for matrix in (diagonal, rescaled):
        rhs, jacobian = _linear(matrix)
        outcomes.append(
            local_stability(
                rhs,
                np.zeros(2),
                config=_config(),
                jacobian=jacobian,
                jacobian_reference="analytic similar matrix",
            )
        )
    for result in outcomes:
        assert result.classification is LocalSpectrum.ASYMPTOTICALLY_STABLE
        assert result.effective_spectral_tolerance == pytest.approx(2.01e-8)
        assert result.observed_active_neutral_modes == 0


def test_threshold_equality_is_neutral_spectrum() -> None:
    """A real part exactly on the user band is never called growing or decaying."""
    matrix = np.array([[1.0]])
    rhs, jacobian = _linear(matrix)
    result = local_stability(
        rhs,
        np.zeros(1),
        config=_config(spectral_absolute=1.0, spectral_relative=0.0),
        jacobian=jacobian,
        jacobian_reference="threshold equality",
    )
    assert result.classification is LocalSpectrum.NEUTRAL_SPECTRUM


@dataclass(frozen=True)
class _Identity:
    digest: str = "sha256:" + "a" * 64


@dataclass(frozen=True)
class _Values:
    state: object


@dataclass(frozen=True)
class _Artifact:
    matrix: NDArray[np.float64]
    state_names: tuple[str, ...] = ("x", "y")
    model: object = _Identity()

    def evaluate(self, _t: object, _y: object, **_parameters: object) -> _Values:
        """Return the independently supplied symbolic matrix."""
        return _Values(self.matrix)


def test_symbolic_artifact_preserves_coordinates_and_identity() -> None:
    """The structural derivative route records order and source without fallback."""
    matrix = np.diag(np.array([-1.0, -2.0]))
    rhs, _ = _linear(matrix)
    artifact = _Artifact(matrix)
    result = local_stability(
        rhs,
        np.zeros(2),
        config=_config(),
        derivatives=artifact,
        coordinate_system="natural",
    )
    assert result.status is LocalStabilityStatus.ANALYZED
    assert result.jacobian_source is StabilityJacobianSource.OP_SYSTEM_SYMBOLIC
    assert result.coordinates == artifact.state_names
    assert result.model_digest == artifact.model.digest


def test_symbolic_artifact_failure_does_not_fall_back() -> None:
    """A failed declared structural route remains a Jacobian failure."""

    class BrokenArtifact:
        state_names = ("x",)
        model = _Identity()

        def evaluate(self, _t: object, _y: object, **_parameters: object) -> _Values:
            message = "unsupported structural derivative"
            raise NotImplementedError(message)

    result = local_stability(
        lambda _t, value: value,
        np.zeros(1),
        config=_config(),
        derivatives=BrokenArtifact(),
    )
    assert result.status is LocalStabilityStatus.JACOBIAN_FAILURE
    assert result.jacobian_source is StabilityJacobianSource.OP_SYSTEM_SYMBOLIC
    assert result.jacobian is None
    assert "unsupported structural derivative" in result.message


def test_numpy_op_system_symbolic_and_finite_difference_contract() -> None:
    """The pinned compiler artifact and an independent difference route agree."""
    op_system = pytest.importorskip("op_system")
    compiled = op_system.compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"net": "k*(x-z)"},
        "equations": {"x": "-net", "z": "net"},
    })
    artifact = op_system.symbolic_derivatives(compiled)
    rate = np.array(0.4)
    parameters = {"k": rate}
    state = np.array([2.0, 2.0])
    expected = np.array([[-rate, rate], [rate, -rate]])

    def rhs(time: float, value: NDArray[np.float64]) -> NDArray[np.float64]:
        return compiled.eval_fn(time, value, **parameters)

    symbolic = local_stability(
        rhs,
        state,
        config=_config(),
        derivatives=artifact,
        parameters=parameters,
        invariants=np.array([[1.0, 1.0]]),
    )
    numerical = local_stability(
        rhs,
        state,
        config=_config(),
        invariants=np.array([[1.0, 1.0]]),
    )
    assert symbolic.status is LocalStabilityStatus.ANALYZED
    assert numerical.status is LocalStabilityStatus.ANALYZED
    assert symbolic.coordinates == artifact.state_names
    assert symbolic.model_digest == artifact.model.digest
    np.testing.assert_allclose(symbolic.jacobian, expected, rtol=0, atol=0)
    np.testing.assert_allclose(numerical.jacobian, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        symbolic.tangent_eigenvalues, numerical.tangent_eigenvalues, rtol=1e-10
    )

    unsupported = op_system.compile_spec({
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "np.abs(x)"},
    })
    with pytest.raises(NotImplementedError, match="Unsupported"):
        op_system.symbolic_derivatives(unsupported)


def test_eigensolver_failure_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LAPACK failure returns partial Jacobian evidence without a classification."""
    matrix = np.array([[-1.0]])
    rhs, jacobian = _linear(matrix)

    def fail(_matrix: object) -> Never:
        msg = "synthetic eigensolver failure"
        raise np.linalg.LinAlgError(msg)

    monkeypatch.setattr(np.linalg, "eigvals", fail)
    result = local_stability(
        rhs,
        np.zeros(1),
        config=_config(),
        jacobian=jacobian,
        jacobian_reference="analytic",
    )
    assert result.status is LocalStabilityStatus.EIGENSOLVE_FAILED
    assert result.jacobian is not None
    assert result.classification is None


def test_non_numpy_state_and_malformed_declarations_are_rejected() -> None:
    """The host-only API never performs a hidden device or container conversion."""
    with pytest.raises(TypeError, match="eager NumPy"):
        local_stability(lambda _t, value: value, [0.0], config=_config())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distinct"):
        local_stability(
            lambda _t, value: value,
            np.zeros(2),
            config=_config(),
            coordinates=("x", "x"),
        )
    with pytest.raises(ValueError, match="jacobian_reference"):
        local_stability(
            lambda _t, value: value,
            np.zeros(1),
            config=_config(),
            jacobian=lambda _t, _y: np.zeros((1, 1)),
        )


def test_jax_state_is_rejected_before_any_hidden_host_transfer() -> None:
    """The eager diagnostic requires an explicit device-to-host caller boundary."""
    jnp = pytest.importorskip("jax.numpy")
    state = jnp.zeros(1)
    with pytest.raises(TypeError, match="eager NumPy state"):
        local_stability(lambda _t, value: value, state, config=_config())
    candidate = SteadyStateResult(
        state=state,
        residual=state,
        converged=jnp.ones((), dtype=jnp.bool_),
        iterations=jnp.asarray(1),
        residual_norm=jnp.asarray(0.0),
        step_norm=jnp.asarray(0.0),
        dt=jnp.asarray(1.0),
        invariant_drift=jnp.asarray(0.0),
    )
    with pytest.raises(TypeError, match="eager NumPy state"):
        local_stability(lambda _t, value: value, candidate, config=_config())
