# Biogeochemical network tutorial

The repository's
[`biogeochemical_network.py`](https://github.com/ACCIDDA/op_engine/blob/main/examples/biogeochemical_network.py)
example compares explicit and IMEX methods on a size-structured
phytoplankton-zooplankton model. It is both a numerical diagnostic and an
example of stage-dependent operator construction.

This example deliberately uses NumPy, SciPy, and Matplotlib so it can compare
against `scipy.integrate.solve_ivp` and produce figures. The core splitting
pattern is portable, but the complete example is not intended as a JAX program.

## Run it

From a development checkout:

```bash
uv sync --dev
uv run python examples/biogeochemical_network.py
```

Outputs are written to `examples/output/biogeochemical/`:

- bin trajectories comparing explicit Heun with SciPy RK45;
- bin trajectories comparing IMEX Euler and IMEX TR-BDF2 with SciPy BDF;
- a runtime-versus-output-spacing figure;
- CSV, text, and JSON metadata describing the run.

The default run covers twenty simulated years and includes a runtime sweep. For
quick experimentation, reduce `total_time_days`, `sweep_total_days`, or
`n_bins` in `main()`.

## State and reaction term

For `n` size bins, the flattened state contains `n` phytoplankton values
followed by `n` zooplankton values. `ModelCore` stores this as `(2n, 1)`, where
the first dimension is the operator axis and the singleton dimension is the
subgroup axis.

The reaction callable contains the terms assigned to the explicit partition.
For an IMEX run, `run_op_engine()` separately constructs the selected matrix
`A(t, y_stage)` and passes only the remainder as the explicit RHS. This avoids
counting a process in both partitions.

## Compare the three splits

The example defines three progressively richer implicit partitions:

| Split | Terms placed in `A(t, y)` | Diagnostic purpose |
| --- | --- | --- |
| A | Linear phytoplankton loss and zooplankton mortality | Minimal, mostly time-dependent implicit part |
| B | Split A plus the state-dependent phytoplankton grazing sink | Move the dominant dissipative diagonal term implicit |
| C | Split B plus scaled phytoplankton/zooplankton cross-coupling | Test a larger block operator without making every nonlinear term implicit |

The comparison is useful because an IMEX result depends on both the integrator
and the partition. Compare methods under the same split before attributing a
difference to the time integrator. The example uses split B for its canonical
figures and runs all three splits in the timing diagnostics.

`ETA_CROSS` controls the cross-coupling admitted to split C. It is a model-split
choice, not a solver tolerance.

## Choose when operators are rebuilt

`OPERATOR_MODE` controls the information used by
`make_base_builder_for_split()`:

- `"frozen"` builds `A` once from the initial state and time;
- `"time"` rebuilds it at each stage time while retaining the initial state;
- `"stage_state"` rebuilds it from both `StageOperatorContext.t` and
  `StageOperatorContext.y`.

The default, `"stage_state"`, demonstrates the general nonlinear-splitting
contract. The builder is wrapped by `make_stage_operator_factory()`:

```python
base_builder = make_base_builder_for_split(
    split,
    model=model,
    y0_flat=run.y0_flat,
)

if method == "imex-heun-tr":
    operators = OperatorSpecs(
        default=make_stage_operator_factory(
            base_builder,
            scheme="trapezoidal",
        ),
    )
elif method == "imex-trbdf2":
    operators = OperatorSpecs(
        tr=make_stage_operator_factory(base_builder, scheme="trapezoidal"),
        bdf2=make_stage_operator_factory(base_builder, scheme="implicit-euler"),
    )
```

This factory boundary is important. Adaptive attempts and TR-BDF2 stages use
different effective step sizes, so precomputed `(L, R)` matrices would be wrong
unless the step size were fixed and uniform.

## Read the diagnostics

Use the outputs as checks, not as a universal benchmark:

1. Compare Heun against RK45 to check the unsplit reaction implementation.
2. Compare IMEX methods against BDF on the same output grid.
3. Compare splits A, B, and C at the same method and tolerances.
4. Inspect convergence as output spacing decreases before comparing runtimes.
5. Check the manifest for operator mode, tolerances, split settings, and output
   paths before interpreting a figure.

The example clips negative stage values when `CLIP_NONNEGATIVE` is enabled.
That stabilization choice can affect the numerical comparison and should be
reported alongside the solver and split.

For small, copyable configurations, start with the
[solver-method guide](solver-methods.md). Use this example when you need to see
how stage time, stage state, and alternative partitions fit together in a full
workflow.
