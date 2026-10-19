# Checks at requested output times

`RunConfig(state_checks=OutputStateCheckConfig(...))` records nonnegativity,
finiteness and optional monotone-counter checks during a deterministic solve
or frozen-mesh replay. The checks leave the numerical solution unchanged and
are disabled by default. Read the record from `solver.last_state_diagnostics`;
the existing return value of `run` and replay remains unchanged, including
nonlinear diagnostics for SDIRK2.

```python
--8<-- "examples/output_state_checks.py"
```

The example deliberately uses a coarse Euler step. The member with rate 0.1
has output values `1, 0.9, 0.81`; the member with rate 2 has `1, -1, 1`.
The latter is flagged even though its final value is positive. In a separate
example, a cumulative counter has values `5, 4, 3`: every state is positive,
but its two negative increments are flagged.

## Scope and interpretation

These checks inspect the **requested output grid**, including the initial
state. They do not inspect internal accepted substeps, rejected attempts or
RK stages. For example, the two Euler updates `1 → -1 → 1` would escape a
nonnegativity flag if only the first and last values were requested outputs.
A successful output record therefore does not prove positivity throughout a
solve, numerical accuracy or physical validity. Use a suitable solver, step
size or state representation for the model.

`nonnegative_indices` selects components on the leading state axis; `None`
checks all components. Use a subset when signed states, such as velocities,
can legitimately be negative. Checks apply to the coordinates stored by the
solver. For log coordinates or other transformed states, decode the stored
trajectory before checking physical-state nonnegativity. `monotone_indices`
selects cumulative components
whose output increments should be nonnegative. A model's observation
transformation may need separate checks; the solver cannot infer which
derived observations should be monotone.

All remaining subgroup and field axes are retained in the diagnostic arrays.
A simulation bank can therefore keep a flag per member without collapsing
failures across the whole bank. Counts flag each output or interval once if
any selected component fails; they do not count failing components.

| Field | Meaning |
| --- | --- |
| `n_outputs` | Number of inspected output records, including the initial state |
| `complete` | Every output on the solve's grid was inspected; false for a partial failed run |
| `min_state` | Minimum selected state over those outputs; NaNs propagate |
| `n_negative_outputs` | Output records with any selected state below the negative tolerance |
| `n_nonfinite_outputs` | Output records with any nonfinite component, including unselected components |
| `min_increment` | Minimum selected cumulative increment; `None` when disabled, positive infinity when no interval exists |
| `n_decreasing_intervals` | Output intervals with a cumulative decrement; `None` when disabled |
| `valid` | Native flags for a complete, finite record with no threshold violations on the checked outputs |

`negative_tolerance` and `monotone_tolerance` are finite, nonnegative absolute
thresholds. Equality at minus the tolerance is accepted. They change the
flags, not the recorded minima or trajectory; no value is clamped or projected.
The increment tolerance is an amount per output interval, not a rate scaled
by its duration.

## Storage and backends

The eager driver collects running minima and counts, including when
`ModelCoreOptions(store_history=False)` discards output history. Whole-grid
scan drivers reduce diagnostics over their output trajectory with native array
operations, rather than unrolling a diagnostic loop. NumPy, JAX and CPU
PyTorch have independent count checks; JAX JIT/vmap and state gradients are
covered. Flags stay in the state's array namespace; the checks do not extract
host scalars inside a compiled solve.

`diagnose_trajectory(trajectory, config)` performs the same output checks on
an existing real floating array shaped `(output, state, *other_axes)`, including
stored stochastic paths or provider outputs. It checks the supplied records;
the standalone function cannot determine whether a larger solve completed.
The core option above applies to `CoreSolver`; provider prepared runners and
stochastic solver configurations have their own interfaces.

Retain these records with the simulation results when choosing how to handle
flagged draws. Rejecting parameter draws changes the simulated distribution;
use a stated acceptance rule consistently for fitting and calibration.
