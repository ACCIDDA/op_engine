# Accepted meshes: compatibility and accuracy

A frozen accepted mesh is static controller output. Replay invokes its numerical
steps and bypasses adaptive error estimates and acceptance decisions. Its AD
results are derivatives **conditional on that mesh**. A compatible mesh may
still become inaccurate when model parameters or initial conditions move.

The core contract distinguishes these questions before testing accuracy:

| Metadata | Replay treatment |
| --- | --- |
| Finite/increasing output grid and positive interval steps summing to each interval | Always validated by `AdaptiveStepSchedule` |
| Producing resolved method, state shape/axis order, relevant operator axis, TR-BDF2 gamma, nonlinear stage solver/configuration | Known mismatches reject before stepping |
| Caller-supplied structural model identity | Compared when recording/replay use an identity; it must describe equations/geometry/layout, not fitted parameter values |
| Array namespace and dtype | Descriptive; portable meshes may change NumPy/JAX/Torch or precision |
| Discovery tolerances, controller limits, and reference parameter regime | Descriptive; changing fitted values or requested accuracy requires an external accuracy decision, not an automatic incompatibility |
| Callback bodies and arbitrary custom solver internals | No automatic semantic identity guarantee; use the structural identity and independently qualify changed implementations |

`AdaptiveScheduleProvenance` is attached to schedules created by live adaptive
runs. `RunConfig.replay_model_id` and `replay_reference_regime` are optional
static labels supplied by the caller. Record concrete nominal values or a
stable parameter-set identifier in the reference label at an eager boundary;
replay does not inspect or convert dynamic parameter arrays.

Manually constructed meshes and legacy schedules with `provenance=None` remain
accepted by default after mesh/output-grid validation. Their method/model
compatibility is unverified and remains the caller's responsibility. Use
`RunConfig(replay_require_provenance=True)` to reject unverified schedules.
Adding a provenance record is not an accuracy certificate, and copying labels
from another model does not make that model compatible.

Controller settings are recorded to explain discovery. Replay ignores error
control even if the supplied configuration requests tighter tolerances. A
provider may impose stronger configuration/context checks on its own artifact;
these core rules do not remove those provider checks.

A practical reuse policy is:

1. Discover a live mesh at a documented nominal parameter regime.
2. Reuse it while changing dynamic values, retaining structural model identity.
3. Compare a replay trajectory to a fresh accurate solve or to a refined mesh
   at representative new parameter values. Check the scientific objective and
   needed derivatives as well as state error.
4. If the independent discrepancy exceeds the intended accuracy, rediscover
   the mesh at the new regime, recheck agreement, and freeze that mesh for the
   next differentiation phase.

The structured example accompanying this guide executes that policy and shows
an accuracy failure despite compatible provenance. Discrete refresh decisions
remain outside the differentiated numerical solve.

Run the [structured refresh example](https://github.com/ACCIDDA/op_engine/blob/main/examples/replay_mesh_refresh.py):

```shell
uv run python examples/replay_mesh_refresh.py
```

It transfers mass between two generic pools with `A'=-k*A^2`, `B'=+k*A^2`,
a shared rate and group-local multipliers. The independent solution is
`A(t)=A0/(1+k*A0*t)`, with conserved `A+B`. A mesh discovered at rate 0.05
remains accurate nearby; rate 3.0 is structurally compatible but exceeds the
example's `1e-4` state-error policy on the old mesh. Fresh discovery restores
that accuracy. JSON reports both discrepancies and accepted-step counts.
This policy checks this fixture's state error; different scientific objectives
and derivative accuracy need their own independent checks.
