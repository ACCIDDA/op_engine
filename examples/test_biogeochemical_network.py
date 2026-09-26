"""Regression tests for the biogeochemical example's IMEX partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_engine.matrix_ops import StageOperatorContext

from .biogeochemical_network import (
    build_model,
    build_run_spec,
    flat_from_tensor,
    make_base_builder_for_split,
    make_explicit_remainder_tensor,
    tensor_from_flat,
)

if TYPE_CHECKING:
    from .biogeochemical_network import SplitName


@pytest.mark.parametrize("split", ["A", "B", "C"])
def test_imex_partition_reconstructs_full_rhs(split: SplitName) -> None:
    """The explicit remainder plus ``A(t, y)y`` equals the full model RHS."""
    model = build_model(n_bins=3)
    run = build_run_spec(
        model=model,
        seed=123,
        total_time_days=2.0,
        dt_out_main_days=1.0,
    )
    state = tensor_from_flat(run.y0_flat)
    time = 0.75
    base_builder = make_base_builder_for_split(
        split,
        model=model,
        y0_flat=run.y0_flat,
    )
    remainder = make_explicit_remainder_tensor(run.rhs_tensor, base_builder)

    full = flat_from_tensor(run.rhs_tensor(time, state))
    explicit = flat_from_tensor(remainder(time, state))
    operator = base_builder(
        StageOperatorContext(t=time, y=state, stage="test"),
    )
    reconstructed = explicit + np.asarray(operator @ run.y0_flat)

    np.testing.assert_allclose(reconstructed, full, rtol=1e-13, atol=1e-13)
