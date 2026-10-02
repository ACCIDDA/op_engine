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
from op_engine import ThinningSample

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine

if TYPE_CHECKING:
    from flepimop2.typing import Array


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
def test_provider_smooth_births_resume_reject_and_preserve_pending_draws(
    backend: str,
) -> None:
    """Candidate-time rates wake an initially dormant producer on both backends."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = _birth_system()
    engine = _engine(thinning_rate_bound=2)
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
    )
    np.testing.assert_array_equal(np.asarray(result)[:, 1], [0, 0, 0, 1, 2, 2])
    assert result.__array_namespace__() is xp
    assert sampler.indices == [0, 1, 2, 3]
    assert sampler.rates == [2, 2, 2, 2]
    assert engine.validate_system(system) is None
