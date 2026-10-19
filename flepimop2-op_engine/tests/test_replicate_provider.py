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

"""Installed real-provider contracts for independent banks and count sidecars."""

from __future__ import annotations

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum
from op_engine import FiringRecorder

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine


def test_replicate_provider_preserves_flat_order_forcing_and_accepted_counts() -> None:
    """Common forcing activates separate clocks; counts reconstruct local states."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "factorize_axes": ["rep"],
            "axes": [
                {"name": "rep", "coords": ["a", "b"]},
                {"name": "age", "coords": ["young", "old"]},
                {"name": "time", "type": "continuous", "coords": [0.0, 1.0]},
            ],
            "time_interpolation": "previous",
            "state": ["S[rep,age]", "I[rep,age]"],
            "initial_state": {"S[rep,age]": 20.0, "I[rep,age]": 0.0},
            "transitions": [
                {
                    "name": "infect",
                    "from": "S[rep,age]",
                    "to": "I[rep,age]",
                    "rate": "beta[time,rep]",
                },
                {
                    "name": "recover",
                    "from": "I[rep,age]",
                    "to": "S[rep,age]",
                    "rate": "gamma",
                },
            ],
        }
    )
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            mode="stochastic",
            stochastic_method="replicate-direct-ssa",
            replicate_axis="rep",
            random_seed=83,
        ),
    )
    times = np.array([0.0, 0.5, 1.0, 2.0])
    params = {
        "beta": ParameterValue(
            np.array([[0.0, 0.3], [0.3, 0.0]]),
            ResolvedShape(axis_names=("time", "rep"), sizes=(2, 2)),
        ),
        "gamma": ParameterValue(np.asarray(0.1), ResolvedShape()),
    }
    recorder = FiringRecorder()
    result = np.asarray(
        engine.run(
            system, times, {}, params, firing_recorder=recorder, replicate_ids=[8, 2]
        )
    )
    unrecorded = np.asarray(engine.run(system, times, {}, params, replicate_ids=[8, 2]))
    np.testing.assert_array_equal(result, unrecorded)
    np.testing.assert_array_equal(result[:, 0], times)
    states = result[:, 1:].reshape((len(times), 2, 2, 2))
    np.testing.assert_array_equal(states.sum(axis=1), 20.0)
    cumulative = recorder.cumulative
    expected_infected = (cumulative[:, :2] - cumulative[:, 2:]).transpose(0, 2, 1)
    np.testing.assert_array_equal(states[:, 1], expected_infected)
    assert not states[:3, 1, 0].any()
    assert cumulative[-1, :2, 0].sum() > 0


def test_plain_direct_provider_accepts_recorder_without_changing_result() -> None:
    """A provider count sidecar leaves the ordinary flat trajectory API intact."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "state": ["X"],
            "initial_state": {"X": 0.0},
            "transitions": [{"name": "birth", "to": "X", "rate": "rate"}],
        }
    )
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            mode="stochastic",
            stochastic_method="direct-ssa",
            random_seed=8,
        ),
    )
    recorder = FiringRecorder()
    result = np.asarray(
        engine.run(
            system,
            np.arange(4.0),
            {},
            {"rate": ParameterValue(np.asarray(4.0), ResolvedShape())},
            firing_recorder=recorder,
        )
    )
    np.testing.assert_array_equal(recorder.cumulative[:, 0, 0], result[:, 1])


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "stochastic", "stochastic_method": "replicate-direct-ssa"},
        {
            "mode": "deterministic",
            "stochastic_method": "replicate-direct-ssa",
            "replicate_axis": "rep",
        },
        {"replicate_axis": "rep"},
    ],
)
def test_invalid_replicate_configuration(settings: dict[str, str]) -> None:
    """Replicate-only options cannot silently select another execution path."""
    with pytest.raises(ValueError, match="replicate"):
        OpEngineEngineConfig(**settings)
