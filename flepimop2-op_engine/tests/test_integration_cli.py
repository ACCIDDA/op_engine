"""Integration test for the flepimop2 simulate pipeline using the op_engine provider."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest  # noqa: TC002
import yaml
from flepimop2.backend.abc import build as build_backend
from flepimop2.configuration import ConfigurationModel
from flepimop2.engine.abc import build as build_engine
from flepimop2.meta import RunMeta
from flepimop2.parameter.abc import build as build_parameter
from flepimop2.system.abc import build as build_system


def _write_sir_stepper(script_path: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        (
            '"""SIR model plugin for flepimop2 demo."""\n\n'
            "import numpy as np\n"
            "from numpy.typing import NDArray\n\n\n"
            "def stepper(\n"
            "    t: float,  # noqa: ARG001\n"
            "    y: NDArray[np.float64],\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            ") -> NDArray[np.float64]:\n"
            '    """dY/dt for the SIR model."""\n'
            "    y_s, y_i, _ = np.asarray(y, dtype=float)\n"
            "    infection = (beta * y_s * y_i) / np.sum(y)\n"
            "    recovery = gamma * y_i\n"
            "    dydt = [-infection, infection - recovery, recovery]\n"
            "    return np.array(dydt, dtype=float)\n"
        ),
        encoding="utf-8",
    )


def _write_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": "SIR_op_engine",
        "system": [{"module": "wrapper", "script": "model_input/plugins/SIR.py"}],
        "engine": [
            {
                "module": "op_engine",
                "config": {
                    "method": "heun",
                    "adaptive": False,
                    "strict": True,
                    "rtol": 1.0e-6,
                    "atol": 1.0e-9,
                },
            }
        ],
        "simulate": {
            "demo": {"times": [0, 10, 20]},
        },
        "backend": [{"module": "csv"}],
        "parameter": {
            "beta": {"module": "fixed", "value": 0.3},
            "gamma": {"module": "fixed", "value": 0.1},
            "s0": {"module": "fixed", "value": 999},
            "i0": {"module": "fixed", "value": 1},
            "r0": {"module": "fixed", "value": 0},
        },
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _select_simulate_target(config_model: ConfigurationModel, target: str) -> object:
    # Avoid private flepimop2 helpers. We pick a known target explicitly.
    simulate_block = config_model.simulate
    try:
        return simulate_block[target]
    except Exception as exc:  # pragma: no cover
        msg = f"simulate target {target!r} not found in configuration"
        raise KeyError(msg) from exc


def test_simulate_pipeline_writes_csv(  # noqa: PLR0914
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    End-to-end integration test for the simulate pipeline (in-process).

    Validates:
      - YAML config parsing
      - wrapper system loading from a script path
      - provider engine resolution: `engine: module: op_engine`
      - op_engine-backed execution
      - csv backend writes output
    """
    # Arrange: build an isolated project layout under tmp_path.
    monkeypatch.chdir(tmp_path)

    (tmp_path / "model_output").mkdir(parents=True, exist_ok=True)
    _write_sir_stepper(tmp_path / "model_input" / "plugins" / "SIR.py")
    _write_config(tmp_path / "configs" / "SIR_op_engine.yml")

    # Act: replicate the public simulate pipeline (no subprocess, no private imports).
    config_model = ConfigurationModel.from_yaml(Path("configs/SIR_op_engine.yml"))
    simulate_cfg = _select_simulate_target(config_model, target="demo")

    # The simulate config object in flepimop2 exposes these in SimulateCommand.run:
    # - system / engine / backend indices (names)
    # - t_eval (derived from times)
    system_cfg = config_model.systems[simulate_cfg.system].model_dump()
    engine_cfg = config_model.engines[simulate_cfg.engine].model_dump()
    backend_cfg = config_model.backends[simulate_cfg.backend].model_dump()

    s0 = build_parameter(config_model.parameters["s0"])
    i0 = build_parameter(config_model.parameters["i0"])
    r0 = build_parameter(config_model.parameters["r0"])
    y0 = np.array(
        [s0.sample().item(), i0.sample().item(), r0.sample().item()], dtype=np.float64
    )

    params = {
        k: build_parameter(v).sample().item()
        for k, v in config_model.parameters.items()
        if k not in {"s0", "i0", "r0"}
    }

    system = build_system(system_cfg)
    engine = build_engine(engine_cfg)
    backend = build_backend(backend_cfg)

    res = engine.run(system, simulate_cfg.t_eval, y0, params)
    backend.save(res, RunMeta(name="SIR_op_engine", action="simulate"))

    # Assert: CSV output exists and is sane.
    csv_files = sorted((tmp_path / "model_output").glob("*.csv"))
    assert csv_files, "expected csv backend to write at least one output file"

    arr = np.loadtxt(csv_files[-1], delimiter=",")
    assert arr.ndim == 2
    assert arr.shape[1] == 1 + 3  # time + 3-state SIR
    assert arr.shape[0] == len(simulate_cfg.t_eval)
