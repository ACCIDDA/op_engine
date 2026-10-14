"""Call-local NumPy random streams shared by the convenience samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

RandomStream: TypeAlias = np.random.Generator | np.random.SeedSequence
"""An advancing NumPy generator or a replayable seed-sequence initialization."""

_SamplerT = TypeVar("_SamplerT")


def _generator(
    rng: RandomStream | None, *, seed: int | None = None
) -> np.random.Generator:
    """Resolve a stream without copying an advancing generator.

    Returns:
        The supplied generator, or a fresh generator initialized from a seed.

    Raises:
        ValueError: If a stream and integer seed are supplied together.
        TypeError: If an explicit stream has an unsupported type.
    """
    if rng is not None and seed is not None:
        msg = "rng and seed are mutually exclusive"
        raise ValueError(msg)
    if rng is None:
        return np.random.default_rng(seed)
    if not isinstance(rng, (np.random.Generator, np.random.SeedSequence)):
        msg = "rng must be a numpy.random.Generator or SeedSequence"
        raise TypeError(msg)
    return np.random.default_rng(rng)


def _select_sampler(
    sampler: _SamplerT | None,
    rng: RandomStream | None,
    *,
    state: object,
    make_sampler: Callable[[np.random.Generator], _SamplerT],
) -> _SamplerT:
    """Resolve a run-local convenience sampler or preserve explicit injection.

    Returns:
        The injected sampler, or a new sampler wrapping the explicit generator.

    Raises:
        ValueError: If no random source is supplied or sources conflict.
    """
    if rng is not None:
        if sampler is not None:
            msg = "rng and explicit samplers are mutually exclusive"
            raise ValueError(msg)
        _require_numpy_state(state)
        return make_sampler(_generator(rng))
    if sampler is None:
        msg = "provide a sampler or rng"
        raise ValueError(msg)
    return sampler


def _require_numpy_state(state: object) -> None:
    """Validate the NumPy convenience path before any random draw or state update.

    Raises:
        TypeError: If a native state needs a backend-specific injected sampler.
    """
    if not isinstance(state, np.ndarray):
        msg = "rng convenience requires NumPy state; inject a native sampler otherwise"
        raise TypeError(msg)
