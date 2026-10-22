"""Explicit, bounded CPU Torch compilation for sequential solver loops.

Importing this module does not import Torch. Constructing an adapter opts into
native Inductor compilation; ordinary solver calls retain their eager defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, cast

from .loop_ops import LoopAdapter

if TYPE_CHECKING:
    from .loop_ops import LoopBody, ScanFunction


def torch_compile_loop_adapter(
    *, max_steps: int = 128, max_elements: int = 65_536
) -> LoopAdapter:
    """Construct an explicitly registered, bounded sequential CPU adapter.

    Supports Tensor/None trees in tuples, lists and dictionaries, homogeneous
    CPU float32/64, positive static input lengths and first-order autograd.
    The sequential loop is statically captured with fullgraph Inductor; graph
    breaks, disabled compilation and unsupported inputs raise. Checkpointing
    and GPU execution are not part of this adapter's qualified contract.

    Args:
        max_steps: Largest positive scan length accepted by this adapter.
        max_elements: Bound on input, carry and stacked output tensor elements.

    Returns:
        An unregistered adapter; register it for ``array_api_compat.torch``.

    Raises:
        ValueError: If bounds or Torch version are unsupported.
        ImportError: If the optional Torch dependency is absent.
    """  # noqa: DOC502 - optional import propagates ImportError
    if type(max_steps) is not int or not 1 <= max_steps <= 128:
        msg = "max_steps must be an integer in [1, 128]"
        raise ValueError(msg)
    if type(max_elements) is not int or not 1 <= max_elements <= 1_048_576:
        msg = "max_elements must be an integer in [1, 1048576]"
        raise ValueError(msg)
    torch = import_module("torch")
    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if version < (2, 14):
        msg = "Compiled loops require Torch >=2.14; eager Torch is unchanged"
        raise ValueError(msg)
    driver = _TorchScan(
        torch, import_module("torch.utils._pytree"), max_steps, max_elements
    )
    # Torch specialization caches use code identity. Isolate each explicit
    # adapter's recompilation budget without retaining model inputs globally.
    unique = FunctionType(_TorchScan.run.__code__.replace(), globals())
    compiled = torch.compile(
        MethodType(unique, driver), backend="inductor", fullgraph=True, dynamic=False
    )

    def scan(body: LoopBody, carry: Any, inputs: Any) -> tuple[Any, Any]:  # noqa: ANN401
        if torch.compiler.is_compiling():
            return driver.run(body, carry, inputs)
        driver.check_inputs(carry, inputs)
        return cast("tuple[Any, Any]", compiled(body, carry, inputs))

    return LoopAdapter(scan=cast("ScanFunction", scan))


@dataclass(frozen=True, slots=True)
class _TorchScan:
    torch: Any
    pytree: Any
    max_steps: int
    max_elements: int

    def leaves(self, tree: Any) -> list[Any]:  # noqa: ANN401
        result = []
        for leaf in self.pytree.tree_leaves(tree):
            if leaf is None:
                continue
            if not isinstance(leaf, self.torch.Tensor):
                msg = "Compiled loop trees must contain only Tensor/None leaves"
                raise TypeError(msg)
            if leaf.device.type != "cpu" or leaf.dtype not in (  # noqa: PLR6201
                self.torch.float32,
                self.torch.float64,
            ):
                msg = "Compiled loop tensors must be CPU float32/float64"
                raise ValueError(msg)
            result.append(leaf)
        return result

    def check_inputs(self, carry: Any, inputs: Any) -> tuple[int, int]:  # noqa: ANN401
        leaves = self.leaves(inputs)
        if not leaves or any(value.ndim == 0 for value in leaves):
            msg = "Compiled scan inputs require a leading step dimension"
            raise ValueError(msg)
        length = leaves[0].shape[0]
        if not 1 <= length <= self.max_steps:
            msg = "Compiled scan length must be positive and within max_steps"
            raise ValueError(msg)
        if any(value.shape[0] != length for value in leaves):
            msg = "Compiled scan inputs must share a leading step dimension"
            raise ValueError(msg)
        leaves += self.leaves(carry)
        if any(value.dtype != leaves[0].dtype for value in leaves):
            msg = "Compiled scan tensors must use one floating dtype"
            raise ValueError(msg)
        elements = sum(value.numel() for value in leaves)
        if elements > self.max_elements:
            msg = "Compiled scan input/carry exceeds max_elements"
            raise ValueError(msg)
        return length, elements

    def signature(self, tree: Any, reference: Any) -> tuple[Any, list[tuple[int, ...]]]:  # noqa: ANN401
        leaves = self.leaves(tree)
        if any(value.dtype != reference.dtype for value in leaves):
            msg = "Compiled scan body must preserve the floating dtype"
            raise ValueError(msg)
        _, structure = self.pytree.tree_flatten(tree)
        return structure, [tuple(value.shape) for value in leaves]

    def run(self, body: LoopBody, carry: Any, inputs: Any) -> tuple[Any, Any]:  # noqa: ANN401
        if not self.torch.compiler.is_compiling():
            msg = "Required Torch loop compilation is disabled"
            raise RuntimeError(msg)
        length, elements = self.check_inputs(carry, inputs)
        reference = self.leaves(inputs)[0]
        carry_signature = self.signature(carry, reference)
        outputs = []
        output_signature = None
        for index in range(length):
            carry, output = body(
                carry,
                self.pytree.tree_map(
                    lambda value, step=index: None if value is None else value[step],
                    inputs,
                ),
            )
            if self.signature(carry, reference) != carry_signature:
                msg = "Compiled scan body must preserve carry structure and shapes"
                raise ValueError(msg)
            signature = self.signature(output, reference)
            if index == 0:
                output_signature = signature
                stacked_size = length * sum(
                    value.numel() for value in self.leaves(output)
                )
                if elements + stacked_size > self.max_elements:
                    msg = "Compiled scan stacked output exceeds max_elements"
                    raise ValueError(msg)
            elif signature != output_signature:
                msg = "Compiled scan output structure and shapes must remain fixed"
                raise ValueError(msg)
            outputs.append(output)
        stacked = self.pytree.tree_map(
            lambda *values: None if values[0] is None else self.torch.stack(values),
            *outputs,
        )
        return carry, stacked


__all__ = ["torch_compile_loop_adapter"]
