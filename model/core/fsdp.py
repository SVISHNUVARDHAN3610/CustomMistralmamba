"""Small FSDP2/DTensor compatibility helpers."""

from __future__ import annotations

from torch import Tensor


def local_dtensor(tensor: Tensor) -> Tensor:
    """Return a regular Tensor for FSDP2 DTensors, otherwise tensor itself.

    FSDP2 stores parameters as DTensors. Most stock modules handle those
    through FSDP2's dispatch paths, but custom hand-written math such as
    ``activation * parameter`` or stacked-weight ``bmm`` cannot mix regular
    activation tensors with DTensor parameters. ``full_tensor()`` preserves the
    original global shape; ``to_local()`` only returns the rank-local shard and
    is therefore shape-wrong for replicated activation math.
    """
    if type(tensor).__name__ == "DTensor":
        return tensor.full_tensor()
    return tensor
