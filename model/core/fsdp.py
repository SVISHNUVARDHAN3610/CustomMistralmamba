"""Small FSDP2/DTensor compatibility helpers."""

from __future__ import annotations

from torch import Tensor


def local_dtensor(tensor: Tensor) -> Tensor:
    """Return the local value for FSDP2 DTensors, otherwise the tensor itself.

    FSDP2 stores parameters as DTensors. Most stock modules handle those
    through FSDP2's dispatch paths, but custom hand-written math such as
    ``activation * parameter`` or stacked-weight ``bmm`` cannot mix regular
    activation tensors with DTensor parameters. During an FSDP2 forward,
    pre-forward hooks have already unsharded parameters, so ``to_local()``
    provides the compatible local tensor while preserving autograd routing.
    """
    if type(tensor).__name__ == "DTensor":
        return tensor.to_local()
    return tensor
