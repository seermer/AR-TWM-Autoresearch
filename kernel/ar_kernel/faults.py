"""Failures that are not the agent's: the phase runs once more, then the run stops (KernelFault)
or the node ends (NodeFault)."""
from __future__ import annotations

import errno

DISK_ERRNOS = {errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS}


class KernelFault(Exception):
    """A clearly identified failure of the machine, the kernel or the LLM provider."""


class NodeFault(Exception):
    """The node's own model broke training or eval. `status`: the node's status once it ends."""

    def __init__(self, message: str, status: str | None = None) -> None:
        super().__init__(message)
        self.status = status


def disk_fault(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno in DISK_ERRNOS
