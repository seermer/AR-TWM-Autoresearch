from pathlib import Path

from ar_kernel.sandbox.runner import Mounts, _run_kwargs


def _args(**kw):
    m = Mounts(agent=Path("/a"), workspace=Path("/w"), staging=Path("/s"), context=Path("/c"),
               contract=Path("/k"), sockets=Path("/so"), agent_readonly=False)
    return _run_kwargs("img", "name", m, ["true"], {}, 1, 1, **kw)


def test_containers_have_network_access_by_default():
    assert _args()["network_mode"] == "bridge"


def test_a_configured_network_is_passed_to_docker():
    a = _args(network="none")
    assert a["network_mode"] == "none"
    assert a["read_only"] and a["cap_drop"] == ["ALL"]       # the rest of the sandbox is unchanged

