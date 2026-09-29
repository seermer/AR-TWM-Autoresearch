from pathlib import Path

from ar_kernel.sandbox.runner import Mounts, _docker_args


def _args(**kw):
    m = Mounts(agent=Path("/a"), workspace=Path("/w"), staging=Path("/s"), context=Path("/c"),
               store=Path("/st"), contract=Path("/k"), sockets=Path("/so"), agent_readonly=False)
    return _docker_args("img", "name", m, ["true"], {}, 1, 1, **kw)


def test_containers_are_offline_unless_a_network_is_asked_for():
    a = _args()
    assert a[a.index("--network") + 1] == "none"


def test_a_configured_network_is_passed_to_docker():
    a = _args(network="bridge")
    assert a[a.index("--network") + 1] == "bridge"
    assert "--read-only" in a and "--cap-drop" in a          # the rest of the sandbox is unchanged
