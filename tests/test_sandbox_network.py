from pathlib import Path

from ar_kernel.sandbox.runner import Mounts, _docker_args


def _args(**kw):
    m = Mounts(agent=Path("/a"), workspace=Path("/w"), staging=Path("/s"), context=Path("/c"),
               contract=Path("/k"), sockets=Path("/so"), agent_readonly=False)
    return _docker_args("img", "name", m, ["true"], {}, 1, 1, **kw)


def test_containers_have_network_access_by_default():
    a = _args()
    assert a[a.index("--network") + 1] == "bridge"


def test_a_configured_network_is_passed_to_docker():
    a = _args(network="none")
    assert a[a.index("--network") + 1] == "none"
    assert "--read-only" in a and "--cap-drop" in a          # the rest of the sandbox is unchanged

