"""`--interface real` must follow the robot to whatever LAN it is on (factory 192.168.123.x first, else the NIC sharing a /24 with --robot-ip)."""
import decoupled_wbc.control.utils.network_utils as nu


def _with(fake, monkeypatch):
    monkeypatch.setattr(nu, "get_network_interfaces", lambda: fake)


def test_factory_lan_wins(monkeypatch):
    _with({"lo": ["127.0.0.1"], "eno1": ["192.168.50.243"], "enx6c1ff7cdf694": ["192.168.123.222"]}, monkeypatch)
    assert nu.resolve_interface("real", "192.168.50.67") == ("enx6c1ff7cdf694", "real")


def test_follows_robot_ip_wired_first(monkeypatch):
    _with({"lo": ["127.0.0.1"], "wlp7s0": ["192.168.50.220"], "eno1": ["192.168.50.243"], "enx6c1ff7cdf694": []}, monkeypatch)
    assert nu.resolve_interface("real", "192.168.50.67") == ("eno1", "real")
    monkeypatch.delenv("ROBOT_IP", raising=False)
    assert nu.resolve_interface("real", None) == ("real", "real")           # nothing to go on: unchanged fallback
    monkeypatch.setenv("ROBOT_IP", "192.168.50.67")
    assert nu.resolve_interface("real", None) == ("eno1", "real")


def test_explicit_values_untouched(monkeypatch):
    _with({"lo": ["127.0.0.1"], "eno1": ["192.168.50.243"]}, monkeypatch)
    assert nu.resolve_interface("eno1", "192.168.50.67") == ("eno1", "real")
    assert nu.resolve_interface("192.168.50.243", None) == ("192.168.50.243", "real")
    assert nu.resolve_interface("sim", None)[1] == "sim"
