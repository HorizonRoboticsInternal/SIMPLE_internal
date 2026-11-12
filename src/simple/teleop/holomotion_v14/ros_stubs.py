"""
Minimal stand-ins for the ROS 2 modules ``humanoid_policy.policy_node_29dof`` imports.

On the robot the node is an rclpy ``Node`` wired to topics. In SIMPLE the same class is driven synchronously from the
sim loop, so all it needs from ROS is: parameters (``declare_parameter``), a logger, publishers that remember the last
message, and message types that accept the attributes the node sets. Subscriptions and timers are inert: the sim node
calls the node's callbacks itself.

Installed only for modules that are not importable already, so a real ROS environment is never shadowed.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any


class _Msg:
    """Permissive message: any attribute can be set, unknown attributes read as nested messages."""

    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Msg()
        setattr(self, name, value)
        return value


def _msg_type(name: str) -> type:
    return type(name, (_Msg,), {})


class _Param:
    def __init__(self, value: Any) -> None:
        self.value = value


class _Publisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.last = None
        self.count = 0

    def publish(self, msg: Any) -> None:
        self.last = msg
        self.count += 1


class _Logger:
    def __init__(self, name: str, quiet: bool) -> None:
        self.name = name
        self.quiet = quiet

    def _emit(self, level: str, msg: str) -> None:
        if self.quiet and level in ("DEBUG", "INFO"):
            return
        print(f"[v14 {self.name}] {level}: {msg}", flush=True)

    def debug(self, msg: str, *a, **k) -> None:
        pass

    def info(self, msg: str, *a, **k) -> None:
        self._emit("INFO", msg)

    def warn(self, msg: str, *a, **k) -> None:
        self._emit("WARN", msg)

    warning = warn

    def error(self, msg: str, *a, **k) -> None:
        self._emit("ERROR", msg)


class Node:
    """rclpy.node.Node stand-in. Parameters come from ``Node.parameter_overrides`` (set before construction)."""

    parameter_overrides: dict[str, Any] = {}
    quiet_logs: bool = False

    def __init__(self, node_name: str, *args, **kwargs) -> None:
        self._node_name = node_name
        self._params: dict[str, _Param] = {}
        self._logger = _Logger(node_name, self.quiet_logs)
        self._publishers: dict[str, _Publisher] = {}

    def get_logger(self) -> _Logger:
        return self._logger

    def declare_parameter(self, name: str, default: Any = None, *args, **kwargs) -> _Param:
        param = _Param(self.parameter_overrides.get(name, default))
        self._params[name] = param
        return param

    def get_parameter(self, name: str) -> _Param:
        return self._params.get(name, _Param(self.parameter_overrides.get(name)))

    def create_publisher(self, msg_type: Any, topic: str, *args, **kwargs) -> _Publisher:
        pub = _Publisher(topic)
        self._publishers[topic] = pub
        return pub

    def create_subscription(self, *args, **kwargs) -> None:
        return None

    def create_timer(self, *args, **kwargs) -> None:
        return None

    def get_clock(self) -> Any:
        return _Msg(now=lambda: _Msg(to_msg=lambda: _Msg(sec=0, nanosec=0)))

    def destroy_node(self) -> None:
        pass


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def install(share_dir: str) -> None:
    """Register the stand-in modules (only those that do not import for real)."""

    def missing(name: str) -> bool:
        if name in sys.modules:
            return False
        try:
            importlib.import_module(name)
            return False
        except Exception:
            return True

    stubs: dict[str, types.ModuleType] = {}
    if missing("rclpy"):
        stubs["rclpy"] = _module("rclpy", init=lambda *a, **k: None, ok=lambda: False,
                                 shutdown=lambda *a, **k: None, spin=lambda *a, **k: None)
        stubs["rclpy.node"] = _module("rclpy.node", Node=Node, ParameterDescriptor=_msg_type("ParameterDescriptor"))
        stubs["rclpy.qos"] = _module("rclpy.qos", QoSProfile=_msg_type("QoSProfile"),
                                     DurabilityPolicy=_Msg(), HistoryPolicy=_Msg(), ReliabilityPolicy=_Msg())
        stubs["rclpy.executors"] = _module("rclpy.executors", ExternalShutdownException=type("ExternalShutdownException", (Exception,), {}))
    if missing("ament_index_python"):
        stubs["ament_index_python"] = _module("ament_index_python")
        stubs["ament_index_python.packages"] = _module("ament_index_python.packages",
                                                      get_package_share_directory=lambda _pkg: share_dir)
    for pkg, names in (("builtin_interfaces.msg", ["Time"]),
                       ("holomotion_interfaces.msg", ["MotionReference", "MotionTrackerReference"]),
                       ("robo_orchard_pico_msg_ros2.msg", ["VRState"]),
                       ("std_msgs.msg", ["Float32MultiArray", "String", "UInt8"]),
                       ("unitree_hg.msg", ["LowState", "HandCmd", "MotorCmd"])):
        if missing(pkg):
            parent = pkg.rsplit(".", 1)[0]
            if parent not in sys.modules and missing(parent):
                stubs[parent] = _module(parent)
            stubs[pkg] = _module(pkg, **{n: _msg_type(n) for n in names})
    sys.modules.update(stubs)
