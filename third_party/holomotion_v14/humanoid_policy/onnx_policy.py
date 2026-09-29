"""Compatibility alias for the shared ONNX/TensorRT policy loader."""

import sys

from holomotion_policy_core import onnx_policy as _implementation


sys.modules[__name__] = _implementation
