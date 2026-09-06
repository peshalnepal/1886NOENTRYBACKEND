# routes/runtime_ref.py
"""
Module-level handle to the PipelineRuntime singleton.

The route blueprints need the runtime, but the runtime is constructed in
main.py — importing it from there would be circular. main.py calls
`set_runtime()` once after constructing it; blueprints call `get_runtime()`
at request time (not import time), so the binding is always resolved.
"""

_runtime = None


def set_runtime(runtime):
    global _runtime
    _runtime = runtime


def get_runtime():
    if _runtime is None:
        raise RuntimeError("Pipeline runtime has not been initialized")
    return _runtime
