"""Routes whose database work must outlive an unavailable runtime.

`get_manager` raises 503 before the handler body runs, so a route that declares
`Optional[Manager]` and guards for `None` must be wired to `get_manager_optional`
— otherwise the guard is unreachable and the delete fails exactly when the
manager is down.
"""

import inspect
import typing
import unittest

from fastapi.params import Depends

from dependencies import get_manager, get_manager_optional
from routes import camera_routes, device_routes, site_routes

# (module, handler name) for every route that tolerates a missing manager.
MANAGER_OPTIONAL_ROUTES = [
    (device_routes, "delete_device"),
    (site_routes, "delete_site"),
    (site_routes, "link_device_to_site"),
    (site_routes, "update_site"),
    (site_routes, "update_site_settings"),
    (camera_routes, "delete_camera"),
]

# Routes that genuinely cannot do their work without the manager.
MANAGER_REQUIRED_ROUTES = [
    (site_routes, "create_site_camera"),
    (device_routes, "reconcile_edge_cameras"),
]


def _manager_param(module, handler_name):
    handler = getattr(module, handler_name)
    param = inspect.signature(handler).parameters["manager"]
    if not isinstance(param.default, Depends):
        raise AssertionError(f"{handler_name}: manager is not a Depends(...)")
    return param


def _is_optional(annotation) -> bool:
    return type(None) in typing.get_args(annotation)


class ManagerDependencyWiringTests(unittest.TestCase):
    def test_optional_manager_routes_use_the_non_raising_dependency(self):
        for module, handler_name in MANAGER_OPTIONAL_ROUTES:
            with self.subTest(handler=handler_name):
                param = _manager_param(module, handler_name)
                self.assertIs(
                    param.default.dependency,
                    get_manager_optional,
                    f"{handler_name} guards for a missing manager, so it must not "
                    f"depend on get_manager (which 503s first)",
                )
                self.assertTrue(
                    _is_optional(param.annotation),
                    f"{handler_name} must be annotated Optional[Manager]",
                )

    def test_manager_required_routes_still_fail_fast(self):
        for module, handler_name in MANAGER_REQUIRED_ROUTES:
            with self.subTest(handler=handler_name):
                param = _manager_param(module, handler_name)
                self.assertIs(param.default.dependency, get_manager)
                self.assertFalse(_is_optional(param.annotation))


class RouteBoundaryTests(unittest.TestCase):
    def test_site_routes_does_not_import_camera_route_internals(self):
        """Both create paths share `routes._camera_serializers` instead."""
        source = inspect.getsource(site_routes)
        self.assertNotIn("from routes.camera_routes import", source)


if __name__ == "__main__":
    unittest.main()
