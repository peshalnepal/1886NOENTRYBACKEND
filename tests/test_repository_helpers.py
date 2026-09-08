import unittest

from pydantic import BaseModel

from application.repositories._helpers import model_patch


class PatchModel(BaseModel):
    name: str | None = None
    enabled: bool | None = None


class RepositoryHelpersTests(unittest.TestCase):
    def test_model_patch_keeps_explicit_none_by_default(self):
        patch = model_patch(PatchModel(name=None))

        self.assertEqual(patch, {"name": None})

    def test_model_patch_can_drop_none_values(self):
        patch = model_patch(PatchModel(name=None, enabled=True), drop_none=True)

        self.assertEqual(patch, {"enabled": True})


if __name__ == "__main__":
    unittest.main()
