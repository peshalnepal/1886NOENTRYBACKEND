import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from application.services.alert_image_storage import AlertImageStorageService
from application.services.clip_storage import EventClipService


class SignedBlobUrlTests(unittest.TestCase):
    def services(self):
        for service_type, ttl in ((EventClipService, 168), (AlertImageStorageService, 720)):
            service = object.__new__(service_type)
            service._sas_account_name = "test-account"
            service._sas_account_key = "test-key-not-a-real-credential"
            service.container_name = "test-container"
            service.SAS_TTL_HOURS = ttl
            yield service

    def test_services_share_read_only_signing_with_their_own_expiry(self):
        for service in self.services():
            with self.subTest(service=type(service).__name__):
                before = datetime.now(timezone.utc)
                with patch("application.services.storage_common.generate_blob_sas", return_value="sas-token") as sign:
                    result = service._signed_url(blob_name="image.jpg", blob_url="https://storage.test/image.jpg")
                after = datetime.now(timezone.utc)
                self.assertEqual(result, "https://storage.test/image.jpg?sas-token")
                sign.assert_called_once()
                arguments = sign.call_args.kwargs
                self.assertEqual(arguments["account_name"], service._sas_account_name)
                self.assertEqual(arguments["account_key"], service._sas_account_key)
                self.assertEqual(arguments["container_name"], service.container_name)
                self.assertEqual(arguments["blob_name"], "image.jpg")
                self.assertEqual(str(arguments["permission"]), "r")
                self.assertGreaterEqual(arguments["expiry"], before + timedelta(hours=service.SAS_TTL_HOURS))
                self.assertLessEqual(arguments["expiry"], after + timedelta(hours=service.SAS_TTL_HOURS))

    def test_missing_account_name_or_key_returns_original_url(self):
        for service in self.services():
            for field in ("_sas_account_name", "_sas_account_key"):
                with self.subTest(service=type(service).__name__, field=field):
                    with patch.object(service, field, ""):
                        with patch("application.services.storage_common.generate_blob_sas") as sign:
                            result = service._signed_url(blob_name="image.jpg", blob_url="https://storage.test/image.jpg")
                    self.assertEqual(result, "https://storage.test/image.jpg")
                    sign.assert_not_called()

    def test_empty_token_returns_original_url(self):
        for service in self.services():
            with self.subTest(service=type(service).__name__):
                with patch("application.services.storage_common.generate_blob_sas", return_value=""):
                    self.assertEqual(
                        service._signed_url(blob_name="image.jpg", blob_url="https://storage.test/image.jpg"),
                        "https://storage.test/image.jpg",
                    )

    def test_signing_errors_still_propagate(self):
        for service in self.services():
            with self.subTest(service=type(service).__name__):
                with patch("application.services.storage_common.generate_blob_sas", side_effect=ValueError("invalid key")):
                    with self.assertRaisesRegex(ValueError, "invalid key"):
                        service._signed_url(blob_name="image.jpg", blob_url="https://storage.test/image.jpg")
