import asyncio
import os
import unittest
from unittest.mock import patch

import httpx

from application.services.edgeinference import EdgeCameraInventoryError, EdgeInferenceClient


class EdgeClientTimeoutTests(unittest.TestCase):
    def test_edge_client_disables_proxy_env_by_default(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.internal:3128"}, clear=False):
            os.environ.pop("EDGE_HTTP_TRUST_ENV", None)
            client = EdgeInferenceClient()
            try:
                self.assertFalse(client.trust_env)
                self.assertFalse(getattr(client._client, "_trust_env", None))
            finally:
                asyncio.run(client.close())

    def test_list_cameras_uses_inventory_timeout_and_short_health_probe(self):
        async def scenario():
            seen = []
            with patch.dict(
                os.environ,
                {
                    "EDGE_TIMEOUT_S": "15",
                    "EDGE_CONNECT_TIMEOUT_S": "10",
                    "EDGE_LIST_TIMEOUT_S": "4",
                    "EDGE_LIST_CONNECT_TIMEOUT_S": "2",
                    "EDGE_HEALTH_TIMEOUT_S": "1.25",
                    "EDGE_HEALTH_CONNECT_TIMEOUT_S": "0.75",
                    "EDGE_HTTP_RETRIES": "1",
                },
                clear=False,
            ):
                client = EdgeInferenceClient()

                async def fake_request(method, url, **kwargs):
                    seen.append(("request", method, url, kwargs.get("timeout")))
                    raise httpx.ReadTimeout("inventory stalled")

                async def fake_get(url, **kwargs):
                    seen.append(("health", "GET", url, kwargs.get("timeout")))
                    raise httpx.ReadTimeout("health stalled")

                client._client.request = fake_request
                client._client.get = fake_get

                try:
                    with self.assertRaises(EdgeCameraInventoryError) as ctx:
                        await client.list_cameras(device_url="http://edge.example:19030")

                    message = str(ctx.exception)
                    self.assertIn(
                        "ReadTimeout during GET http://edge.example:19030/api/cameras: inventory stalled",
                        message,
                    )

                    inventory_call = next(item for item in seen if item[0] == "request")
                    inventory_timeout = inventory_call[3]
                    self.assertAlmostEqual(inventory_timeout.read, 4.0)
                    self.assertAlmostEqual(inventory_timeout.connect, 2.0)

                    health_calls = [item for item in seen if item[0] == "health"]
                    self.assertEqual(
                        [item[2] for item in health_calls],
                        [
                            "http://edge.example:19030/health",
                            "http://edge.example:19030/api/health",
                        ],
                    )
                    for _, _, _, timeout in health_calls:
                        self.assertAlmostEqual(timeout.read, 1.25)
                        self.assertAlmostEqual(timeout.connect, 0.75)
                finally:
                    await client.close()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
