"""HTTP boundary to Jetson. NVR never imports the GPU runtime."""

import httpx


class JetsonError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        # Never include the upstream body: it can contain authenticated RTSP URLs.
        super().__init__("Jetson request failed (HTTP {})".format(status_code))


class JetsonClient:
    def __init__(self, settings, *, transport=None):
        self.http = httpx.AsyncClient(
            base_url=str(settings.jetson_base_url).rstrip("/"),
            timeout=httpx.Timeout(settings.jetson_timeout_s, connect=5.0),
            transport=transport,
            trust_env=False,
        )

    async def close(self):
        await self.http.aclose()

    async def request(self, method, path, **kwargs):
        """Decoded JSON, or JetsonError (502 for transport failures / bad JSON)."""
        try:
            response = await self.http.request(method, path, **kwargs)
        except httpx.RequestError:
            raise JetsonError(502) from None
        if not response.is_success:
            raise JetsonError(response.status_code)
        try:
            return response.json()
        except ValueError:
            raise JetsonError(502) from None

    async def cameras(self):
        payload = await self.request("GET", "/cameras")
        if not isinstance(payload, dict) or not isinstance(payload.get("cameras"), list):
            raise JetsonError(502)
        return payload["cameras"]
