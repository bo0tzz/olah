"""Serving cached content to recently verified callers while the Hub rate-limits."""
import json
import os
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from olah.configs import OlahConfig
from olah.errors import UpstreamRateLimited
from olah.server import upstream_rate_limited_handler
from olah.server_routes import router
from olah.utils.rate_limit_fallback import RateLimitFallbackMiddleware

HTTP_CLIENT = httpx.AsyncClient
SHA = "1" * 40
CONTENT = b"tiny file content"
FILE = "/team/demo/resolve/main/file.bin"
RATE_LIMIT_HEADERS = {"retry-after": "60", "ratelimit": '"resolvers";r=0;t=60'}


class _Bytes(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data

    async def __aiter__(self):
        yield self.data


def _json(payload):
    return httpx.Response(200, headers={"content-type": "application/json"}, stream=_Bytes(json.dumps(payload).encode()))


@pytest.fixture
def env(tmp_path, monkeypatch):
    config = OlahConfig()
    config.repos_path = str(tmp_path / "cache")
    config.hf_netloc = "upstream.invalid"
    app = FastAPI()
    app.state.app_settings = SimpleNamespace(config=config)
    app.include_router(router)
    app.add_exception_handler(UpstreamRateLimited, upstream_rate_limited_handler)
    app.add_middleware(RateLimitFallbackMiddleware)
    state = SimpleNamespace(config=config, rate_limited=False, calls=[])

    async def upstream(request):
        state.calls.append((request.method, request.url.path))
        if state.rate_limited:
            return httpx.Response(429, headers=RATE_LIMIT_HEADERS)
        path = request.url.path
        if "/paths-info/" in path:
            return _json([{"type": "file", "path": "file.bin", "size": len(CONTENT), "oid": "2" * 40}])
        if "/resolve/" in path and request.method == "HEAD":
            return httpx.Response(200, headers={"etag": '"blob"', "content-length": str(len(CONTENT)), "x-repo-commit": SHA})
        if "/resolve/" in path:
            start, end = (int(x) for x in request.headers["range"].removeprefix("bytes=").split("-"))
            data = CONTENT[start : end + 1]
            return httpx.Response(
                206,
                headers={"content-length": str(len(data)), "content-range": f"bytes {start}-{end}/{len(CONTENT)}"},
                stream=_Bytes(data),
            )
        if path.endswith("/tree/" + SHA + "/"):
            return _json([{"type": "file", "path": "file.bin", "size": len(CONTENT), "oid": "2" * 40}])
        return _json({"id": "team/demo", "sha": SHA})

    class UpstreamClient(HTTP_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.MockTransport(upstream))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", UpstreamClient)

    async def request(path=FILE, token="Bearer alice"):
        async with HTTP_CLIENT(transport=httpx.ASGITransport(app=app), base_url="http://olah.test", follow_redirects=True) as client:
            return await client.get(path, headers={"authorization": token})

    state.request = request
    state.access_dir = os.path.join(config.repos_path, "access", "models", "team", "demo")
    return state


@pytest.mark.asyncio
async def test_verified_caller_is_served_from_cache_while_rate_limited(env):
    assert (await env.request()).content == CONTENT
    env.rate_limited = True
    env.calls.clear()

    response = await env.request()

    assert response.status_code == 200
    assert response.content == CONTENT
    assert response.headers["x-repo-commit"] == SHA
    assert env.calls == [("HEAD", FILE)]


@pytest.mark.asyncio
async def test_unverified_caller_gets_the_rate_limit(env):
    assert (await env.request()).status_code == 200
    env.rate_limited = True

    response = await env.request(token="Bearer mallory")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"


@pytest.mark.asyncio
async def test_cache_miss_while_rate_limited_is_the_rate_limit_not_a_404(env):
    assert (await env.request()).status_code == 200
    env.rate_limited = True

    response = await env.request("/team/demo/resolve/main/other.bin")

    assert response.status_code == 429
    assert response.headers["ratelimit"] == RATE_LIMIT_HEADERS["ratelimit"]


@pytest.mark.asyncio
async def test_expired_verification_gets_the_rate_limit(env):
    assert (await env.request()).status_code == 200
    stale = time.time() - env.config.rate_limit_fallback_ttl - 1
    for marker in os.scandir(env.access_dir):
        if marker.is_file():
            os.utime(marker.path, (stale, stale))
    env.rate_limited = True

    assert (await env.request()).status_code == 429


@pytest.mark.asyncio
async def test_zero_ttl_disables_the_fallback_and_records_nothing(env):
    env.config.rate_limit_fallback_ttl = 0
    assert (await env.request()).status_code == 200
    env.rate_limited = True

    assert (await env.request()).status_code == 429
    assert not os.path.exists(env.access_dir)


@pytest.mark.asyncio
async def test_api_routes_fall_back_too(env):
    tree = f"/api/models/team/demo/tree/{SHA}?recursive=true"
    assert (await env.request(tree)).status_code == 200
    env.rate_limited = True

    response = await env.request(tree)

    assert response.status_code == 200
    assert response.json()[0]["path"] == "file.bin"
