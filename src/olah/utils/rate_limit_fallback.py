# coding=utf-8
# Copyright 2024 XiaHan
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Serve from cache, as offline mode does, when the Hub rate-limits an access check.

Only for a token (or anonymous caller) the Hub granted access to the repo within
``rate-limit-fallback-ttl`` seconds; the grant is recorded as a marker file's
mtime under ``<repos-path>/access``.
"""

import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

from olah.errors import UpstreamRateLimited
from olah.utils.auth_utils import token_hash


@dataclass
class _RequestState:
    rate_limited: Optional[UpstreamRateLimited] = None


_request_state: ContextVar[Optional[_RequestState]] = ContextVar("rate_limit_fallback", default=None)


def is_offline(app) -> bool:
    """Whether this request must not contact the Hub: offline mode, or serving through a rate limit."""
    if app.state.app_settings.config.offline:
        return True
    state = _request_state.get()
    return state is not None and state.rate_limited is not None


def _ttl(app) -> int:
    return getattr(app.state.app_settings.config, "rate_limit_fallback_ttl", 0)


def _repo_dir(app, repo_type: str, org: Optional[str], repo: str) -> str:
    return os.path.join(app.state.app_settings.config.repos_path, "access", repo_type, org or "", repo)


def record_access(app, repo_type: str, org: Optional[str], repo: str, authorization: Optional[str]) -> None:
    if _ttl(app) <= 0:
        return
    marker = os.path.join(_repo_dir(app, repo_type, org, repo), token_hash(authorization))
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "a"):
        os.utime(marker)


def _revision_path(app, repo_type: str, org: Optional[str], repo: str, revision: str) -> Optional[str]:
    if any(part in ("", ".", "..") for part in revision.split("/")):
        return None
    return os.path.join(_repo_dir(app, repo_type, org, repo), "revisions", revision)


def record_revision(app, repo_type: str, org: Optional[str], repo: str, revision: str, commit: str) -> None:
    path = _revision_path(app, repo_type, org, repo, revision)
    if _ttl(app) <= 0 or revision == commit or path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(commit)
    os.replace(tmp, path)


def recorded_revision(app, repo_type: str, org: Optional[str], repo: str, revision: str) -> Optional[str]:
    path = _revision_path(app, repo_type, org, repo, revision)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def serve_from_cache_if_granted(
    app,
    repo_type: str,
    org: Optional[str],
    repo: str,
    authorization: Optional[str],
    rate_limited: UpstreamRateLimited,
) -> None:
    """Switch this request to cache-only mode if the caller was recently granted access; else re-raise."""
    ttl = _ttl(app)
    marker = os.path.join(_repo_dir(app, repo_type, org, repo), token_hash(authorization))
    try:
        granted = ttl > 0 and time.time() - os.stat(marker).st_mtime < ttl
    except OSError:
        granted = False
    if not granted:
        raise rate_limited
    state = _request_state.get()
    if state is None:
        state = _RequestState()
        _request_state.set(state)
    state.rate_limited = rate_limited


class RateLimitFallbackMiddleware:
    """Replace an error produced while serving through a rate limit with the Hub's 429.

    A cache miss in cache-only mode yields offline mode's 404/401, which a
    client would take as final; the 429 makes it back off and retry instead.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = _RequestState()
        token = _request_state.set(state)
        replaced = False

        async def guarded_send(message):
            nonlocal replaced
            if replaced:
                return
            if (
                message["type"] == "http.response.start"
                and message["status"] >= 400
                and state.rate_limited is not None
            ):
                replaced = True
                await state.rate_limited.response()(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, receive, guarded_send)
        finally:
            _request_state.reset(token)
