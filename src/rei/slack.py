"""Slack Web API client: paged history, thread harvesting, and posting.

Rate limits, as of the May 2025 changes: `conversations.history` and
`conversations.replies` are throttled to 1 request per minute and 15 objects per
page for *non-Marketplace distributed* apps. Internal apps built inside your own
workspace — which is what this bot is — keep the old Tier 3 behaviour of roughly
50 requests per minute and up to 1,000 objects per page. The client below does
not assume either: it requests a large page, observes what it actually receives,
and adapts its pacing, so a full backfill completes correctly under either
regime (just slower under the strict one). Retry-After is always honoured.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from .http import HttpClient

log = logging.getLogger(__name__)

BASE = "https://slack.com/api"
# Below this page size we are being served the restricted tier and must pace at
# roughly one request per minute to avoid a wall of 429s.
RESTRICTED_PAGE_HINT = 20
RESTRICTED_DELAY_SECONDS = 61.0
NORMAL_DELAY_SECONDS = 1.2


class SlackError(RuntimeError):
    pass


class SlackClient:
    def __init__(self, token: str, *, page_size: int = 200) -> None:
        self.token = token
        self.page_size = page_size
        self._http = HttpClient("slack", timeout=45.0)
        self._restricted = False

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SlackClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _get(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        data = self._http.request_json(
            "GET", f"{BASE}/{method}", headers=self._auth_headers, params=params
        )
        if not data.get("ok"):
            raise SlackError(f"{method} failed: {data.get('error')} ({data})")
        return data

    def _post(self, method: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {**self._auth_headers, "Content-Type": "application/json; charset=utf-8"}
        data = self._http.request_json("POST", f"{BASE}/{method}", headers=headers, json_body=body)
        if not data.get("ok"):
            raise SlackError(f"{method} failed: {data.get('error')} ({data})")
        return data

    def auth_test(self) -> dict[str, Any]:
        return self._get("auth.test", {})

    def fetch_history(
        self,
        channel: str,
        *,
        oldest: float | None = None,
        latest: float | None = None,
        max_messages: int | None = None,
        progress_every: int = 500,
    ) -> list[dict[str, Any]]:
        """Page through channel history, oldest bound inclusive.

        Passing oldest=None fetches everything the workspace retains, which is
        what the initial backfill does.
        """
        params: dict[str, Any] = {
            "channel": channel,
            "limit": self.page_size,
            "inclusive": True,
        }
        if oldest is not None:
            params["oldest"] = f"{oldest:.6f}"
        if latest is not None:
            params["latest"] = f"{latest:.6f}"

        messages: list[dict[str, Any]] = []
        pages = 0

        while True:
            data = self._get("conversations.history", params)
            batch = data.get("messages", [])
            messages.extend(batch)
            pages += 1

            if pages == 1 and 0 < len(batch) <= RESTRICTED_PAGE_HINT < self.page_size:
                self._restricted = True
                log.warning(
                    "Slack returned only %d messages for a page size of %d — this app is on "
                    "the restricted tier. Pacing at one request per minute.",
                    len(batch),
                    self.page_size,
                )

            if len(messages) >= progress_every and pages % 5 == 0:
                log.info("fetched %d messages across %d pages", len(messages), pages)

            if max_messages is not None and len(messages) >= max_messages:
                return messages[:max_messages]
            if not data.get("has_more"):
                break
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            params["cursor"] = cursor
            time.sleep(RESTRICTED_DELAY_SECONDS if self._restricted else NORMAL_DELAY_SECONDS)

        log.info("fetched %d messages across %d pages", len(messages), pages)
        return messages

    def fetch_replies(self, channel: str, thread_ts: str) -> list[dict[str, Any]]:
        data = self._get(
            "conversations.replies", {"channel": channel, "ts": thread_ts, "limit": self.page_size}
        )
        return data.get("messages", [])

    def post_message(
        self,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
    ) -> str:
        body: dict[str, Any] = {"channel": channel, "text": text}
        if blocks:
            body["blocks"] = blocks
        if thread_ts:
            body["thread_ts"] = thread_ts
        data = self._post("chat.postMessage", body)
        return str(data.get("ts", ""))


def window_bounds(days: int, now: datetime | None = None) -> tuple[datetime, datetime]:
    end = now or datetime.now(tz=UTC)
    return end - timedelta(days=days), end


def harvest_resolutions(
    client: SlackClient, report_channel: str, since_days: int = 30
) -> list[tuple[str, str, str | None, str]]:
    """Pull human replies to past report messages.

    When someone replies in-thread to a daily report — "this was the partner-API
    cert expiring, fixed" — that is the single most valuable text in the whole
    system, because it is a real resolution written by someone who knew the
    answer. Harvesting those turns the report channel itself into the RAG
    corpus, at no extra integration cost.

    Returns (parent_ts, text, author, reply_ts) tuples for the caller to attach
    to clusters.
    """
    since = (datetime.now(tz=UTC) - timedelta(days=since_days)).timestamp()
    parents = client.fetch_history(report_channel, oldest=since)
    out: list[tuple[str, str, str | None, str]] = []

    for parent in parents:
        if not parent.get("thread_ts") or int(parent.get("reply_count", 0)) == 0:
            continue
        for reply in client.fetch_replies(report_channel, parent["thread_ts"]):
            if reply.get("ts") == parent.get("ts"):
                continue
            if reply.get("bot_id") or reply.get("subtype") == "bot_message":
                continue  # our own thread of quote IDs, not a human note
            text = (reply.get("text") or "").strip()
            if len(text) < 15:
                continue
            out.append((parent["ts"], text, reply.get("user"), reply["ts"]))
    return out
