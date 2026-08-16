"""Embeddings, with a cache in front and a deterministic offline encoder behind.

Cost note. Templates are cached by content hash, so a template is embedded once
ever. After the initial backfill a normal day embeds only genuinely new
templates — typically a handful — which at voyage-3-lite pricing rounds to zero.
The cache is the reason this can run daily for free rather than re-embedding
several thousand strings every morning.
"""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np

from .config import Settings
from .http import ApiError, HttpClient
from .store import Store

log = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
OFFLINE_DIM = 512

# Voyage accounts without a payment method are capped at 3 requests and 10,000
# tokens per minute. That is fine for a daily run, which embeds a handful of new
# templates, but a first backfill of several hundred templates walks straight
# into it. Rather than failing, detect the tier from the 429 body and switch to
# small, paced batches: slower, but it finishes.
FREE_TIER_MARKER = "payment method"
FREE_TIER_RPM = 3
FREE_TIER_BATCH = 24


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def offline_embed(text: str, dim: int = OFFLINE_DIM) -> np.ndarray:
    """Deterministic hashed character-n-gram encoder.

    This is a *test double*, not a semantic model. It gives real lexical
    similarity — templates sharing wording land near each other — which is
    enough to exercise the whole pipeline, run CI without secrets, and provide
    an honest "lexical" arm in the eval. It is never used when a Voyage key is
    present. Anything it can cluster, a real embedding model can also cluster;
    the reverse is not true, which is exactly the gap the eval measures.
    """
    vector = np.zeros(dim, dtype=np.float32)
    normalised = " " + " ".join(text.lower().split()) + " "
    for n in (3, 4, 5):
        for i in range(max(len(normalised) - n + 1, 0)):
            gram = normalised[i : i + n]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[idx] += sign
    # Sublinear scaling damps the effect of very long templates.
    vector = np.sign(vector) * np.log1p(np.abs(vector))
    return _l2(vector)


class Embedder:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.api_calls = 0
        self.is_offline = settings.offline or not settings.voyage_api_key.get_secret_value()
        self.model = "offline-hash-v1" if self.is_offline else settings.voyage_model
        if self.is_offline and not settings.offline:
            log.warning("No REI_VOYAGE_API_KEY set — falling back to the offline encoder.")

    def embed(self, texts_by_hash: dict[str, str]) -> dict[str, np.ndarray]:
        """Embed a {template_hash: template_text} map, using the cache first."""
        if not texts_by_hash:
            return {}

        hashes = list(texts_by_hash)
        cached = self.store.get_cached_embeddings(hashes, self.model)
        missing = [h for h in hashes if h not in cached]
        log.info(
            "embeddings: %d requested, %d cache hits, %d to compute",
            len(hashes),
            len(cached),
            len(missing),
        )
        if not missing:
            return cached

        if self.is_offline:
            fresh = {h: offline_embed(texts_by_hash[h]) for h in missing}
        else:
            fresh = self._call_voyage({h: texts_by_hash[h] for h in missing})

        self.store.put_embeddings(fresh, self.model)
        return {**cached, **fresh}

    def _call_voyage(self, batch: dict[str, str]) -> dict[str, np.ndarray]:
        headers = {
            "Authorization": f"Bearer {self.settings.voyage_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        out: dict[str, np.ndarray] = {}
        items = list(batch.items())
        size = self.settings.embedding_batch_size
        min_interval = 0.0
        last_call = 0.0
        index = 0

        with HttpClient("voyage", max_attempts=3) as client:
            while index < len(items):
                chunk = items[index : index + size]
                payload = {
                    "input": [text for _h, text in chunk],
                    "model": self.settings.voyage_model,
                    "input_type": "document",
                    "truncation": True,
                }

                wait = min_interval - (time.monotonic() - last_call)
                if wait > 0:
                    time.sleep(wait)

                try:
                    response = client.request_json(
                        "POST", VOYAGE_URL, headers=headers, json_body=payload
                    )
                except ApiError as exc:
                    if exc.status == 429 and FREE_TIER_MARKER in exc.body and min_interval == 0.0:
                        min_interval = 60.0 / FREE_TIER_RPM
                        size = min(size, FREE_TIER_BATCH)
                        remaining = len(items) - index
                        log.warning(
                            "Voyage account has no payment method: throttling to %d req/min "
                            "in batches of %d. %d templates left, roughly %.0f minutes. "
                            "Adding a card at dashboard.voyageai.com removes this "
                            "(the 200M free tokens still apply).",
                            FREE_TIER_RPM,
                            size,
                            remaining,
                            remaining / size / FREE_TIER_RPM,
                        )
                        continue  # retry the same slice with a smaller batch
                    raise

                last_call = time.monotonic()
                self.api_calls += 1
                data = sorted(response.get("data", []), key=lambda d: d.get("index", 0))
                if len(data) != len(chunk):
                    raise RuntimeError(
                        f"Voyage returned {len(data)} embeddings for {len(chunk)} inputs"
                    )
                for (template_hash, _text), item in zip(chunk, data, strict=True):
                    out[template_hash] = _l2(np.asarray(item["embedding"], dtype=np.float32))

                index += len(chunk)
                if min_interval and index % (size * 10) == 0:
                    log.info("embedded %d/%d templates", index, len(items))
        return out
