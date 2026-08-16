"""Generate a labelled fixture set in the real #error-alerts message format.

The renewal service posts Slack rich-text blocks as alternating label /
preformatted-value pairs, which render in the fallback text as:

    methodName ```renewalHandler``` correlationId ```3f7a…```
    error ```type: payment-failure
             message: Renewal Flow: Card declined: Your card was declined.```
    message ```Renewal Flow for 1e7c… : Renewal Flow: Card declined…```

Every pathology reproduced here was observed in a real seven-day dump:

  - the quote ID lives in prose ("for quote: <uuid>"), never in a `quoteId` key
  - `correlationId` is a UUID that is *not* a quote, and the old script counted
    it as one — over a week it claimed 162 affected quotes on 125 errors
  - "Request failed with status code 400" and "response-validation" carry no
    information at all without `methodName`, and together account for 40% of
    all errors
  - policy wordings and endorsement clauses are posted to the same channel and
    are not errors

Ground truth is `true_cluster`. Regenerate with:

    python scripts/generate_fixtures.py
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

random.seed(20260816)

ROOT = Path(__file__).resolve().parents[1]
BASE_TS = datetime(2026, 8, 15, 22, 0, tzinfo=UTC)

EMAILS = [
    "avery.dunn@example.com",
    "priya.n@example.org",
    "dan.oleary@example.net",
    "s.whitfield@example.com",
]


def _uuid() -> str:
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def _cus() -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "cus_" + "".join(random.choice(alphabet) for _ in range(14))


def _rich_text(pairs: list[tuple[str, str]]) -> tuple[dict, str]:
    """Build the rich_text block and its flattened text rendering."""
    elements: list[dict] = []
    rendered: list[str] = []
    for label, value in pairs:
        elements.append(
            {"type": "rich_text_section", "elements": [{"type": "text", "text": f" {label} "}]}
        )
        elements.append(
            {"type": "rich_text_preformatted", "elements": [{"type": "text", "text": value}]}
        )
        rendered.append(f" {label} ```{value}```")
    return {"type": "rich_text", "elements": elements}, "".join(rendered).strip()


def _error_message(
    ts: datetime,
    *,
    method: str,
    error_type: str,
    message: str,
    quote_id: str,
    true_cluster: str,
    cause: dict | None = None,
    outer_message: str | None = None,
) -> dict:
    # The nested `error:` blob is what the old script's regex was matching on —
    # it scraped "message": "..." out of here and ignored everything else.
    nested: dict = {"type": error_type, "message": message}
    if cause:
        nested["cause"] = cause
    error_value = f"type: {error_type}\nmessage: {message}\nerror: {json.dumps(nested)}"

    pairs = [
        ("methodName", method),
        ("correlationId", _uuid()),  # NOT the quote — must never be counted as one
        ("error", error_value),
        ("message", outer_message or f"Renewal Flow for {quote_id} : {message}"),
    ]
    block, text = _rich_text(pairs)
    return {
        "type": "message",
        "ts": f"{ts.timestamp():.6f}",
        "bot_id": "B00RENEWAL1",
        "text": text,
        "blocks": [block],
        "_true_cluster": true_cluster,
        "_true_quote": quote_id,
    }


def _document_noise(ts: datetime) -> dict:
    """A contract clause document. Shares the channel; is not an error."""
    body = (
        '"clauseText": "&lt;div id=\\"1042\\" class=\\"editor-content\\"&gt;'
        "&lt;h3&gt;Limitation of Liability&lt;/h3&gt;"
        '&lt;ul style=\\"font-family: Arial;\\"&gt;&lt;li style=\\"font-size: 11px;\\"&gt;'
        "&lt;label&gt;Customer:&lt;/label&gt;&lt;span&gt;Example Trading Co"
        "&lt;/span&gt;&lt;/li&gt;&lt;li&gt;&lt;label&gt;Term:&lt;/label&gt;"
        "&lt;span&gt;4.00pm on 30/08/2026 to 4.00pm on 30/08/2027&lt;/span&gt;&lt;/li&gt;"
        "&lt;/ul&gt;&lt;/div&gt;"
    )
    return {
        "type": "message",
        "ts": f"{ts.timestamp():.6f}",
        "bot_id": "B00DOCS0001",
        "text": body,
        "_true_cluster": None,
        "_true_quote": None,
    }


def build() -> list[dict]:
    messages: list[dict] = []
    clock = BASE_TS

    def tick() -> datetime:
        nonlocal clock
        clock += timedelta(seconds=random.randint(20, 600))
        return clock

    # 1. Unanswered underwriting question. Highest-volume real error.
    for _ in range(51):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="quoteValidationHandler",
                error_type="validation",
                message="Please select the option.",
                quote_id=quote,
                true_cluster="unanswered_question",
                outer_message=f"Renewal Flow: Validation failed for quote: {quote}",
            )
        )

    # 2 & 3. "response-validation" — meaningless alone, emitted by two unrelated
    # adapters. Only methodName separates them.
    for _ in range(18):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="partnerQuoteAdapter",
                error_type="response-validation",
                message="response-validation",
                quote_id=quote,
                true_cluster="partner_api_validation",
                outer_message=(
                    f"Renewal Flow: Can't fetch quote from partner-api-validation "
                    f"for quote: {quote}"
                ),
            )
        )
    for _ in range(10):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="resellerPortalSync",
                error_type="response-validation",
                message="response-validation",
                quote_id=quote,
                true_cluster="empty_premium_array",
                outer_message=(
                    f"totalAmount is not allowed to be an empty array for quote: {quote}"
                ),
            )
        )

    # 4 & 5. Bare HTTP 400 from two different handlers.
    for _ in range(14):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="forQuote",
                error_type="unknown",
                message="Request failed with status code 400",
                quote_id=quote,
                true_cluster="http400_quote_docs",
                outer_message=f"Renewal Flow: Failed to get quote docs for quote: {quote}",
            )
        )
    for _ in range(8):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="pricingEngineClient",
                error_type="unknown",
                message="Request failed with status code 400",
                quote_id=quote,
                true_cluster="http400_pricing",
                outer_message=f"Renewal Flow: Repricing rejected for quote: {quote}",
            )
        )

    # 6. SQL timeout, two wordings from two services.
    for _ in range(13):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="renewalHandler",
                error_type="unknown",
                message=(
                    "Execution Timeout Expired.  The timeout period elapsed prior to "
                    "completion of the operation or the server is not responding."
                ),
                quote_id=quote,
                true_cluster="db_timeout",
            )
        )
    for _ in range(5):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="renewalWorker",
                error_type="unknown",
                message=(
                    "Timeout expired. The timeout period elapsed prior to completion "
                    "of the operation."
                ),
                quote_id=quote,
                true_cluster="db_timeout",
            )
        )

    # 7. Card declined, with a nested Stripe cause.
    for _ in range(9):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="renewalHandler",
                error_type="payment-failure",
                message="Renewal Flow: Card declined: Your card was declined. (generic_decline)",
                quote_id=quote,
                true_cluster="card_declined",
                cause={"type": "card-declined", "name": "StripeServiceError"},
            )
        )

    # 8. Subscription setup failure, same Stripe cause, different operation.
    for _ in range(4):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="forQuote",
                error_type="unknown",
                message="Renewal Flow: Failed to setup a monthly subscription",
                quote_id=quote,
                true_cluster="subscription_setup_failed",
                cause={"type": "card-declined", "name": "StripeServiceError"},
            )
        )

    # 9. No default payment method on the Stripe customer.
    for _ in range(6):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="collectPayment",
                error_type="payment-failure",
                message=(
                    f"Could not find a default payment method for Stripe user {_cus()} "
                    f"with email {random.choice(EMAILS)}"
                ),
                quote_id=quote,
                true_cluster="payment_no_default_method",
            )
        )

    # 10. Reference-data gap. Partner-API occupation ID is not a UUID.
    for _ in range(7):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="renewalHandler",
                error_type="unknown",
                message=(
                    f"Renewal Flow: Can't save renewal records - Occupation not found "
                    f"for partner-api ID: {random.randint(1000, 9999)} for quote:{quote}"
                ),
                quote_id=quote,
                true_cluster="occupation_not_found",
            )
        )

    # 11. CRM sync failure.
    for _ in range(5):
        quote = _uuid()
        messages.append(
            _error_message(
                tick(),
                method="crmSyncHandler",
                error_type="unknown",
                message="Renewal Flow: Can't update CRM contacts",
                quote_id=quote,
                true_cluster="crm_sync",
            )
        )

    # 12. Total payable vs total paid mismatch.
    for _ in range(6):
        quote = _uuid()
        payable = round(random.uniform(400, 2400), 2)
        messages.append(
            _error_message(
                tick(),
                method="renewalHandler",
                error_type="unknown",
                message=(
                    f"Renewal Flow: Total Payable ({payable}) does not match "
                    f"Total Paid ({round(payable - random.uniform(1, 90), 2)})"
                ),
                quote_id=quote,
                true_cluster="total_mismatch",
            )
        )

    # Endorsement clause documents. Not errors; must be filtered, not "missed".
    for _ in range(40):
        messages.append(_document_noise(tick()))

    random.shuffle(messages)
    return messages


def main() -> None:
    messages = build()
    labelled = {
        m["ts"]: {"true_cluster": m["_true_cluster"], "true_quote": m["_true_quote"]}
        for m in messages
        if m["_true_cluster"]
    }
    clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]

    fixtures = ROOT / "fixtures"
    fixtures.mkdir(exist_ok=True)
    (fixtures / "slack_messages.json").write_text(json.dumps({"messages": clean}, indent=2))
    (fixtures / "labelled.json").write_text(json.dumps(labelled, indent=2))

    counts: dict[str, int] = {}
    for entry in labelled.values():
        counts[entry["true_cluster"]] = counts.get(entry["true_cluster"], 0) + 1
    noise = len(clean) - len(labelled)
    print(f"wrote {len(clean)} messages: {len(labelled)} errors across {len(counts)} true clusters")
    print(f"      plus {noise} endorsement documents that must be filtered out")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}  {name}")


if __name__ == "__main__":
    main()
