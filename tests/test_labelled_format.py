"""Tests for the real #error-alerts payload shape.

Every case here was derived from an actual seven-day dump of the channel. The
original script scored 20% coverage on this format and attributed 1 quote out of
125 errors, because it only ever looked for JSON.
"""

from __future__ import annotations

import unittest

from rei.extract import extract_events, is_document_noise, labelled_pairs, parse_error_value

QUOTE = "1e7c3216-77c0-4a1b-ad64-2f0c9d5b8e31"
CORRELATION = "3f7a91c4-0b2e-4d88-9a17-5c6e2b7f4a90"


def rich_message(pairs: list[tuple[str, str]], ts: str = "1786800000.000100") -> dict:
    elements = []
    rendered = []
    for label, value in pairs:
        elements.append(
            {"type": "rich_text_section", "elements": [{"type": "text", "text": f" {label} "}]}
        )
        elements.append(
            {"type": "rich_text_preformatted", "elements": [{"type": "text", "text": value}]}
        )
        rendered.append(f" {label} ```{value}```")
    return {
        "ts": ts,
        "bot_id": "B00RENEWAL1",
        "text": "".join(rendered).strip(),
        "blocks": [{"type": "rich_text", "elements": elements}],
    }


CARD_DECLINED = [
    ("methodName", "renewalHandler"),
    ("correlationId", CORRELATION),
    (
        "error",
        "type: payment-failure\n"
        "message: Renewal Flow: Card declined: Your card was declined. (generic_decline)\n"
        'error: {"type": "payment-failure", "cause": '
        '{"type": "card-declined", "name": "StripeServiceError"}}',
    ),
    ("message", f"Renewal Flow for {QUOTE} : Renewal Flow: Card declined."),
]


class TestLabelledPairs(unittest.TestCase):
    def test_pairs_read_from_blocks(self):
        pairs = labelled_pairs(rich_message(CARD_DECLINED))
        self.assertEqual(
            [label for label, _v in pairs], ["methodName", "correlationId", "error", "message"]
        )

    def test_pairs_fall_back_to_flattened_text(self):
        """Slack renders preformatted values fenced in backticks; if blocks are
        missing we must still recover the structure."""
        msg = rich_message(CARD_DECLINED)
        del msg["blocks"]
        pairs = labelled_pairs(msg)
        self.assertEqual([label for label, _v in pairs][:2], ["methodName", "correlationId"])

    def test_error_value_subfields(self):
        parsed = parse_error_value(CARD_DECLINED[2][1])
        self.assertEqual(parsed["type"], "payment-failure")
        self.assertTrue(parsed["message"].startswith("Renewal Flow: Card declined"))
        self.assertEqual(parsed["cause_type"], "card-declined")
        self.assertEqual(parsed["cause_name"], "StripeServiceError")


class TestExtraction(unittest.TestCase):
    def test_full_event(self):
        (event,) = extract_events([rich_message(CARD_DECLINED)])
        self.assertEqual(event.service, "renewalHandler")
        self.assertEqual(event.error_type, "payment-failure")
        self.assertEqual(event.cause_type, "card-declined")
        self.assertEqual(event.correlation_id, CORRELATION)
        self.assertIn("Card declined", event.message)

    def test_correlation_id_is_never_the_quote(self):
        """The old script counted every UUID as an affected quote, which is why
        it claimed 162 quotes over a week with 125 errors."""
        (event,) = extract_events([rich_message(CARD_DECLINED)])
        self.assertEqual(event.quote_id, QUOTE)
        self.assertNotEqual(event.quote_id, CORRELATION)
        self.assertEqual(event.quote_id_confidence, 0.8)

    def test_quote_found_in_prose_variants(self):
        for phrasing in (
            f"Renewal Flow: Failed to get quote docs for quote: {QUOTE}",
            f"Occupation not found for partner-api ID: 4821 for quote:{QUOTE}",
            f"Renewal Flow for {QUOTE} : something failed",
            f"quoteId = {QUOTE}",
        ):
            with self.subTest(phrasing=phrasing):
                msg = rich_message(
                    [
                        ("methodName", "renewalHandler"),
                        ("correlationId", CORRELATION),
                        ("error", "type: unknown\nmessage: Renewal failed"),
                        ("message", phrasing),
                    ]
                )
                (event,) = extract_events([msg])
                self.assertEqual(event.quote_id, QUOTE)

    def test_generic_error_keeps_its_handler(self):
        """'Request failed with status code 400' is emitted by several handlers.
        Without methodName they are indistinguishable."""
        msg = rich_message(
            [
                ("methodName", "pricingEngineClient"),
                ("correlationId", CORRELATION),
                ("error", "type: unknown\nmessage: Request failed with status code 400"),
                ("message", f"Renewal Flow: Repricing rejected for quote: {QUOTE}"),
            ]
        )
        (event,) = extract_events([msg])
        self.assertEqual(event.service, "pricingEngineClient")
        self.assertEqual(event.http_status, 400)

    def test_two_generic_400s_from_different_handlers_do_not_merge(self):
        from rei.pipeline import build_templates

        def make(handler: str, ts: str) -> dict:
            return rich_message(
                [
                    ("methodName", handler),
                    ("correlationId", CORRELATION),
                    ("error", "type: unknown\nmessage: Request failed with status code 400"),
                    ("message", f"failed for quote: {QUOTE}"),
                ],
                ts=ts,
            )

        events = extract_events(
            [
                make("forQuote", "1786800000.000100"),
                make("pricingEngineClient", "1786800001.000200"),
            ]
        )
        templates, _counts, _examples, _by = build_templates(events)
        self.assertEqual(len(templates), 2, "different handlers must stay separate")


class TestDocumentNoise(unittest.TestCase):
    def test_endorsement_clause_is_filtered(self):
        msg = {
            "ts": "1786800002.000300",
            "bot_id": "B00DOCS0001",
            "text": (
                '"clauseText": "&lt;div class=\\"editor-content\\"&gt;&lt;h3&gt;Limitation '
                "of Liability&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;&lt;label&gt;Customer:&lt;/label&gt;"
                "&lt;span&gt;Example Trading Co&lt;/span&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;"
            ),
        }
        self.assertTrue(is_document_noise(msg))
        self.assertEqual(extract_events([msg]), [])

    def test_real_errors_are_not_filtered(self):
        self.assertFalse(is_document_noise(rich_message(CARD_DECLINED)))


class TestRealWorldQuirks(unittest.TestCase):
    """Cases taken verbatim in shape from a real seven-day dump."""

    TRUNCATED = (
        'type: partner-api message: response-validation error: { "quoteId": '
        f'"{QUOTE}", "errorType": "response-validation", "error": {{ "_original": {{'
    )

    def test_space_separated_subfields_are_parsed(self):
        """Several handlers emit sub-fields on one line. A line-anchored parser
        finds only `type`, loses the message and silently drops the event."""
        parsed = parse_error_value(self.TRUNCATED)
        self.assertEqual(parsed["type"], "partner-api")
        self.assertEqual(parsed["message"], "response-validation")

    def test_handler_survives_a_truncated_payload(self):
        msg = rich_message(
            [
                ("methodName", "processQuote"),
                ("correlationId", CORRELATION),
                ("error", self.TRUNCATED),
            ]
        )
        (event,) = extract_events([msg])
        self.assertEqual(event.service, "processQuote")
        self.assertEqual(event.error_type, "partner-api")
        self.assertEqual(event.quote_id, QUOTE)  # found inside the truncated JSON

    def test_serialised_exception_does_not_fragment_the_taxonomy(self):
        """An AxiosError blob carries per-request timeouts and URLs. Left in, it
        turned one CRM-sync failure into eight singleton report lines."""
        from rei.pipeline import build_templates

        messages = [
            rich_message(
                [
                    ("methodName", "fullUpdate"),
                    ("correlationId", CORRELATION),
                    (
                        "error",
                        "type: unknown\nmessage: Renewal Flow: Can't update CRM "
                        'contacts: {"message":"Request failed with status code 400",'
                        f'"name":"AxiosError","config":{{"timeout":{1000 + i},'
                        f'"url":"https://api.CRM.io/contacts/{i}"}}}}',
                    ),
                ],
                ts=f"178680000{i}.000100",
            )
            for i in range(6)
        ]
        events = extract_events(messages)
        self.assertEqual(len(events), 6)
        templates, _counts, _examples, _by = build_templates(events)
        self.assertEqual(
            len(templates), 1, f"expected one template, got {list(templates.values())}"
        )

    def test_truncated_block_falls_back_to_the_text_rendering(self):
        """When Slack truncates a long payload it drops the trailing
        preformatted element from `blocks` while the opening survives in `text`.
        Trusting `blocks` unconditionally lost the whole `error` field on every
        partner-api validation failure — the messages with the most content."""
        msg = rich_message(
            [
                ("methodName", "processQuote"),
                ("correlationId", CORRELATION),
                ("error", self.TRUNCATED),
            ]
        )
        # Simulate the truncation: blocks keep only the first two pairs.
        msg["blocks"][0]["elements"] = msg["blocks"][0]["elements"][:4]
        msg["text"] = msg["text"].rsplit("```", 1)[0]  # unterminated final fence

        pairs = labelled_pairs(msg)
        self.assertIn("error", [label for label, _v in pairs])
        (event,) = extract_events([msg])
        self.assertEqual(event.service, "processQuote")
        self.assertEqual(event.error_type, "partner-api")
        self.assertEqual(event.message, "response-validation")

    def test_stripe_payment_method_ids_are_masked(self):
        from rei.masking import normalize

        a = normalize("Renewal Flow: Failed to create pm_1QFdHXAeWz5UJAB1R8Y9oZE1 setup intent")
        b = normalize("Renewal Flow: Failed to create pm_9ZZaaBBccDD11eeFF22gg setup intent")
        self.assertEqual(a, b)
        self.assertIn("<STRIPE_ID>", a)

    def test_system_events_and_answer_payloads_are_not_errors(self):
        from rei.extract import is_reportable

        joined = {"ts": "1.0", "subtype": "channel_join", "text": "<@U04> has joined the channel"}
        answers = {
            "ts": "2.0",
            "bot_id": "B1",
            "text": '"id": "abc", "answer": "1" }, { "id": "def"',
        }
        self.assertFalse(is_reportable(joined))
        self.assertFalse(is_reportable(answers))
        self.assertTrue(is_reportable(rich_message(CARD_DECLINED)))


if __name__ == "__main__":
    unittest.main()
