from __future__ import annotations

import json
import unittest

from rei.baseline import baseline_groups
from rei.extract import extract_events

QUOTE = "8afd54b3-e236-4dcf-806e-3024680a3c9b"
OTHER = "6487f61d-812f-4546-9008-57185b095dca"
THIRD = "bbe5490c-d1c8-465d-b45c-c18432dc08b7"


def slack_message(payload: dict, ts: str = "1755300000.000100") -> dict:
    text = json.dumps(payload)
    return {
        "ts": ts,
        "text": text,
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_preformatted",
                        "elements": [{"type": "text", "text": text}],
                    }
                ],
            }
        ],
    }


class TestExtraction(unittest.TestCase):
    def test_quote_id_taken_from_named_key_not_from_noise(self):
        msg = slack_message(
            {
                "quoteId": QUOTE,
                "error": "Renewal Flow: Failed to get quote docs",
                "context": {"correlationId": OTHER, "policyId": THIRD},
            }
        )
        (event,) = extract_events([msg])
        self.assertEqual(event.quote_id, QUOTE)
        self.assertEqual(event.quote_id_confidence, 1.0)
        self.assertCountEqual(event.related_ids, [OTHER, THIRD])

    def test_baseline_attributes_every_uuid_as_a_quote(self):
        """Documents the bug this replaces, so a regression would be visible."""
        msg = slack_message(
            {
                "quoteId": QUOTE,
                "error": "Renewal Flow: Failed to get quote docs",
                "context": {"correlationId": OTHER, "policyId": THIRD},
            }
        )
        groups = baseline_groups([msg])
        (entries,) = groups.values()
        _key, quote_ids = entries[0]
        self.assertEqual(len(quote_ids), 3)  # three IDs claimed for one quote

    def test_picks_the_most_specific_message(self):
        msg = slack_message(
            {
                "message": "error",
                "error": {"message": "totalAmount is not allowed to be an empty array"},
            }
        )
        (event,) = extract_events([msg])
        self.assertEqual(event.message, "totalAmount is not allowed to be an empty array")

    def test_http_status_and_service_captured(self):
        msg = slack_message(
            {
                "error": "Request failed with status code 400",
                "statusCode": 400,
                "service": "pricing-engine",
            }
        )
        (event,) = extract_events([msg])
        self.assertEqual(event.http_status, 400)
        self.assertEqual(event.service, "pricing-engine")

    def test_serialised_payloads_are_not_chosen_as_the_message(self):
        msg = slack_message(
            {"message": '{"a": 1, "b": 2, "c": 3}', "error": "Payment authorisation declined"}
        )
        (event,) = extract_events([msg])
        self.assertEqual(event.message, "Payment authorisation declined")

    def test_non_json_text_falls_back_to_regex(self):
        msg = {"ts": "1755300000.000200", "text": 'oops "error": "Card was declined" happened'}
        (event,) = extract_events([msg])
        self.assertEqual(event.message, "Card was declined")

    def test_message_with_no_error_is_skipped(self):
        msg = {"ts": "1755300000.000300", "text": "deploy finished successfully"}
        self.assertEqual(extract_events([msg]), [])

    def test_ambiguous_quote_is_left_unattributed(self):
        msg = slack_message({"error": "Renewal failed", "a": OTHER, "b": THIRD})
        (event,) = extract_events([msg])
        self.assertIsNone(event.quote_id)
        self.assertEqual(event.quote_id_confidence, 0.0)

    def test_one_event_per_message(self):
        msg = slack_message({"quoteId": QUOTE, "error": "boom"})
        self.assertEqual(len(extract_events([msg, msg])), 2)


if __name__ == "__main__":
    unittest.main()
