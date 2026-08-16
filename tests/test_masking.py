from __future__ import annotations

import unittest

from rei.masking import is_low_information, normalize, template_hash


class TestMasking(unittest.TestCase):
    def test_uuid_in_message_collapses_to_one_template(self):
        """The exact pathology from the production report: because the quote ID
        sits inside the message text, every occurrence was its own '1 x' line."""
        a = normalize(
            "Renewal Flow: Failed to get quote docs for: 8afd54b3-e236-4dcf-806e-3024680a3c9b"
        )
        b = normalize(
            "Renewal Flow: Failed to get quote docs for: 6487f61d-812f-4546-9008-57185b095dca"
        )
        self.assertEqual(a, b)
        self.assertEqual(template_hash(a), template_hash(b))
        self.assertIn("<UUID>", a)

    def test_status_codes_are_preserved(self):
        """400 and 500 are different failures and must never merge."""
        a = normalize("Request failed with status code 400")
        b = normalize("Request failed with status code 500")
        self.assertNotEqual(a, b)
        self.assertIn("400", a)

    def test_amounts_are_masked_but_status_codes_are_not(self):
        a = normalize("Total Payable (1234.56) does not match Total Paid (1200.00)")
        b = normalize("Total Payable (89.10) does not match Total Paid (12.00)")
        self.assertEqual(a, b)
        self.assertNotIn("1234", a)

    def test_stripe_and_email_masking(self):
        a = normalize(
            "Could not find a default payment method for Stripe user cus_QkpGVwpmGv6jPT "
            "with email avery.dunn@example.com"
        )
        b = normalize(
            "Could not find a default payment method for Stripe user cus_ZZ12aabbCCDD99 "
            "with email someone.else@example.org"
        )
        self.assertEqual(a, b)
        self.assertIn("<STRIPE_ID>", a)
        self.assertIn("<EMAIL>", a)

    def test_slack_mailto_markup(self):
        masked = normalize("email <mailto:a.b@example.com|a.b@example.com> failed")
        self.assertEqual(masked, "email <EMAIL> failed")

    def test_repeated_placeholders_collapse(self):
        one = normalize(
            "docs missing: 8afd54b3-e236-4dcf-806e-3024680a3c9b, "
            "6487f61d-812f-4546-9008-57185b095dca, bbe5490c-d1c8-465d-b45c-c18432dc08b7"
        )
        two = normalize("docs missing: 8afd54b3-e236-4dcf-806e-3024680a3c9b")
        self.assertEqual(one, two)

    def test_long_templates_are_bounded(self):
        self.assertLessEqual(len(normalize("word " * 5000)), 1010)

    def test_generic_transport_errors_flagged_as_low_information(self):
        for text in [
            "Request failed with status code 400",
            "Request failed",
            "Bad Request",
            "Internal Server Error",
            "An unexpected error occurred",
            "error",
            "invalid",
            "response-validation",
        ]:
            with self.subTest(text=text):
                self.assertTrue(is_low_information(normalize(text)), text)

    def test_specific_errors_not_flagged(self):
        for text in [
            "Could not find a default payment method for Stripe user cus_X with email a@example.com",
            "Execution Timeout Expired. The timeout period elapsed prior to completion.",
            "totalAmount is not allowed to be an empty array",
            "Please select the option.",
        ]:
            with self.subTest(text=text):
                self.assertFalse(is_low_information(normalize(text)), text)


if __name__ == "__main__":
    unittest.main()
