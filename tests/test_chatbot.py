import unittest
from unittest.mock import patch

from src import fintalkbot


class FinTalkBotTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "risk_label": "High",
            "expense_trend": "Deteriorating",
            "savings_trend": "Deteriorating",
            "avg_income": 50000,
            "expense_ratio_mean": 0.92,
            "neg_savings_freq": 0.50,
            "mc_summary": "Worst INR 81371, median INR 141694, best INR 231093",
        }

    def test_build_context_contains_full_signals(self):
        context = fintalkbot.build_context(self.profile)
        self.assertIn("Risk Level: High", context)
        self.assertIn("Expense Trend: Deteriorating", context)
        self.assertIn("Savings Trend: Deteriorating", context)
        self.assertIn("Investment Simulation", context)
        self.assertIn("231093", context)

    def test_no_key_uses_fallback_mode(self):
        response, mode, source = fintalkbot.get_financial_advice(
            "Should I invest now?",
            self.profile,
            provider="OpenAI",
            api_key="",
        )
        self.assertEqual(mode, "Fallback")
        self.assertEqual(source, "none")
        self.assertIn("high-risk", response.lower())

    def test_provider_route_success(self):
        def fake_provider(question, context, api_key, model, chat_history):
            self.assertEqual(api_key, "test-key")
            self.assertIn("User Financial Profile", context)
            self.assertEqual(model, "gpt-4o-mini")
            return "AI response"

        with patch.dict(fintalkbot.PROVIDER_MAP, {"OpenAI": fake_provider}, clear=False):
            response, mode, source = fintalkbot.get_financial_advice(
                "How can I improve?",
                self.profile,
                provider="OpenAI",
                api_key="test-key",
                model="gpt-4o-mini",
                chat_history=[{"role": "user", "content": "Older question"}],
            )

        self.assertEqual(mode, "AI")
        self.assertEqual(source, "user")
        self.assertEqual(response, "AI response")

    def test_invalid_key_error_gets_classified(self):
        def fake_provider(*args, **kwargs):
            raise Exception("invalid api key")

        with patch.dict(fintalkbot.PROVIDER_MAP, {"OpenAI": fake_provider}, clear=False):
            response, mode, source = fintalkbot.get_financial_advice(
                "How can I improve?",
                self.profile,
                provider="OpenAI",
                api_key="bad-key",
                model="gpt-4o-mini",
            )

        self.assertEqual(mode, "Fallback")
        self.assertEqual(source, "user")
        self.assertIn("authentication failed", response.lower())

    def test_quota_error_gets_classified(self):
        def fake_provider(*args, **kwargs):
            raise Exception("quota exceeded")

        with patch.dict(fintalkbot.PROVIDER_MAP, {"OpenAI": fake_provider}, clear=False):
            response, mode, source = fintalkbot.get_financial_advice(
                "How can I improve?",
                self.profile,
                provider="OpenAI",
                api_key="key",
                model="gpt-4o-mini",
            )

        self.assertEqual(mode, "Fallback")
        self.assertEqual(source, "user")
        self.assertIn("quota", response.lower())

    def test_memory_window_last_four_messages(self):
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        messages = fintalkbot._openai_style_messages("final", "ctx", history)
        contents = [m["content"] for m in messages]
        joined = "|".join(contents)
        self.assertNotIn("u1", joined)
        self.assertNotIn("a1", joined)
        self.assertIn("u2", joined)
        self.assertIn("a2", joined)
        self.assertIn("u3", joined)
        self.assertIn("a3", joined)

    def test_test_api_key_success(self):
        def fake_provider(*args, **kwargs):
            return "Connection OK"

        with patch.dict(fintalkbot.PROVIDER_MAP, {"OpenAI": fake_provider}, clear=False):
            ok, message, source = fintalkbot.test_api_key("OpenAI", "good-key", "gpt-4o-mini")

        self.assertTrue(ok)
        self.assertEqual(source, "user")
        self.assertIn("successful", message.lower())

    @patch.dict("os.environ", {"SYSTEM_GROQ_API_KEY": "system-key"}, clear=True)
    def test_resolve_api_key_uses_system_when_user_missing(self):
        key, source = fintalkbot.resolve_api_key("", "Groq")
        self.assertEqual(key, "system-key")
        self.assertEqual(source, "system")

    @patch.dict("os.environ", {"SYSTEM_GROQ_API_KEY": "system-key"}, clear=True)
    def test_resolve_api_key_prefers_user_over_system(self):
        key, source = fintalkbot.resolve_api_key("user-key", "Groq")
        self.assertEqual(key, "user-key")
        self.assertEqual(source, "user")


if __name__ == "__main__":
    unittest.main()
