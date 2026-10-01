"""Unit tests for shared provider error classification helpers."""

import unittest

from src.provider_errors import is_daily_quota_error, is_quota_error


class ProviderError(Exception):
    """Small provider-like exception used to exercise public error attributes."""

    def __init__(self, message: str, *, code=None, status_code=None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class TestProviderErrors(unittest.TestCase):
    def test_numeric_code_429_is_quota_but_not_daily(self):
        error = ProviderError("Too many requests", code=429)

        self.assertTrue(is_quota_error(error))
        self.assertFalse(is_daily_quota_error(error))

    def test_numeric_status_code_429_is_quota_but_not_daily(self):
        error = ProviderError("Rate limited", status_code=429)

        self.assertTrue(is_quota_error(error))
        self.assertFalse(is_daily_quota_error(error))

    def test_resource_exhausted_is_quota_but_not_daily(self):
        error = ProviderError("RESOURCE_EXHAUSTED")

        self.assertTrue(is_quota_error(error))
        self.assertFalse(is_daily_quota_error(error))

    def test_quota_exceeded_is_quota_but_not_daily(self):
        error = ProviderError("Quota exceeded for this request")

        self.assertTrue(is_quota_error(error))
        self.assertFalse(is_daily_quota_error(error))

    def test_known_daily_quota_id_is_daily_quota(self):
        error = ProviderError(
            "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            code=429,
        )

        self.assertTrue(is_quota_error(error))
        self.assertTrue(is_daily_quota_error(error))

    def test_daily_quota_wording_is_daily_quota(self):
        error = ProviderError("Daily quota exceeded", code=429)

        self.assertTrue(is_quota_error(error))
        self.assertTrue(is_daily_quota_error(error))

    def test_ordinary_503_is_not_quota(self):
        error = ProviderError("Service unavailable", code=503)

        self.assertFalse(is_quota_error(error))
        self.assertFalse(is_daily_quota_error(error))

    def test_401_is_not_quota(self):
        error = ProviderError("Unauthorized", status_code=401)

        self.assertFalse(is_quota_error(error))
        self.assertFalse(is_daily_quota_error(error))

    def test_unrelated_value_error_is_not_quota(self):
        error = ValueError("Embedding response was malformed")

        self.assertFalse(is_quota_error(error))
        self.assertFalse(is_daily_quota_error(error))

    def test_exception_chain_is_inspected(self):
        provider_error = ProviderError(
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            code=429,
        )
        try:
            raise provider_error
        except ProviderError as cause:
            try:
                raise RuntimeError("Provider request failed") from cause
            except RuntimeError as wrapped:
                error = wrapped

        self.assertTrue(is_quota_error(error))
        self.assertTrue(is_daily_quota_error(error))


if __name__ == "__main__":
    unittest.main()
