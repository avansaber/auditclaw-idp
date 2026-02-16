"""Tests for IDP compliance check modules (16 tests with mocked APIs).

Covers Google Workspace (4 checks x 2 tests) and Okta (4 checks x 2 tests).
All tests use unittest.mock; no real API credentials needed.
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_google_service(users):
    """Create a mock Google Admin SDK service that returns the given users."""
    service = MagicMock()
    service.users.return_value.list.return_value.execute.return_value = {
        "users": users,
        "nextPageToken": None,
    }
    return service


def _make_google_user(email, enrolled_2sv=True, enforced_2sv=True, is_admin=False,
                       last_login_days_ago=5, suspended=False, password_strength="STRONG"):
    """Build a Google Workspace user dict for testing."""
    last_login = (datetime.now(timezone.utc) - timedelta(days=last_login_days_ago)).isoformat()
    return {
        "primaryEmail": email,
        "isEnrolledIn2Sv": enrolled_2sv,
        "isEnforcedIn2Sv": enforced_2sv,
        "isAdmin": is_admin,
        "lastLoginTime": last_login,
        "suspended": suspended,
        "passwordStrength": password_strength,
    }


# ---------------------------------------------------------------------------
# Google MFA Tests (2)
# ---------------------------------------------------------------------------

class TestGoogleMFAChecks:
    def test_all_users_enrolled_and_enforced(self):
        """All active users have 2SV enrolled + enforced -> pass."""
        from checks.google_mfa import run_google_mfa_checks
        users = [
            _make_google_user("alice@example.com", enrolled_2sv=True, enforced_2sv=True),
            _make_google_user("bob@example.com", enrolled_2sv=True, enforced_2sv=True),
        ]
        service = _mock_google_service(users)

        result = run_google_mfa_checks(service)
        assert result["check"] == "google_mfa"
        assert result["provider"] == "google_workspace"
        assert result["status"] == "pass"
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_some_users_not_enrolled(self):
        """Some users missing 2SV enrollment -> fail."""
        from checks.google_mfa import run_google_mfa_checks
        users = [
            _make_google_user("alice@example.com", enrolled_2sv=True, enforced_2sv=True),
            _make_google_user("bob@example.com", enrolled_2sv=False, enforced_2sv=False),
            _make_google_user("carol@example.com", enrolled_2sv=True, enforced_2sv=False),
        ]
        service = _mock_google_service(users)

        result = run_google_mfa_checks(service)
        assert result["status"] == "fail"
        assert result["passed"] == 1
        assert result["failed"] == 2


# ---------------------------------------------------------------------------
# Google Admins Tests (2)
# ---------------------------------------------------------------------------

class TestGoogleAdminChecks:
    def test_two_admins_with_2sv(self):
        """2 super admins, both with 2SV -> pass."""
        from checks.google_admins import run_google_admin_checks
        users = [
            _make_google_user("admin1@example.com", is_admin=True, enrolled_2sv=True, enforced_2sv=True),
            _make_google_user("admin2@example.com", is_admin=True, enrolled_2sv=True, enforced_2sv=True),
            _make_google_user("user1@example.com", is_admin=False),
        ]
        service = _mock_google_service(users)

        result = run_google_admin_checks(service)
        assert result["check"] == "google_admins"
        assert result["status"] == "pass"
        # 1 count finding (pass) + 2 admin MFA findings (pass) = 3 total, 3 passed
        assert result["passed"] == 3
        assert result["failed"] == 0

    def test_too_many_admins(self):
        """6 super admins (too many) -> fail on count."""
        from checks.google_admins import run_google_admin_checks
        users = [
            _make_google_user(f"admin{i}@example.com", is_admin=True, enrolled_2sv=True, enforced_2sv=True)
            for i in range(6)
        ]
        service = _mock_google_service(users)

        result = run_google_admin_checks(service)
        assert result["status"] == "fail"
        # Count check fails, but all 6 admin MFA checks pass
        assert result["failed"] >= 1
        # Verify the count finding specifically
        count_finding = [f for f in result["findings"] if "count" in f["resource"]][0]
        assert count_finding["status"] == "fail"
        assert "6" in count_finding["detail"]


# ---------------------------------------------------------------------------
# Google Inactive Tests (2)
# ---------------------------------------------------------------------------

class TestGoogleInactiveChecks:
    def test_all_logins_recent(self):
        """All users logged in within 90 days -> pass."""
        from checks.google_inactive import run_google_inactive_checks
        users = [
            _make_google_user("alice@example.com", last_login_days_ago=10),
            _make_google_user("bob@example.com", last_login_days_ago=45),
        ]
        service = _mock_google_service(users)

        result = run_google_inactive_checks(service)
        assert result["check"] == "google_inactive"
        assert result["status"] == "pass"
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_user_inactive_120_days(self):
        """User inactive for 120 days -> fail."""
        from checks.google_inactive import run_google_inactive_checks
        users = [
            _make_google_user("alice@example.com", last_login_days_ago=10),
            _make_google_user("stale@example.com", last_login_days_ago=120),
        ]
        service = _mock_google_service(users)

        result = run_google_inactive_checks(service)
        assert result["status"] == "fail"
        assert result["passed"] == 1
        assert result["failed"] == 1
        stale_finding = [f for f in result["findings"] if "stale@" in f["resource"]][0]
        assert stale_finding["status"] == "fail"
        assert "120" in stale_finding["detail"]


# ---------------------------------------------------------------------------
# Google Passwords Tests (2)
# ---------------------------------------------------------------------------

class TestGooglePasswordChecks:
    def test_all_strong_passwords(self):
        """All users have STRONG passwords -> pass."""
        from checks.google_passwords import run_google_password_checks
        users = [
            _make_google_user("alice@example.com", password_strength="STRONG"),
            _make_google_user("bob@example.com", password_strength="STRONG"),
        ]
        service = _mock_google_service(users)

        result = run_google_password_checks(service)
        assert result["check"] == "google_passwords"
        assert result["status"] == "pass"
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_some_weak_passwords(self):
        """Some users have weak passwords -> fail."""
        from checks.google_passwords import run_google_password_checks
        users = [
            _make_google_user("alice@example.com", password_strength="STRONG"),
            _make_google_user("bob@example.com", password_strength="WEAK"),
            _make_google_user("carol@example.com", password_strength="UNKNOWN"),
        ]
        service = _mock_google_service(users)

        result = run_google_password_checks(service)
        assert result["status"] == "fail"
        assert result["passed"] == 1
        assert result["failed"] == 2


# ---------------------------------------------------------------------------
# Okta MFA Tests (2)
# ---------------------------------------------------------------------------

class TestOktaMFAChecks:
    @patch("checks.okta_mfa.OktaClient", create=True)
    def test_all_users_have_factors(self, _mock_cls):
        """All active users have MFA factors -> pass."""
        from checks.okta_mfa import _run_okta_mfa_checks_async
        import asyncio

        # Build mock users
        user1 = MagicMock()
        user1.id = "u1"
        user1.profile.login = "alice@example.com"

        user2 = MagicMock()
        user2.id = "u2"
        user2.profile.login = "bob@example.com"

        # Build mock factors
        factor1 = MagicMock()
        factor1.status = "ACTIVE"

        factor2 = MagicMock()
        factor2.status = "ACTIVE"

        # Mock the OktaClient
        mock_client = MagicMock()

        async def mock_list_users(params):
            return [user1, user2], None, None

        async def mock_list_factors(user_id):
            return [factor1, factor2], None, None

        mock_client.list_users = mock_list_users
        mock_client.list_factors = mock_list_factors

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_mfa.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_mfa_checks_async(config))

        assert result["check"] == "okta_mfa"
        assert result["provider"] == "okta"
        assert result["status"] == "pass"
        assert result["passed"] == 2
        assert result["failed"] == 0

    @patch("checks.okta_mfa.OktaClient", create=True)
    def test_user_with_zero_factors(self, _mock_cls):
        """User with no MFA factors -> fail."""
        from checks.okta_mfa import _run_okta_mfa_checks_async
        import asyncio

        user1 = MagicMock()
        user1.id = "u1"
        user1.profile.login = "alice@example.com"

        user2 = MagicMock()
        user2.id = "u2"
        user2.profile.login = "nofactor@example.com"

        factor1 = MagicMock()
        factor1.status = "ACTIVE"

        mock_client = MagicMock()

        async def mock_list_users(params):
            return [user1, user2], None, None

        async def mock_list_factors(user_id):
            if user_id == "u1":
                return [factor1], None, None
            return [], None, None

        mock_client.list_users = mock_list_users
        mock_client.list_factors = mock_list_factors

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_mfa.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_mfa_checks_async(config))

        assert result["status"] == "fail"
        assert result["passed"] == 1
        assert result["failed"] == 1
        nofactor = [f for f in result["findings"] if "nofactor@" in f["resource"]][0]
        assert nofactor["status"] == "fail"


# ---------------------------------------------------------------------------
# Okta Passwords Tests (2)
# ---------------------------------------------------------------------------

class TestOktaPasswordChecks:
    @patch("checks.okta_passwords.OktaClient", create=True)
    def test_policy_meets_thresholds(self, _mock_cls):
        """Password policy meets all compliance thresholds -> pass."""
        from checks.okta_passwords import _run_okta_password_checks_async
        import asyncio

        policy = MagicMock()
        policy.name = "Default Policy"
        policy.settings.password.complexity.min_length = 14
        policy.settings.password.age.history_count = 6
        policy.settings.password.age.max_age_days = 60
        policy.settings.password.lockout.max_attempts = 3

        mock_client = MagicMock()

        async def mock_list_policies(params):
            return [policy], None, None

        mock_client.list_policies = mock_list_policies

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_passwords.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_password_checks_async(config))

        assert result["check"] == "okta_passwords"
        assert result["status"] == "pass"
        assert result["passed"] == 4
        assert result["failed"] == 0

    @patch("checks.okta_passwords.OktaClient", create=True)
    def test_min_length_too_short(self, _mock_cls):
        """Password minLength=8 (too short) -> fail."""
        from checks.okta_passwords import _run_okta_password_checks_async
        import asyncio

        policy = MagicMock()
        policy.name = "Weak Policy"
        policy.settings.password.complexity.min_length = 8
        policy.settings.password.age.history_count = 3
        policy.settings.password.age.max_age_days = 180
        policy.settings.password.lockout.max_attempts = 10

        mock_client = MagicMock()

        async def mock_list_policies(params):
            return [policy], None, None

        mock_client.list_policies = mock_list_policies

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_passwords.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_password_checks_async(config))

        assert result["status"] == "fail"
        # All 4 sub-checks fail: min_length=8, history=3, max_attempts=10, max_age=180
        assert result["failed"] == 4
        assert result["passed"] == 0


# ---------------------------------------------------------------------------
# Okta Inactive Tests (2)
# ---------------------------------------------------------------------------

class TestOktaInactiveChecks:
    @patch("checks.okta_inactive.OktaClient", create=True)
    def test_all_logins_recent(self, _mock_cls):
        """All users logged in within 90 days -> pass."""
        from checks.okta_inactive import _run_okta_inactive_checks_async
        import asyncio

        user1 = MagicMock()
        user1.id = "u1"
        user1.profile.login = "alice@example.com"
        user1.last_login = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

        user2 = MagicMock()
        user2.id = "u2"
        user2.profile.login = "bob@example.com"
        user2.last_login = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        mock_client = MagicMock()

        async def mock_list_users(params):
            return [user1, user2], None, None

        mock_client.list_users = mock_list_users

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_inactive.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_inactive_checks_async(config))

        assert result["check"] == "okta_inactive"
        assert result["status"] == "pass"
        assert result["passed"] == 2
        assert result["failed"] == 0

    @patch("checks.okta_inactive.OktaClient", create=True)
    def test_user_inactive_120_days(self, _mock_cls):
        """User inactive 120 days -> fail."""
        from checks.okta_inactive import _run_okta_inactive_checks_async
        import asyncio

        user1 = MagicMock()
        user1.id = "u1"
        user1.profile.login = "active@example.com"
        user1.last_login = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

        user2 = MagicMock()
        user2.id = "u2"
        user2.profile.login = "stale@example.com"
        user2.last_login = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()

        mock_client = MagicMock()

        async def mock_list_users(params):
            return [user1, user2], None, None

        mock_client.list_users = mock_list_users

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_inactive.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_inactive_checks_async(config))

        assert result["status"] == "fail"
        assert result["passed"] == 1
        assert result["failed"] == 1
        stale_finding = [f for f in result["findings"] if "stale@" in f["resource"]][0]
        assert stale_finding["status"] == "fail"


# ---------------------------------------------------------------------------
# Okta Sessions Tests (2)
# ---------------------------------------------------------------------------

class TestOktaSessionChecks:
    @patch("checks.okta_sessions.OktaClient", create=True)
    def test_mfa_required_bounded_session(self, _mock_cls):
        """MFA required, session bounded -> pass."""
        from checks.okta_sessions import _run_okta_session_checks_async
        import asyncio

        policy = MagicMock()
        policy.name = "Global Session"
        policy.settings.maxSessionLifetimeMinutes = 480   # 8 hours
        policy.settings.maxSessionIdleMinutes = 30        # 30 min
        policy.settings.useMfaForFactorEnrollment = True
        policy.conditions.authContext.authType = "MFA"

        mock_client = MagicMock()

        async def mock_list_policies(params):
            return [policy], None, None

        mock_client.list_policies = mock_list_policies

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_sessions.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_session_checks_async(config))

        assert result["check"] == "okta_sessions"
        assert result["status"] == "pass"
        assert result["passed"] == 3
        assert result["failed"] == 0

    @patch("checks.okta_sessions.OktaClient", create=True)
    def test_no_mfa_unbounded_session(self, _mock_cls):
        """No MFA, unbounded session -> fail."""
        from checks.okta_sessions import _run_okta_session_checks_async
        import asyncio

        policy = MagicMock()
        policy.name = "Weak Session"
        policy.settings.maxSessionLifetimeMinutes = 1440  # 24 hours
        policy.settings.maxSessionIdleMinutes = 120       # 2 hours
        policy.settings.useMfaForFactorEnrollment = False
        policy.conditions.authContext.authType = "PASSWORD"

        mock_client = MagicMock()

        async def mock_list_policies(params):
            return [policy], None, None

        mock_client.list_policies = mock_list_policies

        config = {"orgUrl": "https://test.okta.com", "token": "test-token"}

        with patch("checks.okta_sessions.OktaClient", return_value=mock_client):
            result = asyncio.run(_run_okta_session_checks_async(config))

        assert result["status"] == "fail"
        # MFA fail, lifetime fail (1440 > 720), idle fail (120 > 60) = 3 fails
        assert result["failed"] == 3
        assert result["passed"] == 0


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------

class TestCheckRegistry:
    def test_all_checks_registered(self):
        """Verify all 8 checks are in the ALL_CHECKS registry."""
        from checks import ALL_CHECKS, GOOGLE_CHECKS, OKTA_CHECKS
        expected = {
            "google_mfa", "google_admins", "google_inactive", "google_passwords",
            "okta_mfa", "okta_passwords", "okta_inactive", "okta_sessions",
        }
        assert set(ALL_CHECKS.keys()) == expected
        assert len(ALL_CHECKS) == 8
        assert len(GOOGLE_CHECKS) == 4
        assert len(OKTA_CHECKS) == 4

    def test_google_okta_partitions(self):
        """Google and Okta check sets are disjoint and cover all checks."""
        from checks import ALL_CHECKS, GOOGLE_CHECKS, OKTA_CHECKS
        google_keys = set(GOOGLE_CHECKS.keys())
        okta_keys = set(OKTA_CHECKS.keys())
        assert google_keys & okta_keys == set()
        assert google_keys | okta_keys == set(ALL_CHECKS.keys())
