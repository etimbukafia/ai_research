import pytest

from enterprise_agent_lab.rate_limit import RequestRateLimiter


def test_default_rate_limit_has_margin_below_fifteen_rpm() -> None:
    limiter = RequestRateLimiter()

    assert limiter.interval_seconds == pytest.approx(5.0)


def test_rate_limit_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RequestRateLimiter(rpm=0)
    with pytest.raises(ValueError):
        RequestRateLimiter(safety_factor=1.1)
