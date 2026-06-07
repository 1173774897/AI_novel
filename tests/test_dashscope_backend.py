"""DashScope 重试/限流工具测试。"""

import pytest

from src.imagegen.dashscope_backend import _retry_delay


class _FakeHeaders:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def get(self, key: str, default=None):
        return self._values.get(key, default)


class _FakeResp:
    def __init__(self, retry_after: str | None = None):
        self.headers = _FakeHeaders(
            {"Retry-After": retry_after} if retry_after else {}
        )


@pytest.mark.signature
def test_retry_delay_exponential():
    assert _retry_delay(0) == 5.0
    assert _retry_delay(1) == 10.0
    assert _retry_delay(4) == 60.0
    assert _retry_delay(10) == 60.0


@pytest.mark.signature
def test_retry_delay_honors_retry_after():
    resp = _FakeResp("12")
    assert _retry_delay(0, resp) == 12.0
