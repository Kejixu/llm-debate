"""call_with_retry: backoff schedule, exhaustion, and non-retryable propagation."""

import pytest

from llm_debate.retry import call_with_retry
from llm_debate.types import CLICallError


class FlakyOp:
    """Fails with CLICallError n times, then returns a value."""

    def __init__(self, failures: int, value: str = "ok") -> None:
        self.failures = failures
        self.value = value
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise CLICallError(f"transient failure #{self.calls}")
        return self.value


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


async def test_success_first_try_no_sleep() -> None:
    sleeps = SleepRecorder()
    op = FlakyOp(failures=0)
    assert await call_with_retry(op, sleep=sleeps) == "ok"
    assert op.calls == 1
    assert sleeps.delays == []


async def test_fail_twice_then_succeed_with_exponential_backoff() -> None:
    sleeps = SleepRecorder()
    op = FlakyOp(failures=2)
    assert await call_with_retry(op, max_retries=2, base_delay_s=2.0, sleep=sleeps) == "ok"
    assert op.calls == 3
    assert sleeps.delays == [2.0, 4.0]  # base * 2**attempt


async def test_exhausted_retries_raises_last_error() -> None:
    sleeps = SleepRecorder()
    op = FlakyOp(failures=99)
    with pytest.raises(CLICallError, match="transient failure #3"):
        await call_with_retry(op, max_retries=2, sleep=sleeps)
    assert op.calls == 3  # 1 try + 2 retries, then stop
    assert sleeps.delays == [2.0, 4.0]  # no sleep after the final failure


async def test_non_cli_error_propagates_immediately() -> None:
    sleeps = SleepRecorder()

    async def buggy() -> str:
        raise ValueError("a bug, not CLI flakiness")

    with pytest.raises(ValueError):
        await call_with_retry(buggy, sleep=sleeps)
    assert sleeps.delays == []  # bugs are never retried
