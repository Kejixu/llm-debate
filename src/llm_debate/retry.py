"""call_with_retry — the ONE wrapper every CLI call goes through.

Review decision 5A: timeout, backoff, and failure classification live here
once, not copy-pasted per adapter. Retries only CLICallError (transient CLI
trouble: timeouts, rate limits, bad exits, malformed output); anything else
is a bug and propagates immediately.
"""

import asyncio
from collections.abc import Awaitable, Callable

from llm_debate.types import CLICallError


async def call_with_retry[T](
    op: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    base_delay_s: float = 2.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run op; on CLICallError retry up to max_retries with exponential backoff.

    Delays are base_delay_s * 2**attempt (2s, 4s, ...). `sleep` is injectable
    so tests assert the backoff schedule without waiting real time.
    """
    last_error: CLICallError | None = None
    for attempt in range(max_retries + 1):
        try:
            return await op()
        except CLICallError as exc:
            last_error = exc
            if attempt < max_retries:
                await sleep(base_delay_s * 2**attempt)
    assert last_error is not None
    raise last_error
