"""Bounded-concurrency async map. A fixed pool of workers pulling from a queue: scheduling tens of thousands of
coroutines against one asyncio.Semaphore is quadratic (each acquire scans the waiter list)."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable


async def pmap[T, R](fn: Callable[[T], Awaitable[R]], xs: Iterable[T], concurrency: int) -> list[R]:
    xs = list(xs)
    out: list = [None] * len(xs)
    queue = asyncio.Queue()
    for i, x in enumerate(xs):
        queue.put_nowait((i, x))

    async def worker():
        while not queue.empty():
            i, x = queue.get_nowait()
            out[i] = await fn(x)

    await asyncio.gather(*[worker() for _ in range(min(concurrency, len(xs)))])
    return out
