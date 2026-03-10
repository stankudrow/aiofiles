import asyncio
import time
from collections.abc import AsyncIterable, Generator, Iterable
from typing import Any

from aiofiles.base import to_agen, wrap


class TestToAsyncGeneratorWrapper:
    """Test suite for the `to_agen` decorator."""

    def _iter_io(
        self, it: Iterable, *, with_timeout: float = 1.0, with_result: Any = 42
    ) -> Generator[Any, None, None]:
        for item in it:
            time.sleep(with_timeout)
            yield item
        return with_result

    async def _do_aiter(
        self, ait: AsyncIterable[str], *, acc: list[str], lock: asyncio.Lock
    ) -> str:
        letters: list[str] = []
        async for item in ait:
            async with lock:
                letters.append(item)
                acc.append(item)
            # `async for` does not responsible for coroutine switching
            # so we need to yield control back to the event loop
            await asyncio.sleep(0)
        return "".join(letters)

    async def test_to_agen(self) -> None:
        lock = asyncio.Lock()
        timeout: float = 0.01
        words: list[str] = ["Hello", "Aiofiles"]
        word_lengths: list[int] = [len(word) for word in words]
        accumulator: list[str] = []
        tasks: list[asyncio.Task] = []

        start = time.time()
        for word in words:
            task = asyncio.create_task(
                self._do_aiter(
                    to_agen(self._iter_io)(word, with_timeout=timeout),
                    acc=accumulator,
                    lock=lock,
                )
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        result: str = "".join(accumulator)

        assert max(word_lengths) * timeout < elapsed < sum(word_lengths) * timeout * 2
        assert set(results) == set(words)

        for word in words:
            # testing non-sequential ordering
            assert word not in result


class TestToCoroutineWrapper:
    """Test suite for the wrap decorator."""

    def _do_io(self, *, with_timeout: float = 1.0, with_result: Any = 42) -> Any:
        time.sleep(with_timeout)
        return with_result

    async def test_wrap(self) -> None:
        tasks: list[asyncio.Task] = []
        seconds = list(range(1, 11))

        start = time.time()
        for second in seconds:
            task = asyncio.create_task(
                wrap(self._do_io)(with_timeout=second / 10, with_result=second)
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        assert 0.1 < elapsed < 2  # 2 will do
        assert all(result == answer for result, answer in zip(results, seconds))
