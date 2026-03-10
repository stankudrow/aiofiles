import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from functools import partial, wraps
from queue import Empty, Queue
from typing import Any


def to_agen(cb: Callable) -> Callable:
    @wraps(cb)
    async def _wrapper(*args, **kwargs) -> AsyncIterator:
        def _iterate(
            q: Queue, *, next_item_event: threading.Event, eoi_event: threading.Event
        ) -> None:
            try:
                for row in cb(*args, **kwargs):
                    # The `next_item_event` is cleared here
                    # so that the current iteration will be blocked at the end
                    # until the main generator allows the next iteration.
                    # By this `yield-like` lazy behaviour is achieved
                    # and the queue is filled successively and on-demand.
                    next_item_event.clear()
                    q.put(row)
                    next_item_event.wait()
            finally:
                eoi_event.set()

        loop = asyncio.get_running_loop()
        queue: Queue = Queue()  # thread-safe
        ready_for_item = threading.Event()
        end_of_iteration = threading.Event()
        gen = partial(
            _iterate,
            q=queue,
            next_item_event=ready_for_item,
            eoi_event=end_of_iteration,
        )
        loop.run_in_executor(None, gen)

        while True:
            # In case the iterator is exhausted at the very beginning
            if end_of_iteration.is_set():
                break
            try:
                # The `get_nowait` method is a remedy here
                # because `queue.get()` could block the thread
                # when queue is empty while EOI was not set.
                # Playing with timeouts can also get the iteration stuck.
                item = queue.get_nowait()
            except Empty:
                continue
            ready_for_item.set()
            queue.task_done()
            yield item
        queue.join()

    return _wrapper


def wrap(cb: Callable) -> Callable:
    @wraps(cb)
    async def _wrapper(*args, **kwargs) -> Any:
        return await asyncio.to_thread(cb, *args, **kwargs)

    return _wrapper


class AsyncBase:
    def __init__(self, file, loop, executor):
        self._file = file
        self._executor = executor
        self._ref_loop = loop

    @property
    def _loop(self):
        return self._ref_loop or asyncio.get_running_loop()

    def __aiter__(self):
        return self

    def __repr__(self):
        return super().__repr__() + " wrapping " + repr(self._file)

    async def __anext__(self):
        """Simulate normal file iteration."""

        if line := await self.readline():
            return line
        raise StopAsyncIteration


class AsyncIndirectBase(AsyncBase):
    def __init__(self, name, loop, executor, indirect):
        self._indirect = indirect
        self._name = name
        super().__init__(None, loop, executor)

    @property
    def _file(self):
        return self._indirect()

    @_file.setter
    def _file(self, v):
        pass  # discard writes


class AiofilesContextManager(Awaitable, AbstractAsyncContextManager):
    """An adjusted async context manager for aiofiles."""

    __slots__ = ("_coro", "_obj")

    def __init__(self, coro):
        self._coro = coro
        self._obj = None

    def __await__(self):
        if self._obj is None:
            self._obj = yield from self._coro.__await__()
        return self._obj

    async def __aenter__(self):
        return await self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.get_running_loop().run_in_executor(
            None, self._obj._file.__exit__, exc_type, exc_val, exc_tb
        )
        self._obj = None
