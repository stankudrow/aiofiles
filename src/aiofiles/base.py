import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager
from functools import partial, wraps
from queue import Empty, Queue


def to_agen(cb: Callable) -> Callable:
    @wraps(cb)
    async def _wrapper(*args, **kwargs) -> AsyncIterator:
        def _iterate(
            q: Queue, *, next_item_event: threading.Event, eoi_event: threading.Event
        ):
            try:
                for row in cb(*args, **kwargs):
                    # the event is cleared so that the iteration gets blocked
                    # until the main generator allows the next iteration
                    next_item_event.clear()
                    q.put(row)
                    next_item_event.wait()
            finally:
                eoi_event.set()  # end of iteration

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
            # in case the iterator is exhausted at this point
            if end_of_iteration.is_set():
                break
            try:
                # the `get_nowait` method is a remedy here
                # because `queue.get()` could block the thread
                # when queue is empty, but EOI was not set yet
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
    async def _wrapper(*args, **kwargs) -> Coroutine:
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
