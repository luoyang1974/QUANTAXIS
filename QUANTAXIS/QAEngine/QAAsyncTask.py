import asyncio
import sys
import traceback

import async_timeout


class QA_AsyncTask:
    _source_traceback = None
    _closed = False
    _explicit = False
    _task = None

    def __init__(self, coro, scheduler, loop):
        self._loop = loop
        self._coro = coro
        self._scheduler = scheduler
        self._started = loop.create_future()

        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(2))

    def __repr__(self):
        info = []
        if self._closed:
            info.append('closed')
        elif self._task is None:
            info.append('pending')
        info = ' '.join(info)
        if info:
            info += ' '
        return f'<Job {info}coro=<{self._coro}>>'

    @property
    def active(self):
        return not self.closed and not self.pending

    @property
    def pending(self):
        return self._task is None and not self.closed

    @property
    def closed(self):
        return self._closed

    async def _do_wait(self, timeout):
        async with async_timeout.timeout(timeout):
            await self._started
            if self._task is not None:
                return await self._task
            return None

    async def wait(self, *, timeout=None):
        if self._closed:
            return
        self._explicit = True
        scheduler = self._scheduler
        try:
            return await asyncio.shield(self._do_wait(timeout))
        except asyncio.CancelledError:
            raise
        except Exception:
            if scheduler is not None and hasattr(scheduler, "close_timeout"):
                await self._close(scheduler.close_timeout)
            else:
                await self._close(1.0)  # 默认超时时间
            raise

    async def close(self, *, timeout=None):
        if self._closed:
            return
        self._explicit = True
        if timeout is None:
            if self._scheduler is not None and hasattr(self._scheduler, "close_timeout"):
                timeout = self._scheduler.close_timeout
            else:
                timeout = 1.0  # 默认超时时间
        await self._close(timeout)

    async def _close(self, timeout):
        self._closed = True
        if self._task is None:
            # the task is closed immediately without actual execution
            # it prevents a warning like
            # RuntimeWarning: coroutine 'coro' was never awaited
            self._start()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        scheduler = self._scheduler
        try:
            async with async_timeout.timeout(timeout):
                if self._task is not None:
                    await self._task
        except asyncio.CancelledError:
            pass
        except TimeoutError as exc:
            if self._explicit:
                raise
            context = {'message': "Job closing timed out",
                       'job': self,
                       'exception': exc}
            if self._source_traceback is not None:
                context['source_traceback'] = self._source_traceback
            if scheduler is not None and hasattr(scheduler, "call_exception_handler"):
                scheduler.call_exception_handler(context)
        except Exception as exc:
            if self._explicit:
                raise
            self._report_exception(exc)

    def _start(self):
        assert self._task is None
        self._task = self._loop.create_task(self._coro)
        self._task.add_done_callback(self._done_callback)
        self._started.set_result(None)

    def _done_callback(self, task):
        scheduler = self._scheduler
        if scheduler is not None and hasattr(scheduler, "_done"):
            scheduler._done(self)
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                pass
            else:
                if exc is not None and not self._explicit:
                    self._report_exception(exc)
                    if hasattr(scheduler, "_failed_tasks"):
                        scheduler._failed_tasks.put_nowait(task)
        self._scheduler = None  # drop backref
        self._closed = True

    def _report_exception(self, exc):
        context = {'message': "Job processing failed",
                   'job': self,
                   'exception': exc}
        if self._source_traceback is not None:
            context['source_traceback'] = self._source_traceback
        if self._scheduler is not None and hasattr(self._scheduler, "call_exception_handler"):
            self._scheduler.call_exception_handler(context)
