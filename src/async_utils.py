import asyncio
import types
from contextlib import contextmanager


def is_async_function(func):
    return isinstance(func, types.FunctionType) and asyncio.iscoroutinefunction(func)


@contextmanager
def maybe_async_func(func):
    if is_async_function(func):

        def wrapped(*args, **kwargs):
            return asyncio.run(func(*args, **kwargs))

        yield wrapped
    else:
        yield func
