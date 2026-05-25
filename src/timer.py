import time
import functools


def timer(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            label = name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"[TIMER] {label}: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator
