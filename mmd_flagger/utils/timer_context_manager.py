import time
from contextlib import contextmanager
from typing import Generator

import logging
logger = logging.getLogger(__name__)

class TimeResult:
    """A simple container to hold the duration result."""
    duration: float = 0.0


@contextmanager
def timer(label: str = "Task") -> Generator[TimeResult, None, None]:
    """
    A reusable context manager to measure code blocks.
    Yields a TimeResult object so you can access the duration 
    programmatically after the block finishes.
    
    Usage:
        with timer("Heavy Algorithm") as t:
            run_algorithm()
        print(f"Done! Result was {t.duration}")
    """
    res = TimeResult()
    start = time.perf_counter()
    try:
        yield res
    except Exception as e:
        logger.exception(f"[{label}] Failed after {time.perf_counter() - start:.6f}s: {e}")
        raise RuntimeError(e)
    finally:
        res.duration = time.perf_counter() - start
        logger.debug(f"[{label}] Finished in {res.duration:.6f} seconds")    
