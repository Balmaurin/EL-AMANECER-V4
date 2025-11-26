import hashlib
import time
from functools import wraps


def measure_time(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = time.time() - t0
        return out, dt

    return wrapper


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_query(q: str, max_len: int = 512) -> str:
    q = "".join(ch for ch in q if ch.isprintable())
    return q[:max_len]
