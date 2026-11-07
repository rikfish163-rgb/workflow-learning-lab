"""Resume bounded HTTP ranges; callers must verify the assembled file SHA256.

Existing bytes are untrusted prefixes, never proof of integrity. A lock prevents
concurrent writers; transport errors retain bytes while protocol errors roll back
only the current response. No dependencies, model loading, or GPU access.
"""
import fcntl
import json
import os
import time
from http.client import HTTPException, IncompleteRead
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def fetch_part(item, *, request_bytes=16 * 1024 * 1024, max_failures=8,
               backoff_seconds=1, timeout=90):
    url, start, end, size, target = item
    if not (0 <= start <= end < size and request_bytes > 0 and max_failures > 0):
        raise ValueError('invalid range or retry limits')
    expected = end - start + 1
    # O_NOFOLLOW prevents a partial-file symlink from redirecting writes.
    fd = os.open(target, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+b', buffering=0) as output:
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        have = os.fstat(output.fileno()).st_size
        if have > expected:
            raise ValueError('oversized existing range; preserve for inspection')
        failures = 0
        while have < expected:
            before = have
            request_start = start + have
            request_end = min(request_start + request_bytes - 1, end)
            remaining = request_end - request_start + 1
            request = Request(url, headers={
                'Range': f'bytes={request_start}-{request_end}',
                'Accept-Encoding': 'identity',
            })
            try:
                with urlopen(request, timeout=timeout) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != (
                            f'bytes {request_start}-{request_end}/{size}'):
                        raise ValueError('server did not honor exact byte range')
                    length = response.headers.get('Content-Length')
                    if length is not None and int(length) != remaining:
                        raise ValueError('incorrect Content-Length')
                    if response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise ValueError('encoded byte ranges are unsupported')
                    output.seek(have)
                    while remaining:
                        try:
                            block = response.read(min(1024 * 1024, remaining))
                        except IncompleteRead as exc:
                            block = exc.partial
                            if len(block) > remaining:
                                raise ValueError('oversized partial response') from exc
                            output.write(block)
                            have += len(block)
                            raise EOFError('truncated range') from exc
                        if not block:
                            raise EOFError('truncated range')
                        output.write(block)
                        have += len(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise ValueError('oversized response')
            except ValueError:
                output.truncate(before)
                raise
            except (OSError, HTTPException, EOFError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                    raise
                failures += 1
                if failures >= max_failures:
                    raise
                print(json.dumps({'part': target.name, 'received_bytes': have,
                                  'failure': type(exc).__name__, 'failures': failures}), flush=True)
                time.sleep(backoff_seconds * min(2 ** (failures - 1), 8))
        print(json.dumps({'part': target.name, 'bytes': expected, 'failures': failures}), flush=True)
