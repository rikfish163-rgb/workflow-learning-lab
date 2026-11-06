"""CLI preflight to avoid starting expensive runs on a nearly full filesystem."""
import json
import os
import shutil
import tempfile
from pathlib import Path


def require_space(output, minimum_bytes=512 * 1024 * 1024):
    directory = Path(output).resolve()
    while not directory.exists():
        directory = directory.parent
    if shutil.disk_usage(directory).free < minimum_bytes:
        raise OSError(f"insufficient free space under {directory}; need at least {minimum_bytes} bytes")


def atomic_json(path, value):
    """Keep the previous valid manifest if writing its replacement fails."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.tmp-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
