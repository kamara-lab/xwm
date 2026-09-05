"""Read (and write) the safetensors format, with numpy and nothing else.

Pretrained vision encoders are published as ``.safetensors``, and the reference
reader for that format arrives with PyTorch attached. `xwm` has no PyTorch
dependency and should not acquire one to read a file, so this is a reader for
the format itself.

The format is worth stating in full, because that is why implementing it is
reasonable rather than reckless:

* 8 bytes, little-endian ``u64`` -- the length of the header.
* that many bytes of UTF-8 JSON -- a map from tensor name to
  ``{"dtype", "shape", "data_offsets": [begin, end]}``, plus an optional
  ``__metadata__`` entry of strings.
* the rest of the file -- tensor data, C-contiguous, at those offsets, which
  are relative to the end of the header.

That is the entire specification. There are no variable-length records, no
compression and no pointers, so a wrong offset produces a size mismatch rather
than plausible-looking noise.

The writer exists for tests: a reader verified against a mock of a format is a
reader verified against a belief about the format, so the fixtures are written
as real files and read back through the real path.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["read_safetensors", "write_safetensors", "safetensors_metadata"]

#: safetensors dtype names to numpy. ``BF16`` has no numpy equivalent and is
#: converted on read; every other entry is a direct view.
_DTYPES: dict[str, str] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "|i1",
    "U8": "|u1",
    "BOOL": "|b1",
}

_NUMPY_TO_SAFETENSORS = {np.dtype(v): k for k, v in _DTYPES.items()}


def _header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        body = handle.read(length)
    if len(body) != length:
        raise ValueError(f"{path} claims a {length}-byte header but has {len(body)}")
    return json.loads(body.decode()), 8 + length


def safetensors_metadata(path: str | Path) -> dict[str, Any]:
    """Tensor names, dtypes and shapes, without reading any tensor data.

    Cheap enough to call before deciding whether a checkpoint is the one you
    wanted -- it touches only the header.
    """
    header, _ = _header(Path(path))
    return {
        name: {"dtype": entry["dtype"], "shape": tuple(entry["shape"])}
        for name, entry in header.items()
        if name != "__metadata__"
    }


def read_safetensors(path: str | Path, *, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load tensors as numpy arrays.

    Args:
        names: read only these. The whole file is memory-mapped either way, so
            this saves the copies, not the mapping.

    ``BF16`` tensors are widened to ``float32``: numpy has no bfloat16, and
    silently reinterpreting the bytes as ``float16`` would produce numbers that
    are wrong rather than absent.
    """
    path = Path(path)
    header, start = _header(path)
    wanted = [n for n in header if n != "__metadata__"]
    if names is not None:
        missing = sorted(set(names) - set(wanted))
        if missing:
            raise KeyError(f"{path.name} has no tensor(s) {missing}")
        wanted = list(names)

    blob = np.memmap(path, dtype=np.uint8, mode="r", offset=start)
    out: dict[str, np.ndarray] = {}
    for name in wanted:
        entry = header[name]
        begin, end = entry["data_offsets"]
        shape = tuple(entry["shape"])
        raw = np.asarray(blob[begin:end])
        dtype = entry["dtype"]
        if dtype == "BF16":
            # bfloat16 is the top 16 bits of a float32: widen by placing the
            # bytes in the high half of a zeroed word.
            wide = np.zeros((raw.size // 2, 4), np.uint8)
            wide[:, 2:] = raw.view(np.uint8).reshape(-1, 2)
            values = wide.view("<f4").reshape(-1)
        elif dtype in _DTYPES:
            values = raw.view(_DTYPES[dtype])
        else:
            raise ValueError(f"unsupported safetensors dtype {dtype!r} for tensor {name!r}")
        expected = int(np.prod(shape)) if shape else 1
        if values.size != expected:
            raise ValueError(
                f"tensor {name!r} declares shape {shape} ({expected} values) but its "
                f"byte range holds {values.size}"
            )
        out[name] = np.array(values.reshape(shape), copy=True)
    return out


def write_safetensors(
    path: str | Path, tensors: dict[str, np.ndarray], *, metadata: dict[str, str] | None = None
) -> Path:
    """Write tensors in the real on-disk format. Used to build test fixtures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs, offset = [], 0
    for name, array in tensors.items():
        # `ascontiguousarray` has a minimum of one dimension, so it silently
        # turns a 0-d tensor into a 1-d one. Restore the shape it was given.
        shape = list(np.asarray(array).shape)
        array = np.ascontiguousarray(array).reshape(shape)
        dtype = _NUMPY_TO_SAFETENSORS.get(array.dtype.newbyteorder("<"))
        if dtype is None:
            raise ValueError(f"cannot write dtype {array.dtype} for tensor {name!r}")
        data = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    body = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        for blob in blobs:
            handle.write(blob)
    return path
