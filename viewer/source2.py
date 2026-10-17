"""Source 2 compiled resource container helpers.

Every compiled resource (.vmdl_c, .vtex_c, .vmat_c, ...) starts with the
same block table and, for most of them, stores its contents as a binary
KV3 version 1 blob (LZ4 compressed). This module owns that shared
plumbing so the individual parsers stay focused on their payload:

* ``read_resource_blocks`` maps block tags to absolute file offsets.
* ``read_kv1`` decodes a KV3 v1 blob into plain Python objects.
* ``lz4_block_decompress`` decompresses one raw LZ4 block.

Two KV1 string-table placements exist in compiled files. One folds the
string count into the Bytes4 array, the other keeps it separate; the
reader tries both and accepts the first layout that decodes.
"""

import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

# KV3 v1 node types (BinaryKV3.NodeType.cs).
_NULL = 1
_BOOLEAN = 2
_INT64 = 3
_UINT64 = 4
_DOUBLE = 5
_STRING = 6
_BLOB = 7
_ARRAY = 8
_OBJECT = 9
_ARRAY_TYPED = 10
_INT32 = 11
_UINT32 = 12
_BOOL_TRUE = 13
_BOOL_FALSE = 14
_INT64_ZERO = 15
_INT64_ONE = 16
_DOUBLE_ZERO = 17
_DOUBLE_ONE = 18

_SINGLETONS = {
    _NULL: None,
    _BOOL_TRUE: True,
    _BOOL_FALSE: False,
    _INT64_ZERO: 0,
    _INT64_ONE: 1,
    _DOUBLE_ZERO: 0.0,
    _DOUBLE_ONE: 1.0,
}

KV3_V1_MAGIC = 0x4B563301
S2_HEADER_VERSION = 12
S2_HEADER_SIZE = 16
S2_BLOCK_ENTRY_SIZE = 12


class Source2Error(ValueError):
    pass


class _Truncated(Exception):
    """Raised when a KV1 stream runs out mid-value."""


def lz4_block_decompress(src: bytes) -> bytes:
    """Decompress one raw LZ4 block (no frame header)."""
    out = bytearray()
    i = 0
    n = len(src)
    while i < n:
        token = src[i]
        i += 1
        lit_len = token >> 4
        if lit_len == 15:
            while True:
                if i >= n:
                    raise Source2Error("LZ4 stream truncated in literal length")
                byte = src[i]
                i += 1
                lit_len += byte
                if byte != 255:
                    break
        if lit_len:
            if i + lit_len > n:
                raise Source2Error("LZ4 stream truncated in literals")
            out += src[i:i + lit_len]
            i += lit_len
        if i >= n:
            break
        if i + 2 > n:
            raise Source2Error("LZ4 stream truncated in match offset")
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0 or offset > len(out):
            raise Source2Error(f"Bad LZ4 match offset {offset}")
        match_len = (token & 0xF) + 4
        if (token & 0xF) == 15:
            while True:
                if i >= n:
                    raise Source2Error("LZ4 stream truncated in match length")
                byte = src[i]
                i += 1
                match_len += byte
                if byte != 255:
                    break
        start = len(out) - offset
        if offset >= match_len:
            out += out[start:start + match_len]
        else:
            # Overlapping match: repeat the available pattern.
            pattern = bytes(out[start:])
            repeats = match_len // offset + 1
            out += (pattern * repeats)[:match_len]
    return bytes(out)


def read_resource_blocks(data: bytes) -> Dict[str, Tuple[int, int]]:
    """Return ``{tag: (absolute_offset, size)}`` for a resource file.

    A block entry's offset is relative to the address of its own offset
    field (entry start + 4), not to the start of the file or table.
    """
    if len(data) < S2_HEADER_SIZE:
        raise Source2Error("File too small to be a Source 2 resource")

    _file_size, header_version, _version, table_offset, block_count = (
        struct.unpack_from("<IHHII", data, 0)
    )
    if header_version != S2_HEADER_VERSION:
        raise Source2Error(f"Unsupported resource header version {header_version}")
    if block_count > 256:
        raise Source2Error(f"Implausible block count {block_count}")

    table_start = 8 + table_offset
    blocks: Dict[str, Tuple[int, int]] = {}
    for index in range(block_count):
        entry = table_start + index * S2_BLOCK_ENTRY_SIZE
        if entry + S2_BLOCK_ENTRY_SIZE > len(data):
            break
        tag = data[entry:entry + 4].decode("ascii", errors="replace")
        stored_offset, size = struct.unpack_from("<II", data, entry + 4)
        start = entry + 4 + stored_offset
        if 0 < size and 0 <= start < len(data):
            blocks[tag] = (start, min(size, len(data) - start))
    return blocks


class _KV1Reader:
    """Pull reader over one decompressed KV1 buffer."""

    def __init__(
        self,
        buf: bytes,
        count4: int,
        count8: int,
        string_start: int,
        on_object: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        bytes8_start = string_start - count8 * 8
        if bytes8_start < 4 or string_start > len(buf):
            raise Source2Error("KV1 tables overrun the buffer")
        self.b4 = bytearray(buf[4:bytes8_start])
        self.b8 = bytearray(buf[bytes8_start:string_start])
        self.on_object = on_object
        self.strings: List[str] = []

        string_count = struct.unpack_from("<I", buf, 0)[0]
        if string_count > (len(buf) - string_start):
            raise Source2Error("KV1 string count is implausible")

        pos = string_start
        for _ in range(string_count):
            end = buf.find(b"\x00", pos)
            if end < 0:
                raise _Truncated("string table overruns buffer")
            self.strings.append(buf[pos:end].decode("utf-8", errors="replace"))
            pos = end + 1
        self.types = bytearray(buf[pos:])

    def _take4(self) -> int:
        if len(self.b4) < 4:
            raise _Truncated("Bytes4 exhausted")
        value = struct.unpack_from("<i", self.b4, 0)[0]
        del self.b4[:4]
        return value

    def _take8(self, fmt: str) -> Any:
        if len(self.b8) < 8:
            raise _Truncated("Bytes8 exhausted")
        value = struct.unpack(fmt, self.b8[:8])[0]
        del self.b8[:8]
        return value

    def read_type(self) -> int:
        if not self.types:
            raise _Truncated("type stream exhausted")
        node = self.types.pop(0)
        if node & 0x80:
            node &= 0x7F
            if not self.types:
                raise _Truncated("type flag byte missing")
            self.types.pop(0)
        return node

    def _string_value(self, string_id: int) -> str:
        if string_id == -1:
            return ""
        if string_id < 0 or string_id >= len(self.strings):
            raise _Truncated(f"bad string id {string_id}")
        return self.strings[string_id]

    def read_value(self, node: int) -> Any:
        if node in _SINGLETONS:
            return _SINGLETONS[node]
        if node == _BOOLEAN:
            raise Source2Error("KV1 boolean values need the Bytes1 stream")
        if node in (_INT32, _UINT32):
            return self._take4()
        if node == _STRING:
            return self._string_value(self._take4())
        if node == _BLOB:
            length = self._take4()
            if length != 0:
                raise Source2Error("KV1 blobs need the Bytes1 stream")
            return b""
        if node == _ARRAY:
            count = self._take4()
            if count < 0 or count > 10_000_000:
                raise _Truncated(f"implausible array length {count}")
            return [self.read_member(True) for _ in range(count)]
        if node == _ARRAY_TYPED:
            count = self._take4()
            if count < 0 or count > 10_000_000:
                raise _Truncated(f"implausible array length {count}")
            subtype = self.read_type()
            return [self.read_value(subtype) for _ in range(count)]
        if node == _OBJECT:
            count = self._take4()
            if count < 0 or count > 1_000_000:
                raise _Truncated(f"implausible object size {count}")
            obj: Dict[str, Any] = {}
            for _ in range(count):
                key, value = self.read_member(False)
                obj[key] = value
            if self.on_object is not None:
                self.on_object(obj)
            return obj
        if node in (_INT64, _UINT64):
            return self._take8("<q")
        if node == _DOUBLE:
            return self._take8("<d")
        raise Source2Error(f"Unsupported KV1 node type {node}")

    def read_member(self, in_array: bool) -> Any:
        node = self.read_type()
        if in_array:
            return self.read_value(node)
        name = self._string_value(self._take4())
        return name, self.read_value(node)

    def read_root(self) -> Any:
        root_type = self.read_type()
        if root_type != _OBJECT:
            raise Source2Error(f"KV1 root is not an object (type {root_type})")
        return self.read_value(root_type)


def _decompress_v1(blob: bytes) -> Tuple[bytes, int, int]:
    if len(blob) < 40 or struct.unpack_from("<I", blob, 0)[0] != KV3_V1_MAGIC:
        raise Source2Error("KV3 blob is not version 1")
    compression, bytes1, count4, count8, uncompressed = struct.unpack_from(
        "<5I", blob, 20
    )
    if bytes1:
        raise Source2Error("KV3 Bytes1 stream is not supported")
    if compression == 0:
        buf = blob[40:40 + uncompressed]
    elif compression == 1:
        buf = lz4_block_decompress(blob[40:])
    else:
        raise Source2Error(f"Unsupported KV3 v1 compression {compression}")
    if uncompressed and len(buf) < uncompressed:
        raise _Truncated("KV1 payload is shorter than declared")
    return buf, count4, count8


def read_kv1(
    blob: bytes,
    on_object: Optional[Callable[[Dict[str, Any]], None]] = None,
    tolerate_truncation: bool = False,
) -> Tuple[Optional[Any], bool]:
    """Decode a KV3 v1 blob.

    Returns ``(root, truncated)``. With ``tolerate_truncation`` a blob
    whose tail runs out still returns the objects collected through
    ``on_object`` with a ``None`` root and ``truncated=True``.
    """
    try:
        buf, count4, count8 = _decompress_v1(blob)
    except _Truncated as error:
        if tolerate_truncation:
            return None, True
        raise Source2Error(f"Could not decode KV1 payload: {error}")

    candidates = (
        count4 * 4 + count8 * 8,
        4 + count4 * 4 + count8 * 8,
    )
    last_error: Optional[Exception] = None
    saw_truncation = False

    for string_start in candidates:
        if string_start > len(buf):
            continue
        reader = _KV1Reader(buf, count4, count8, string_start, on_object)
        try:
            root = reader.read_root()
            if isinstance(root, dict) and root:
                return root, False
            last_error = Source2Error("KV1 root did not decode to an object")
        except _Truncated as error:
            saw_truncation = True
            last_error = error
        except Source2Error as error:
            last_error = error

    if saw_truncation and tolerate_truncation:
        return None, True
    raise Source2Error(f"Could not decode KV1 payload: {last_error}")
