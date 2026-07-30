import math
import re
import struct

from typing import Dict, List, Optional, Tuple

from viewer.skeleton import quat_slerp, quat_to_euler
from viewer.smd_parser import (
    SmdBone,
    SmdFrame,
    SmdModel,
    SmdTriangle,
    SmdVertex,
    VertexTarget,
)


DMX_HEADER_PATTERN = re.compile(
    r"<!--\s*dmx\s+encoding\s+(\S+)\s+(\d+)\s+format\s+(\S+)\s+(\d+)\s*-->"
)


class DmxParseError(ValueError):
    def __init__(self, message, offset=0):
        super().__init__(message)
        self.offset = offset


class DmxRef:
    __slots__ = ("element_id",)

    def __init__(self, element_id):
        self.element_id = element_id


class DmxBinaryRef:
    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index


class DmxElement:
    __slots__ = ("type_name", "name", "element_id", "attributes")

    def __init__(self, type_name):
        self.type_name = type_name
        self.name = ""
        self.element_id = ""
        self.attributes = {}


class DmxDocument:
    def __init__(self):
        self.root = None
        self.encoding = ""
        self.encoding_version = 0
        self.format_name = ""
        self.format_version = 0
        self.all_elements: List[DmxElement] = []
        self.elements_by_id: Dict[str, DmxElement] = {}

    def register(self, element):
        self.all_elements.append(element)

        if element.element_id:
            self.elements_by_id[element.element_id] = element

    def resolve(self, value):
        if isinstance(value, DmxElement):
            return value

        if isinstance(value, DmxRef):
            return self.elements_by_id.get(value.element_id)

        if isinstance(value, DmxBinaryRef):
            if 0 <= value.index < len(self.all_elements):
                return self.all_elements[value.index]
            return None

        return None

    def find_first(self, type_name):
        for element in self.all_elements:
            if element.type_name == type_name:
                return element

        return None

    def find_all(self, type_name):
        return [
            element
            for element in self.all_elements
            if element.type_name == type_name
        ]


def is_dmx_file(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            head = handle.read(128)
    except OSError:
        return False

    return head.lstrip().startswith(b"<!--") and b"dmx encoding" in head


def parse_dmx(path: str) -> DmxDocument:
    with open(path, "rb") as handle:
        data = handle.read()

    head_text = data[:256].decode("utf-8", errors="ignore")
    match = DMX_HEADER_PATTERN.search(head_text)

    if not match:
        raise ValueError("Missing DMX header")

    encoding = match.group(1).lower()
    encoding_version = int(match.group(2))

    if encoding.startswith("keyvalues2"):
        document = DmxDocument()
        document.encoding = encoding
        document.encoding_version = encoding_version
        document.format_name = match.group(3)
        document.format_version = int(match.group(4))
        document.root = _parse_keyvalues2(data.decode("utf-8", errors="ignore"), document)
        return document

    if encoding == "binary":
        header_end = data.find(b"\n")

        if header_end < 0:
            header_end = data.find(b"-->")

            if header_end >= 0:
                header_end += 2

        starts = _binary_data_starts(data, header_end)
        profiles = _binary_profiles(encoding_version)

        if encoding_version >= 4:
            string_modes = ["bytes", "count"]
        elif encoding_version >= 2:
            string_modes = ["count", "bytes"]
        else:
            string_modes = ["none"]

        errors = []

        for data_start in starts:
            for string_mode in string_modes:
                for profile in profiles:
                    document = DmxDocument()
                    document.encoding = encoding
                    document.encoding_version = encoding_version
                    document.format_name = match.group(3)
                    document.format_version = int(match.group(4))

                    try:
                        root = _parse_binary(
                            data,
                            encoding_version,
                            document,
                            profile,
                            string_mode,
                            data_start,
                        )

                        if root is None or not document.all_elements:
                            raise DmxParseError("DMX binary parse produced no elements", 0)

                        if not root.type_name:
                            raise DmxParseError("DMX binary parse produced an empty root type", 0)

                        return document

                    except DmxParseError as error:
                        errors.append(
                            (
                                error.offset,
                                f"start:{data_start} strtab:{string_mode} {_profile_description(profile)}: {error}",
                            )
                        )

                    except Exception as error:
                        errors.append(
                            (
                                0,
                                f"start:{data_start} strtab:{string_mode} {_profile_description(profile)}: {error}",
                            )
                        )

        errors.sort(key=lambda item: item[0], reverse=True)

        message = "Could not parse binary DMX.\n\nBest attempt:\n" + errors[0][1]

        if len(errors) > 1:
            message += "\n\nOther attempts:\n" + "\n".join(item[1] for item in errors[1:])

        if errors:
            message += "\n\nHex near best offset:\n" + _hex_dump(data, errors[0][0])

        raise ValueError(message)

    raise ValueError(f"Unsupported DMX encoding: {encoding}")


def _binary_data_starts(data, header_end):
    if header_end < 0:
        return [0]

    start = header_end + 1
    starts = []

    # Some writers null terminate the header line after the newline.
    if start < len(data) and data[start] == 0 and start + 1 < len(data):
        starts.append(start + 1)

    starts.append(start)

    unique = []

    for value in starts:
        if value not in unique and 0 <= value < len(data):
            unique.append(value)

    return unique


def _hex_dump(data, offset, radius=48):
    start = max(0, offset - radius)
    end = min(len(data), offset + radius)

    lines = []

    for i in range(start, end, 16):
        chunk = data[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)

        marker = ""

        if i <= offset < i + 16:
            marker = " <--"

        lines.append(f"{i:08x}: {hex_part:<48} {ascii_part}{marker}")

    return "\n".join(lines)


def _tokenize_keyvalues2(text: str):
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        character = text[i]

        if character in " \t\r\n":
            i += 1
            continue

        if character == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue

        if character in "{}[],":
            tokens.append(character)
            i += 1
            continue

        if character == '"':
            i += 1
            buffer = []

            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    next_char = text[i + 1]
                    mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                    buffer.append(mapping.get(next_char, next_char))
                    i += 2
                else:
                    buffer.append(text[i])
                    i += 1

            i += 1
            tokens.append(("str", "".join(buffer)))
            continue

        start = i

        while i < n and text[i] not in ' \t\r\n{}[],"':
            i += 1

        tokens.append(("str", text[start:i]))

    return tokens


class _KeyValues2Parser:
    def __init__(self, tokens, document):
        self.tokens = tokens
        self.document = document
        self.index = 0

    def peek(self):
        if self.index < len(self.tokens):
            return self.tokens[self.index]

        return None

    def next_token(self):
        token = self.peek()
        self.index += 1
        return token

    def expect_string(self):
        token = self.next_token()

        if isinstance(token, tuple):
            return token[1]

        raise ValueError("Expected a string token in DMX keyvalues2 data")

    def expect_symbol(self, symbol):
        token = self.next_token()

        if token != symbol:
            raise ValueError(f"Expected {symbol} in DMX keyvalues2 data")

    def parse_element(self):
        type_name = self.expect_string()
        return self.parse_element_with_type(type_name)

    def parse_element_with_type(self, type_name):
        self.expect_symbol("{")

        element = DmxElement(type_name)

        while self.peek() != "}":
            name = self.expect_string()
            value_type = self.expect_string()
            value = self.parse_value(value_type)

            element.attributes[name] = value

            if name == "id" and isinstance(value, str):
                element.element_id = value
            elif name == "name" and isinstance(value, str):
                element.name = value

        self.expect_symbol("}")
        self.document.register(element)

        return element

    def parse_value(self, value_type):
        if value_type.endswith("_array"):
            self.expect_symbol("[")

            items = []

            while self.peek() != "]":
                if self.peek() == ",":
                    self.next_token()
                    continue

                if value_type == "element_array":
                    items.append(self.parse_element_array_item())
                else:
                    items.append(self.expect_string())

            self.expect_symbol("]")
            return items

        if value_type == "element":
            token = self.peek()

            if isinstance(token, tuple):
                first = self.expect_string()

                if first == "elementid":
                    return DmxRef(self.expect_string())

                if self.peek() == "{":
                    return self.parse_element_with_type(first)

            return None

        return self.expect_string()

    def parse_element_array_item(self):
        token = self.peek()

        if isinstance(token, tuple):
            first = self.expect_string()

            if first == "elementid":
                return DmxRef(self.expect_string())

            if self.peek() == "{":
                return self.parse_element_with_type(first)

        return None


def _parse_keyvalues2(text: str, document: DmxDocument):
    header_end = text.find("-->")

    if header_end != -1:
        text = text[header_end + 3:]

    tokens = _tokenize_keyvalues2(text)
    parser = _KeyValues2Parser(tokens, document)

    return parser.parse_element()


class _BinaryProfile:
    def __init__(
        self,
        type_index_size,
        attr_name_index_size,
        elem_name_index_size,
        string_value_index_size,
        string_array_inline,
    ):
        self.type_index_size = type_index_size
        self.attr_name_index_size = attr_name_index_size
        self.elem_name_index_size = elem_name_index_size
        self.string_value_index_size = string_value_index_size
        self.string_array_inline = string_array_inline


def _profile_description(profile):
    string_array_mode = "inline" if profile.string_array_inline else "indexed"

    return (
        f"type{profile.type_index_size}/"
        f"attr{profile.attr_name_index_size}/"
        f"elem{profile.elem_name_index_size}/"
        f"str{profile.string_value_index_size}/"
        f"strarray:{string_array_mode}"
    )


def _binary_profiles(version):
    if version >= 5:
        return [
            _BinaryProfile(4, 4, 4, 4, True),
            _BinaryProfile(4, 4, 4, 4, False),
        ]

    if version == 4:
        return [
            _BinaryProfile(2, 2, 2, 2, True),
            _BinaryProfile(2, 2, 2, 2, False),
            _BinaryProfile(2, 2, 2, 4, True),
            _BinaryProfile(2, 4, 2, 4, True),
            _BinaryProfile(2, 4, 2, 2, True),
            _BinaryProfile(4, 4, 4, 4, True),
        ]

    if version == 3:
        return [
            # Documented binary v3 layout:
            # element type and attribute names are short string indices,
            # element names and string values are inline null terminated strings.
            _BinaryProfile(2, 2, 0, 0, True),
            _BinaryProfile(2, 2, 2, 0, True),
            _BinaryProfile(2, 2, 0, 0, False),
            _BinaryProfile(2, 2, 2, 0, False),
        ]

    if version >= 2:
        return [
            _BinaryProfile(2, 2, 0, 0, True),
            _BinaryProfile(2, 2, 2, 0, True),
        ]

    return [
        _BinaryProfile(0, 0, 0, 0, True),
    ]


class _BinaryReader:
    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset
        self.context = ""

    def _take(self, count):
        end = self.offset + count

        if end > len(self.data):
            raise DmxParseError(
                f"Unexpected end of DMX binary data at offset {self.offset}",
                self.offset,
            )

        offset = self.offset
        self.offset = end

        return offset

    def read_bytes(self, count):
        offset = self._take(count)
        return self.data[offset:offset + count]

    def read_uint8(self):
        offset = self._take(1)
        return self.data[offset]

    def read_uint16(self):
        offset = self._take(2)
        return struct.unpack_from("<H", self.data, offset)[0]

    def read_int16(self):
        offset = self._take(2)
        return struct.unpack_from("<h", self.data, offset)[0]

    def read_int32(self):
        offset = self._take(4)
        return struct.unpack_from("<i", self.data, offset)[0]

    def read_float(self):
        offset = self._take(4)
        return struct.unpack_from("<f", self.data, offset)[0]

    def read_cstring(self):
        end = self.data.find(b"\x00", self.offset)

        if end < 0:
            raise DmxParseError(
                f"Unterminated string in DMX binary data at offset {self.offset}",
                self.offset,
            )

        value = self.data[self.offset:end]
        self.offset = end + 1

        return value.decode("utf-8", errors="ignore")


def _read_index(reader, size):
    if size == 2:
        return reader.read_uint16()

    if size == 4:
        return reader.read_int32()

    raise DmxParseError(
        f"Invalid DMX binary index size {size} at offset {reader.offset}",
        reader.offset,
    )


def _read_string_index(reader, string_table, size):
    index = _read_index(reader, size)

    if index == 65535 or index == -1:
        return ""

    if 0 <= index < len(string_table):
        return string_table[index]

    raise DmxParseError(
        f"Bad DMX string index {index} at offset {reader.offset}",
        reader.offset,
    )


def _read_string_table(data, offset, encoding_version, mode):
    if mode == "none":
        return [], offset

    if encoding_version >= 4:
        size_size = 4
        length_or_count = struct.unpack_from("<i", data, offset)[0]
    elif encoding_version >= 2:
        size_size = 2
        length_or_count = struct.unpack_from("<h", data, offset)[0]
    else:
        return [], offset

    if length_or_count < 0 or length_or_count > 100000000:
        raise DmxParseError(
            f"Bad DMX string table value {length_or_count} at offset {offset}",
            offset,
        )

    start = offset + size_size

    if mode == "count":
        strings = []
        pos = start

        for _ in range(length_or_count):
            end = data.find(b"\x00", pos)

            if end < 0:
                raise DmxParseError(
                    f"Unterminated DMX string table entry at offset {pos}",
                    pos,
                )

            strings.append(data[pos:end].decode("utf-8", errors="ignore"))
            pos = end + 1

        return strings, pos

    # mode == "bytes"
    end_target = start + length_or_count

    if end_target > len(data):
        raise DmxParseError(
            f"DMX string table byte length runs past end of file at offset {offset}",
            offset,
        )

    strings = []
    pos = start

    while pos < end_target:
        end = data.find(b"\x00", pos, end_target)

        if end < 0:
            strings.append(data[pos:end_target].decode("utf-8", errors="ignore"))
            pos = end_target
            break

        strings.append(data[pos:end].decode("utf-8", errors="ignore"))
        pos = end + 1

    if pos < end_target:
        if any(byte != 0 for byte in data[pos:end_target]):
            raise DmxParseError(
                f"Unexpected bytes after DMX string table at offset {pos}",
                pos,
            )

        pos = end_target

    return strings, pos


def _parse_binary(
    data: bytes,
    encoding_version: int,
    document: DmxDocument,
    profile,
    string_mode="count",
    data_start=None,
):
    if data_start is None:
        header_end = data.find(b"\n")

        if header_end < 0:
            raise DmxParseError("Invalid DMX binary header", 0)

        data_start = header_end + 1

    string_table, offset = _read_string_table(
        data,
        data_start,
        encoding_version,
        string_mode,
    )

    reader = _BinaryReader(data, offset)

    element_count = reader.read_int32()

    if element_count < 0 or element_count > 10000000:
        raise DmxParseError(
            f"Bad DMX element count {element_count} at offset {reader.offset}",
            reader.offset,
        )

    elements = []

    # Pass 1: read all element headers.
    for element_index in range(element_count):
        reader.context = f"element header {element_index}"

        element = _read_binary_element_header(reader, string_table, profile)

        document.register(element)
        elements.append(element)

    # Pass 2: read attributes for each element.
    for element_index, element in enumerate(elements):
        reader.context = f"element {element_index} {element.type_name} {element.name}"
        _read_binary_attributes(reader, string_table, profile, element, element_index)

    if elements:
        document.root = elements[0]

    return document.root


def _read_binary_element_header(reader, string_table, profile):
    if profile.type_index_size:
        type_name = _read_string_index(reader, string_table, profile.type_index_size)
    else:
        type_name = reader.read_cstring()

    if profile.elem_name_index_size:
        name = _read_string_index(reader, string_table, profile.elem_name_index_size)
    else:
        name = reader.read_cstring()

    uuid_bytes = reader.read_bytes(16)

    element = DmxElement(type_name)
    element.name = name
    element.element_id = _format_uuid(uuid_bytes)

    return element


def _read_binary_attributes(reader, string_table, profile, element, element_index):
    attribute_count = reader.read_int32()

    if attribute_count < 0 or attribute_count > 1000000:
        raise DmxParseError(
            f"Bad DMX attribute count {attribute_count} in element {element_index} "
            f"{element.type_name} {element.name} at offset {reader.offset}",
            reader.offset,
        )

    for attribute_index in range(attribute_count):
        if profile.attr_name_index_size:
            attribute_name = _read_string_index(
                reader,
                string_table,
                profile.attr_name_index_size,
            )
        else:
            attribute_name = reader.read_cstring()

        type_code = reader.read_uint8()

        reader.context = (
            f"element {element_index} {element.type_name} {element.name} "
            f"attr {attribute_index} {attribute_name} type {type_code}"
        )

        try:
            value = _read_binary_value(reader, string_table, profile, type_code)
        except DmxParseError as error:
            raise DmxParseError(f"{reader.context}: {error}", error.offset)

        element.attributes[attribute_name] = value


def _format_uuid(raw: bytes) -> str:
    first, second, third = struct.unpack_from("<IHH", raw, 0)
    tail = raw[8:16].hex()

    return "{%08x-%04x-%04x-%s-%s}" % (
        first,
        second,
        third,
        tail[0:4],
        tail[4:12],
    )


def _array_scalar_code(type_code):
    # This engine branch uses array types starting at 15.
    if 15 <= type_code <= 31:
        scalar_code = type_code - 14

        if 1 <= scalar_code <= 14:
            return scalar_code

    # Other branches / versions.
    if 32 <= type_code <= 63:
        return type_code - 31

    if 128 <= type_code <= 191:
        scalar_code = type_code - 128

        if 1 <= scalar_code <= 14:
            return scalar_code

    if 224 <= type_code <= 255:
        scalar_code = type_code - 224

        if 1 <= scalar_code <= 14:
            return scalar_code

    return None


def _read_binary_value(reader, string_table, profile, type_code):
    if type_code == 1:
        index = reader.read_int32()

        if index == -2:
            reader.read_cstring()
            return DmxBinaryRef(-2)

        return DmxBinaryRef(index)

    if type_code == 2:
        return reader.read_int32()

    if type_code == 3:
        return reader.read_float()

    if type_code == 4:
        return reader.read_uint8() != 0

    if type_code == 5:
        if profile.string_value_index_size:
            return _read_string_index(
                reader,
                string_table,
                profile.string_value_index_size,
            )
        return reader.read_cstring()

    if type_code == 6:
        count = reader.read_int32()

        if count < 0 or count > 100000000:
            raise DmxParseError(
                f"Bad DMX binary blob count {count} at offset {reader.offset}",
                reader.offset,
            )

        return reader.read_bytes(count)

    if type_code == 7:
        return reader.read_int32() * 0.0001

    if type_code == 8:
        offset = reader._take(4)
        return struct.unpack_from("<4B", reader.data, offset)

    if type_code == 9:
        return (reader.read_float(), reader.read_float())

    if type_code == 10:
        return (reader.read_float(), reader.read_float(), reader.read_float())

    if type_code == 11:
        return (
            reader.read_float(),
            reader.read_float(),
            reader.read_float(),
            reader.read_float(),
        )

    if type_code == 12:
        return (reader.read_float(), reader.read_float(), reader.read_float())

    if type_code == 13:
        return (
            reader.read_float(),
            reader.read_float(),
            reader.read_float(),
            reader.read_float(),
        )

    if type_code == 14:
        return tuple(reader.read_float() for _ in range(16))

    scalar_code = _array_scalar_code(type_code)

    if scalar_code is not None and 1 <= scalar_code <= 14:
        count = reader.read_int32()

        if count < 0 or count > 100000000:
            raise DmxParseError(
                f"Bad DMX array count {count} at offset {reader.offset}",
                reader.offset,
            )

        if scalar_code == 5:
            if profile.string_array_inline or not profile.string_value_index_size:
                return [reader.read_cstring() for _ in range(count)]

            return [
                _read_string_index(reader, string_table, profile.string_value_index_size)
                for _ in range(count)
            ]

        return [
            _read_binary_value(reader, string_table, profile, scalar_code)
            for _ in range(count)
        ]

    raise DmxParseError(
        f"Unsupported DMX attribute type {type_code} at offset {reader.offset}",
        reader.offset,
    )


def _attr(element, name):
    if element is None:
        return None

    return element.attributes.get(name)


def _to_float(value, default=0.0):
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default

    return default


def _to_int(value, default=0):
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default

    return default


def _to_string(value, default=""):
    if value is None:
        return default

    if isinstance(value, str):
        return value

    return default


def _to_vector2(value, default=(0.0, 0.0)):
    if value is None:
        return default

    if isinstance(value, (tuple, list)):
        if len(value) >= 2:
            return (float(value[0]), float(value[1]))
        return default

    if isinstance(value, str):
        parts = value.split()

        if len(parts) >= 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except ValueError:
                return default

    return default


def _to_vector3(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        return default

    if isinstance(value, (tuple, list)):
        if len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        return default

    if isinstance(value, str):
        parts = value.split()

        if len(parts) >= 3:
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                return default

    return default


def _to_quaternion(value, default=(0.0, 0.0, 0.0, 1.0)):
    if value is None:
        return default

    if isinstance(value, (tuple, list)):
        if len(value) >= 4:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3]),
            )
        return default

    if isinstance(value, str):
        parts = value.split()

        if len(parts) >= 4:
            try:
                return (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
            except ValueError:
                return default

    return default


def _get_array(element, name):
    value = _attr(element, name)

    if isinstance(value, list):
        return value

    return []


def _get_int_array(element, name):
    return [_to_int(value) for value in _get_array(element, name)]


def _get_float_array(element, name):
    return [_to_float(value) for value in _get_array(element, name)]


def _get_vector2_array(element, name):
    return [_to_vector2(value) for value in _get_array(element, name)]


def _get_vector3_array(element, name):
    return [_to_vector3(value) for value in _get_array(element, name)]


def _get_quaternion_array(element, name):
    return [_to_quaternion(value) for value in _get_array(element, name)]


def _get_time_array(element, name):
    return [_to_float(value) for value in _get_array(element, name)]


def _normalize_vector(value):
    x, y, z = value
    length = math.sqrt(x * x + y * y + z * z)

    if length < 0.00000001:
        return (0.0, 0.0, 1.0)

    return (x / length, y / length, z / length)


def _build_bones(document):
    model_element = document.find_first("DmeModel")

    joints = []

    if model_element is not None:
        joint_list_value = _attr(model_element, "jointList")

        if joint_list_value is None:
            joint_list_value = _attr(model_element, "jointTransforms")

        if isinstance(joint_list_value, list):
            for reference in joint_list_value:
                joint = document.resolve(reference)

                if joint is not None:
                    joints.append(joint)

    if not joints:
        joints = document.find_all("DmeJoint")

    bones = []
    id_by_name = {}

    for index, joint in enumerate(joints):
        name = joint.name or f"bone_{index}"
        bones.append(SmdBone(bone_id=index, name=name, parent_id=-1))
        id_by_name[name] = index

    for index, joint in enumerate(joints):
        for child_reference in _get_array(joint, "children"):
            child = document.resolve(child_reference)

            if child is not None and child.name in id_by_name:
                bones[id_by_name[child.name]].parent_id = index

    transforms_by_name = {}

    if model_element is not None:
        base_states = _get_array(model_element, "baseStates")

        if base_states:
            transform_list = document.resolve(base_states[0])

            if transform_list is not None:
                for transform_reference in _get_array(transform_list, "transforms"):
                    transform = document.resolve(transform_reference)

                    if transform is not None and transform.name:
                        transforms_by_name[transform.name] = (
                            _to_vector3(_attr(transform, "position")),
                            _to_quaternion(_attr(transform, "orientation")),
                        )

    base_transforms = {}

    for bone in bones:
        joint = joints[bone.bone_id]

        position = None
        orientation = None

        if bone.name in transforms_by_name:
            position, orientation = transforms_by_name[bone.name]

        if position is None or orientation is None:
            transform = document.resolve(_attr(joint, "transform"))

            if transform is not None:
                if position is None:
                    position = _to_vector3(_attr(transform, "position"))

                if orientation is None:
                    orientation = _to_quaternion(_attr(transform, "orientation"))

        if position is None:
            position = (0.0, 0.0, 0.0)

        if orientation is None:
            orientation = (0.0, 0.0, 0.0, 1.0)

        base_transforms[bone.bone_id] = (position, orientation)

    return bones, id_by_name, base_transforms


def build_reference_model(document):
    bones, id_by_name, base_transforms = _build_bones(document)

    reference_frame = SmdFrame(
        time=0,
        transforms={
            bone_id: (position, quat_to_euler(orientation))
            for bone_id, (position, orientation) in base_transforms.items()
        },
    )

    model = SmdModel(version=1, bones=bones, frames=[reference_frame])

    vertices = []
    triangles = []
    materials = set()

    pos_pool_of = []
    norm_pool_of = []
    mesh_vertex_lists = []

    min_bound = [math.inf, math.inf, math.inf]
    max_bound = [-math.inf, -math.inf, -math.inf]

    for mesh in document.find_all("DmeMesh"):
        vertex_data = document.resolve(_attr(mesh, "currentState"))

        if vertex_data is None:
            mesh_vertex_lists.append([])
            continue

        positions = _get_vector3_array(vertex_data, "positions")
        pos_indices = _get_int_array(vertex_data, "positionsIndices")
        normals = _get_vector3_array(vertex_data, "normals")
        norm_indices = _get_int_array(vertex_data, "normalsIndices")
        uvs = _get_vector2_array(vertex_data, "textureCoordinates")
        uv_indices = _get_int_array(vertex_data, "textureCoordinatesIndices")

        flip_v = _to_int(_attr(vertex_data, "flipVCoordinates"), 0) != 0

        joint_count = _to_int(_attr(vertex_data, "jointCount"), 0)
        joint_weights = _get_float_array(vertex_data, "jointWeights")
        joint_indices = _get_int_array(vertex_data, "jointIndices")

        current_mesh_vertices = []

        for face_set_reference in _get_array(mesh, "faceSets"):
            face_set = document.resolve(face_set_reference)

            if face_set is None:
                continue

            material_element = document.resolve(_attr(face_set, "material"))

            if material_element is not None:
                material_name = _to_string(_attr(material_element, "mtlName"), "")

                if not material_name:
                    material_name = material_element.name or "material"
            else:
                material_name = "material"

            faces = _get_int_array(face_set, "faces")

            polygon = []
            polygons = []

            for face_index in faces:
                if face_index == -1:
                    if len(polygon) >= 3:
                        polygons.append(polygon)
                    polygon = []
                else:
                    polygon.append(face_index)

            if len(polygon) >= 3:
                polygons.append(polygon)

            for polygon in polygons:
                for corner in range(1, len(polygon) - 1):
                    corner_indices = (polygon[0], polygon[corner], polygon[corner + 1])
                    triangle_indices = []

                    for vertex_index in corner_indices:
                        if vertex_index < 0 or vertex_index >= len(pos_indices):
                            continue

                        pool_index = pos_indices[vertex_index]

                        if 0 <= pool_index < len(positions):
                            position = positions[pool_index]
                        else:
                            position = (0.0, 0.0, 0.0)

                        normal = (0.0, 0.0, 1.0)
                        norm_pool = -1

                        if norm_indices and vertex_index < len(norm_indices):
                            norm_pool = norm_indices[vertex_index]

                            if 0 <= norm_pool < len(normals):
                                normal = normals[norm_pool]

                        uv = (0.0, 0.0)

                        if uv_indices and vertex_index < len(uv_indices):
                            uv_pool = uv_indices[vertex_index]

                            if 0 <= uv_pool < len(uvs):
                                raw_uv = uvs[uv_pool]

                                if flip_v:
                                    uv = (raw_uv[0], 1.0 - raw_uv[1])
                                else:
                                    uv = (raw_uv[0], raw_uv[1])

                        links = []

                        if joint_count > 0 and joint_weights and joint_indices:
                            for weight_slot in range(joint_count):
                                flat = pool_index * joint_count + weight_slot

                                if flat >= len(joint_weights) or flat >= len(joint_indices):
                                    flat = vertex_index * joint_count + weight_slot

                                if flat >= len(joint_weights) or flat >= len(joint_indices):
                                    continue

                                weight = joint_weights[flat]
                                joint_index = joint_indices[flat]

                                if weight > 0.0 and 0 <= joint_index < len(bones):
                                    links.append((joint_index, weight))

                        parent_bone = links[0][0] if links else 0

                        smd_index = len(vertices)

                        vertices.append(
                            SmdVertex(
                                parent_bone=parent_bone,
                                position=position,
                                normal=normal,
                                uv=uv,
                                links=links,
                            )
                        )

                        pos_pool_of.append(pool_index)
                        norm_pool_of.append(norm_pool)
                        current_mesh_vertices.append(smd_index)
                        triangle_indices.append(smd_index)

                        for axis in range(3):
                            min_bound[axis] = min(min_bound[axis], position[axis])
                            max_bound[axis] = max(max_bound[axis], position[axis])

                    if len(triangle_indices) == 3:
                        triangles.append(
                            SmdTriangle(
                                material=material_name,
                                indices=tuple(triangle_indices),
                            )
                        )
                        materials.add(material_name)

        mesh_vertex_lists.append(current_mesh_vertices)

    model.vertices = vertices
    model.triangles = triangles
    model.materials = materials

    if triangles:
        model.has_geometry = True
        model.min_bound = tuple(min_bound)
        model.max_bound = tuple(max_bound)

        model.vertex_targets = _build_vertex_targets(
            document,
            model,
            pos_pool_of,
            norm_pool_of,
            mesh_vertex_lists,
        )
        model.has_animation = bool(model.vertex_targets)

    return model


def _build_vertex_targets(document, model, pos_pool_of, norm_pool_of, mesh_vertex_lists):
    targets = []

    base_positions = [vertex.position for vertex in model.vertices]
    base_normals = [vertex.normal for vertex in model.vertices]

    basis_overrides = {
        i: (base_positions[i], base_normals[i])
        for i in range(len(model.vertices))
    }

    targets.append(VertexTarget(time=0, overrides=basis_overrides, name="basis"))

    next_time = 1

    for mesh_index, mesh in enumerate(document.find_all("DmeMesh")):
        if mesh_index >= len(mesh_vertex_lists):
            continue

        smd_indices_for_mesh = mesh_vertex_lists[mesh_index]

        for delta_reference in _get_array(mesh, "deltaStates"):
            delta = document.resolve(delta_reference)

            if delta is None:
                continue

            delta_positions = _get_vector3_array(delta, "positions")
            delta_pos_indices = _get_int_array(delta, "positionsIndices")
            delta_normals = _get_vector3_array(delta, "normals")
            delta_norm_indices = _get_int_array(delta, "normalsIndices")

            position_offsets = {}

            for j, pool_index in enumerate(delta_pos_indices):
                if j < len(delta_positions):
                    position_offsets[pool_index] = delta_positions[j]

            normal_offsets = {}

            for k, pool_index in enumerate(delta_norm_indices):
                if k < len(delta_normals):
                    normal_offsets[pool_index] = delta_normals[k]

            if not position_offsets:
                continue

            overrides = {}

            for smd_index in smd_indices_for_mesh:
                offset = position_offsets.get(pos_pool_of[smd_index])

                if offset is None:
                    continue

                base_position = base_positions[smd_index]

                new_position = (
                    base_position[0] + offset[0],
                    base_position[1] + offset[1],
                    base_position[2] + offset[2],
                )

                new_normal = base_normals[smd_index]
                normal_offset = normal_offsets.get(norm_pool_of[smd_index])

                if normal_offset is not None:
                    new_normal = _normalize_vector(
                        (
                            new_normal[0] + normal_offset[0],
                            new_normal[1] + normal_offset[1],
                            new_normal[2] + normal_offset[2],
                        )
                    )

                overrides[smd_index] = (new_position, new_normal)

            if overrides:
                name = delta.name or f"delta {next_time}"
                targets.append(VertexTarget(time=next_time, overrides=overrides, name=name))
                next_time += 1

    return targets


def _find_key_interval(times, t):
    if t <= times[0]:
        return 0, 0, 0.0

    if t >= times[-1]:
        last = len(times) - 1
        return last, last, 0.0

    low = 0
    high = len(times) - 1

    while high - low > 1:
        mid = (low + high) // 2

        if times[mid] <= t:
            low = mid
        else:
            high = mid

    span = times[high] - times[low]

    if span <= 0.0:
        amount = 0.0
    else:
        amount = (t - times[low]) / span

    return low, high, amount


def _sample_vector_track(track, t):
    if track is None:
        return None

    times, values = track
    low, high, amount = _find_key_interval(times, t)

    if low == high:
        return values[low]

    a = values[low]
    b = values[high]

    return (
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
        a[2] + (b[2] - a[2]) * amount,
    )


def _sample_quaternion_track(track, t):
    if track is None:
        return None

    times, values = track
    low, high, amount = _find_key_interval(times, t)

    if low == high:
        return values[low]

    return quat_slerp(values[low], values[high], amount)


def build_animation_model(document, reference_model):
    animation_list = document.find_first("DmeAnimationList")

    if animation_list is None:
        return None

    clips = []

    for clip_reference in _get_array(animation_list, "animations"):
        clip = document.resolve(clip_reference)

        if clip is not None:
            clips.append(clip)

    if not clips:
        return None

    clip = clips[0]

    frame_rate = _to_float(_attr(clip, "frameRate"), 30.0)

    if frame_rate <= 0.0:
        frame_rate = 30.0

    time_frame = document.resolve(_attr(clip, "timeFrame"))

    offset = 0.0
    scale = 1.0
    duration = 0.0

    if time_frame is not None:
        offset = _to_float(_attr(time_frame, "offset"), 0.0)
        scale = _to_float(_attr(time_frame, "scale"), 1.0)
        duration = _to_float(_attr(time_frame, "duration"), 0.0)

    if scale <= 0.0:
        scale = 1.0

    if reference_model is not None and reference_model.bones:
        bones = reference_model.bones
        id_by_name = {bone.name: bone.bone_id for bone in bones}

        if reference_model.frames:
            base_transforms = reference_model.frames[0].transforms
        else:
            base_transforms = {}
    else:
        bones, id_by_name, base_quaternions = _build_bones(document)
        base_transforms = {
            bone_id: (position, quat_to_euler(orientation))
            for bone_id, (position, orientation) in base_quaternions.items()
        }

    position_tracks = {}
    rotation_tracks = {}

    for channel_reference in _get_array(clip, "channels"):
        channel = document.resolve(channel_reference)

        if channel is None:
            continue

        target = document.resolve(_attr(channel, "toElement"))

        if target is None:
            continue

        bone_name = target.name

        if bone_name not in id_by_name:
            continue

        attribute_name = _to_string(_attr(channel, "toAttribute"), "")

        log = document.resolve(_attr(channel, "log"))

        if log is None:
            continue

        layers = _get_array(log, "layers")

        if layers:
            layer = document.resolve(layers[0])
        else:
            layer = log

        if layer is None:
            continue

        times = _get_time_array(layer, "times")

        if not times:
            continue

        if attribute_name == "position":
            values = _get_vector3_array(layer, "values")

            if len(values) == len(times):
                position_tracks[bone_name] = (times, values)

        elif attribute_name in ("orientation", "rotation"):
            values = _get_quaternion_array(layer, "values")

            if len(values) == len(times):
                rotation_tracks[bone_name] = (times, values)

    if not position_tracks and not rotation_tracks:
        return None

    max_time = 0.0

    for times, _ in list(position_tracks.values()) + list(rotation_tracks.values()):
        if times:
            max_time = max(max_time, times[-1])

    if duration > 0.0:
        if max_time > 0.0:
            max_time = min(max_time, duration)
        else:
            max_time = duration

    effective_rate = frame_rate * scale
    frame_count = int(round(max_time * effective_rate)) + 1
    frame_count = max(frame_count, 1)

    frames = []

    for frame_index in range(frame_count):
        t = offset + frame_index / effective_rate
        transforms = {}

        for bone_name in set(position_tracks) | set(rotation_tracks):
            bone_id = id_by_name[bone_name]

            base = base_transforms.get(bone_id, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            base_position, base_euler = base

            position = _sample_vector_track(position_tracks.get(bone_name), t)

            if position is None:
                position = base_position

            orientation = _sample_quaternion_track(rotation_tracks.get(bone_name), t)

            if orientation is None:
                euler = base_euler
            else:
                euler = quat_to_euler(orientation)

            transforms[bone_id] = (position, euler)

        frames.append(SmdFrame(time=frame_index, transforms=transforms))

    return SmdModel(version=1, bones=list(bones), frames=frames)


def load_dmx(path: str):
    document = parse_dmx(path)

    reference_model = build_reference_model(document)
    animation_model = build_animation_model(document, reference_model)

    return reference_model, animation_model