from typing import Dict, Tuple


class VmtParseError(ValueError):
    pass


def _tokenize(text: str):
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
        if character in "{}":
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
        while i < n and text[i] not in ' \t\r\n{}"':
            i += 1
        tokens.append(("str", text[start:i]))
    return tokens


def _read_block(tokens, i, out):
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token == "}":
            return i + 1
        if token == "{":
            i = _read_block(tokens, i + 1, None)
            continue
        key = token[1]
        i += 1
        if i < n and tokens[i] == "{":
            # Nested section like Proxies, skip its contents
            i = _read_block(tokens, i + 1, None)
            continue
        if i < n and isinstance(tokens[i], tuple):
            if out is not None:
                out[key.lower()] = tokens[i][1]
            i += 1
    return i


def parse_vmt(path: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a VMT file. Returns (shader_name, params) where params keys
    are lowercased top-level parameters like "$basetexture".
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        tokens = _tokenize(handle.read())

    if not tokens:
        raise VmtParseError("Empty VMT file")

    i = 0
    while i < len(tokens) and not isinstance(tokens[i], tuple):
        i += 1
    if i >= len(tokens):
        raise VmtParseError("No shader name found in VMT")
    shader = tokens[i][1]
    i += 1

    while i < len(tokens) and tokens[i] != "{":
        i += 1

    params: Dict[str, str] = {}
    if i < len(tokens):
        _read_block(tokens, i + 1, params)

    return shader.lower(), params