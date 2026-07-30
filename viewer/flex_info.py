import re

from typing import Dict, Optional


def _read_text_lines(path: str):
    with open(path, "rb") as handle:
        data = handle.read()

    if not data:
        return []

    if data[:3] == b"\xef\xbb\xbf":
        text = data.decode("utf-8-sig", errors="ignore")

    elif data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16", errors="ignore")

    else:
        sample = data[:4096]

        if b"\x00" in sample:
            try:
                text = data.decode("utf-16", errors="ignore")
            except UnicodeError:
                text = data.decode("utf-8", errors="ignore")
        else:
            text = data.decode("utf-8", errors="ignore")

    return text.splitlines()


FPS_PATTERN = re.compile(r"\bfps\s+([0-9]+(?:\.[0-9]+)?)", re.I)
INTENSITY_PATTERN = re.compile(r"\bintensity\s+([-+]?[0-9]*\.?[0-9]+)", re.I)
PROGRESS_PATTERN = re.compile(r"\bprogress\s+([-+]?[0-9]*\.?[0-9]+)", re.I)
PROGRESS_AXIS_PATTERN = re.compile(r"\bprogressaxis\s+([yz])", re.I)

QC_FLEX_PATTERN = re.compile(
    r"\b(defaultflex|flex|flexpair)\b(.*?)\bframe\s+(\d+)",
    re.I,
)

SIMPLE_NAME_PATTERN = re.compile(
    r"^(?:time|frame|index)?\s*(-?\d+)\s*[:=#;-]?\s*(.*)$",
    re.I,
)

DRIVER_BONE_PATTERNS = [
    re.compile(r"\bdriverbone\s+\"([^\"]+)\"", re.I),
    re.compile(r"\bdriverbone\s+(\S+)", re.I),
    re.compile(r"\bdriver\s+bone\s+\"([^\"]+)\"", re.I),
    re.compile(r"\bdriver\s+bone\s+(\S+)", re.I),
    re.compile(r"\bboneflexdriver\b\s+\"([^\"]+)\"", re.I),
]

NUMERIC_START_PATTERN = re.compile(r"^[-+0-9.]")
NUMERIC_TOKEN_PATTERN = re.compile(r"^[-+0-9.]+$")


def parse_flex_info(path: str) -> Dict[str, object]:
    names = {}
    fps = None
    intensity = None
    progress = None
    driver_bone = None
    progress_axis = None

    for raw_line in _read_text_lines(path):
        line = raw_line.strip()

        if not line:
            continue

        fps_match = FPS_PATTERN.search(line)
        if fps_match:
            try:
                fps = float(fps_match.group(1))
            except ValueError:
                pass
            continue

        intensity_match = INTENSITY_PATTERN.search(line)
        if intensity_match:
            try:
                intensity = float(intensity_match.group(1))
            except ValueError:
                pass
            continue

        progress_match = PROGRESS_PATTERN.search(line)
        if progress_match:
            try:
                progress = float(progress_match.group(1))
            except ValueError:
                pass
            continue

        axis_match = PROGRESS_AXIS_PATTERN.search(line)
        if axis_match and progress_axis is None:
            progress_axis = axis_match.group(1).lower()
            continue

        if driver_bone is None:
            found_driver = None

            for pattern in DRIVER_BONE_PATTERNS:
                driver_match = pattern.search(line)

                if driver_match:
                    found_driver = driver_match.group(1)
                    break

            if found_driver:
                driver_bone = found_driver
                continue

        qc_match = QC_FLEX_PATTERN.search(line)
        if qc_match:
            command = qc_match.group(1).lower()
            args = qc_match.group(2)
            frame = int(qc_match.group(3))

            name = None

            quoted_match = re.search(r'"([^"]+)"', args)
            if quoted_match:
                name = quoted_match.group(1)

            elif command == "defaultflex":
                name = "default"

            else:
                tokens = args.split()

                for token in tokens:
                    cleaned = token.strip('",')

                    if cleaned and not NUMERIC_TOKEN_PATTERN.match(cleaned):
                        name = cleaned
                        break

            if name:
                names[frame] = name

            continue

        simple_match = SIMPLE_NAME_PATTERN.match(line)
        if simple_match:
            time_value = int(simple_match.group(1))
            rest = simple_match.group(2).strip()

            if not rest:
                continue

            for separator in (" #", " //", " ;"):
                if separator in rest:
                    rest = rest.split(separator, 1)[0].strip()

            rest = rest.strip().strip('"')

            if not rest:
                continue

            if NUMERIC_START_PATTERN.match(rest):
                continue

            parts = rest.split()

            if len(parts) >= 3 and all(NUMERIC_TOKEN_PATTERN.match(part) for part in parts):
                continue

            names[time_value] = rest

    return {
        "names": names,
        "fps": fps,
        "intensity": intensity,
        "progress": progress,
        "driver_bone": driver_bone,
        "progress_axis": progress_axis,
    }