"""Shared helpers for viewer/tools check scripts.

Headless on purpose: only the parser layer is imported here, never
PySide6 / the GUI, so these scripts run without a display.
"""

import argparse
import json
import math
import os
import sys


def ensure_repo_root():
    # viewer/tools/_common.py -> repo root is two levels up.
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


ensure_repo_root()


def is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def collect_paths(inputs, extensions=None, recursive=False):
    """Expand files/dirs from CLI args into a sorted list of files."""
    extensions = {e.lower().lstrip(".") for e in (extensions or [])}
    found = []
    for item in inputs:
        if os.path.isdir(item):
            for dirpath, _dirs, files in os.walk(item):
                for name in files:
                    if extensions and os.path.splitext(name)[1].lower().lstrip(".") not in extensions:
                        continue
                    found.append(os.path.join(dirpath, name))
                if not recursive:
                    break
        elif os.path.isfile(item):
            found.append(item)
        else:
            found.append(item)  # keep missing paths so caller reports them
    return sorted(set(found))


def add_common_args(parser):
    parser.add_argument("paths", nargs="+", help="Files or folders to check")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recurse into subfolders when a folder is given")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print failures")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures (exit code 1)")


def _finite_vec3(vec):
    return all(math.isfinite(float(v)) for v in vec)


def check_smd_model(model):
    """Sanity-check an SmdModel (from parse_smd / parse_mdl / parse_vmdl_c / load_dmx).

    Returns {"stats": {...}, "warnings": [...], "errors": [...]}.
    """
    warnings = []
    errors = []

    bones = getattr(model, "bones", []) or []
    vertices = getattr(model, "vertices", []) or []
    triangles = getattr(model, "triangles", []) or []
    frames = getattr(model, "frames", []) or []

    stats = {
        "bones": len(bones),
        "vertices": len(vertices),
        "triangles": len(triangles),
        "frames": len(frames),
        "has_geometry": bool(getattr(model, "has_geometry", False)),
        "has_animation": bool(getattr(model, "has_animation", False)),
        "materials": sorted(getattr(model, "materials", set()) or set()),
        "vertex_targets": len(getattr(model, "vertex_targets", []) or []),
        "min_bound": getattr(model, "min_bound", None),
        "max_bound": getattr(model, "max_bound", None),
    }

    if not stats["has_geometry"] and not triangles:
        warnings.append("no geometry (triangles empty)")

    # Bones: duplicate ids, bad parents, cycles.
    bone_ids = [b.bone_id for b in bones]
    if len(bone_ids) != len(set(bone_ids)):
        errors.append("duplicate bone ids")
    bone_id_set = set(bone_ids)
    parent_of = {b.bone_id: b.parent_id for b in bones}
    for b in bones:
        if b.parent_id is not None and b.parent_id >= 0 and b.parent_id not in bone_id_set:
            errors.append(f"bone {b.bone_id} ({b.name!r}) has missing parent {b.parent_id}")
        if b.parent_id == b.bone_id:
            errors.append(f"bone {b.bone_id} ({b.name!r}) is its own parent")
    for bone_id in bone_id_set:
        seen = set()
        cur = bone_id
        while cur is not None and cur >= 0 and cur in parent_of:
            if cur in seen:
                errors.append(f"bone parent cycle reaches bone {cur}")
                break
            seen.add(cur)
            nxt = parent_of[cur]
            cur = nxt if nxt != cur else -1

    # Vertices: finiteness, bone refs, weights, UV range.
    bad_pos = bad_normal = bad_uv = bad_bone = 0
    weight_sum_off = 0
    uv_outside = 0
    for v in vertices:
        if not _finite_vec3(v.position):
            bad_pos += 1
        if not _finite_vec3(v.normal):
            bad_normal += 1
        if not all(math.isfinite(float(u)) for u in v.uv):
            bad_uv += 1
        elif any(u < -0.001 or u > 1.001 for u in v.uv):
            uv_outside += 1
        if v.parent_bone is not None and v.parent_bone >= 0 and bone_id_set and v.parent_bone not in bone_id_set:
            bad_bone += 1
        total = 0.0
        for _bone_id, weight in (v.links or []):
            if weight < 0:
                weight_sum_off += 1
            total += float(weight)
        if v.links and abs(total - 1.0) > 0.01:
            weight_sum_off += 1
    if bad_pos:
        errors.append(f"{bad_pos} vertices with non-finite position")
    if bad_normal:
        warnings.append(f"{bad_normal} vertices with non-finite normal")
    if bad_uv:
        errors.append(f"{bad_uv} vertices with non-finite UV")
    if bad_bone:
        errors.append(f"{bad_bone} vertices reference unknown parent bone")
    if weight_sum_off:
        warnings.append(f"{weight_sum_off} vertices with weight sum != 1 or negative weight")
    if uv_outside:
        warnings.append(f"{uv_outside} vertices with UV outside 0..1")
    stats["uv_outside"] = uv_outside

    # Triangles: index range, degenerate.
    nverts = len(vertices)
    degenerate = out_of_range = 0
    for t in triangles:
        if any(i < 0 or i >= nverts for i in t.indices):
            out_of_range += 1
            continue
        if len(set(t.indices)) < 3:
            degenerate += 1
            continue
        a = vertices[t.indices[0]].position
        b = vertices[t.indices[1]].position
        c = vertices[t.indices[2]].position
        ab = tuple(b[i] - a[i] for i in range(3))
        ac = tuple(c[i] - a[i] for i in range(3))
        cross = (ab[1] * ac[2] - ab[2] * ac[1],
                 ab[2] * ac[0] - ab[0] * ac[2],
                 ab[0] * ac[1] - ab[1] * ac[0])
        if cross == (0.0, 0.0, 0.0):
            degenerate += 1
    if out_of_range:
        errors.append(f"{out_of_range} triangles reference vertex index out of range")
    if degenerate:
        warnings.append(f"{degenerate} degenerate triangles")

    # Bounds vs actual vertex extremes.
    if vertices and stats["min_bound"] is not None and stats["max_bound"] is not None:
        try:
            mins = tuple(min(v.position[i] for v in vertices) for i in range(3))
            maxs = tuple(max(v.position[i] for v in vertices) for i in range(3))
            stats["vertex_min"] = mins
            stats["vertex_max"] = maxs
            if stats["min_bound"][0] > mins[0] + 1e-3 or stats["max_bound"][0] < maxs[0] - 1e-3:
                warnings.append("header bounds do not cover vertex extremes on X")
        except (TypeError, ValueError):
            pass

    # Frames: unknown bones, non-finite transforms.
    bad_frame_bone = bad_frame_value = 0
    for frame in frames:
        for bone_id, (pos, rot) in (frame.transforms or {}).items():
            if bone_id_set and bone_id not in bone_id_set:
                bad_frame_bone += 1
            if not (_finite_vec3(pos) and _finite_vec3(rot)):
                bad_frame_value += 1
    if bad_frame_bone:
        warnings.append(f"{bad_frame_bone} frame transforms reference unknown bone")
    if bad_frame_value:
        errors.append(f"{bad_frame_value} frame transforms with non-finite values")

    return {"stats": stats, "warnings": warnings, "errors": errors}


def report(path, result, args, label=""):
    """Print one result; returns True when it counts as a failure."""
    failed = bool(result.get("errors")) or (args.strict and bool(result.get("warnings")))
    if args.json:
        print(json.dumps({"path": path, "label": label, **result}, default=str))
    elif not args.quiet or failed:
        tag = "FAIL" if result.get("errors") else ("WARN" if result.get("warnings") else "OK")
        print(f"[{tag}] {path}" + (f" ({label})" if label else ""))
        stats = result.get("stats")
        if stats:
            parts = ", ".join(f"{k}={v}" for k, v in stats.items() if k not in ("materials",) and v not in (None, [], {}, False))
            if parts:
                print(f"       stats: {parts}")
            if stats.get("materials"):
                print(f"       materials: {', '.join(stats['materials'])}")
        for w in result.get("warnings", []):
            print(f"       warning: {w}")
        for e in result.get("errors", []):
            print(f"       error: {e}")
    return failed


def run_checks(args, check_one, extensions):
    paths = collect_paths(args.paths, extensions=extensions, recursive=args.recursive)
    if not paths:
        print("error: no files matched", file=sys.stderr)
        return 2
    failures = 0
    for path in paths:
        if not os.path.isfile(path):
            print(f"[FAIL] {path}\n       error: file not found", file=sys.stderr)
            failures += 1
            continue
        try:
            result, label = check_one(path)
        except Exception as error:  # parse errors are check failures, not crashes
            result = {"stats": {}, "warnings": [], "errors": [f"{type(error).__name__}: {error}"]}
            label = ""
        if report(path, result, args, label=label):
            failures += 1
    if not args.json and not args.quiet:
        print(f"\n{len(paths)} file(s) checked, {failures} failure(s)")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    raise NotImplementedError("subclass main()")
