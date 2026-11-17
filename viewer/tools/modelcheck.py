"""Check Source model files headlessly (no GUI).

Dispatch mirrors MainWindow.open_model:
  .smd / .vta   -> parse_smd
  .mdl          -> parse_mdl (Source 1, GoldSrc, HL2 beta. Needs .vvd/.vtx next to it for geometry)
  .vmdl_c       -> parse_vmdl_c (Source 2)
  .dmx          -> load_dmx, checks the reference model plus clip count

Usage:
  python viewer/tools/modelcheck.py path/to/model.smd
  python viewer/tools/modelcheck.py models/ -r
  python viewer/tools/modelcheck.py models/ -r --strict --json
  python -m viewer.tools.modelcheck path/to/model.mdl

Exit code: 0 ok, 1 check failed, 2 no files matched.
"""
import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viewer.tools._common import add_common_args, check_smd_model, run_checks

EXTENSIONS = ("smd", "vta", "mdl", "vmdl_c", "dmx")


def _add_compiled_animation_stats(model, result):
    """Report clip, bodygroup and blend-grid metadata on compiled MDLs."""
    metadata = getattr(model, "metadata", None) or {}
    clips = metadata.get("animation_clips") or []
    warnings = []
    blend_cells = 0
    root_motion_clips = 0

    for _name, clip in clips:
        clip_md = getattr(clip, "metadata", None) or {}
        cells = clip_md.get("blend_cells") or []
        blend_cells += len(cells)
        if clip_md.get("movements"):
            root_motion_clips += 1

        groupsize = clip_md.get("groupsize") or (1, 1)
        try:
            expected = max(1, int(groupsize[0])) * max(1, int(groupsize[1]))
        except (TypeError, ValueError, IndexError):
            expected = 1

        label = clip_md.get("name") or "unnamed"
        if cells and len(cells) != expected:
            warnings.append(
                f"clip {label!r}: {len(cells)} blend cells do not match "
                f"groupsize {tuple(groupsize)}"
            )
        for cell in cells:
            cell_model = cell.get("model")
            if cell_model is None or not getattr(cell_model, "frames", None):
                warnings.append(f"clip {label!r}: empty blend cell")
                break

    result["stats"].update({
        "animation_clips": len(clips),
        "bodygroups": len(metadata.get("bodygroups") or []),
        "blend_cells": blend_cells,
        "root_motion_clips": root_motion_clips,
    })
    result["warnings"].extend(warnings)


def check_one(path):
    lower = path.lower()
    if lower.endswith(".dmx"):
        from viewer.dmx_parser import is_dmx_file, load_dmx
        if not is_dmx_file(path):
            return {"stats": {}, "warnings": [], "errors": ["not a DMX file (magic check failed)"]}, "dmx"
        reference, clips = load_dmx(path)
        if reference is None:
            return {"stats": {"animation_clips": len(clips or [])},
                    "warnings": ["no reference model in DMX"] if clips else [],
                    "errors": [] if clips else ["no reference model and no animation clips"]}, "dmx"
        result = check_smd_model(reference)
        result["stats"]["animation_clips"] = len(clips or [])
        return result, "dmx"

    if lower.endswith(".vmdl_c"):
        from viewer.mdl_parser import is_vmdl_c_file
        from viewer.vmdl_parser import parse_vmdl_c
        if not is_vmdl_c_file(path):
            return {"stats": {}, "warnings": [], "errors": ["not a .vmdl_c file (magic check failed)"]}, "vmdl_c"
        return check_smd_model(parse_vmdl_c(path)), "vmdl_c"

    if lower.endswith(".mdl"):
        from viewer.mdl_parser import parse_mdl
        model = parse_mdl(path)
        result = check_smd_model(model)
        _add_compiled_animation_stats(model, result)
        return result, "mdl"

    # .smd / .vta (and anything else, like the GUI falls back to SMD).
    from viewer.smd_parser import parse_smd
    return check_smd_model(parse_smd(path)), os.path.splitext(lower)[1].lstrip(".")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check Source model files headlessly.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    return run_checks(args, check_one, EXTENSIONS)


if __name__ == "__main__":
    sys.exit(main())
