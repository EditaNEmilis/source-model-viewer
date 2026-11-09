"""Check Source model files headlessly (no GUI).

Dispatch mirrors MainWindow.open_model:
  .smd / .vta   -> parse_smd
  .mdl          -> parse_mdl (Source 1, GoldSrc, HL2 beta; needs .vvd/.vtx next to it for geometry)
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
        from viewer.mdl_parser import is_mdl_file, parse_mdl
        if not is_mdl_file(path):
            return {"stats": {}, "warnings": [], "errors": ["not an MDL file (magic check failed)"]}, "mdl"
        return check_smd_model(parse_mdl(path)), "mdl"

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
