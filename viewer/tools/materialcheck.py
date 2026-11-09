"""Check Source material files headlessly (no GUI).

  .vmt      -> parse_vmt (shader + key/value params)
  .vmat_c   -> parse_vmat_c (Source 2; shader + texture/int/float/vector params)

Warns on common mistakes: missing $basetexture-like param, texture refs
that are empty, texture params pointing at files not found next to the
material (only checked when the ref looks like a plain relative name).

Usage:
  python viewer/tools/materialcheck.py path/to/mat.vmt
  python viewer/tools/materialcheck.py materials/ -r
  python -m viewer.tools.materialcheck path/to/mat.vmat_c

Exit code: 0 ok, 1 check failed, 2 no files matched.
"""
import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viewer.tools._common import add_common_args, run_checks

EXTENSIONS = ("vmt", "vmat_c")


def _check_texture_refs(base_dir, texture_params, warnings):
    for name, ref in sorted(texture_params.items()):
        if not ref or not isinstance(ref, str):
            warnings.append(f"texture param {name!r} has empty value")
            continue
        # Refs are usually like "brick/brickwall" or "brick/brickwall.vtf";
        # only probe the filesystem for simple relative names.
        if ref.startswith(("#", "$", "!", "sprites/", "materials/")):
            continue
        candidate = os.path.join(base_dir, ref)
        candidates = [candidate, candidate + ".vtf", candidate + ".vtex_c"]
        if not any(os.path.isfile(c) for c in candidates):
            warnings.append(f"texture param {name!r} points at missing file {ref!r} (looked next to material)")


def check_one(path):
    lower = path.lower()
    base_dir = os.path.dirname(os.path.abspath(path))
    warnings = []
    errors = []

    if lower.endswith(".vmat_c"):
        from viewer.vmat_parser import is_vmat_c_file, parse_vmat_c
        if not is_vmat_c_file(path):
            return {"stats": {}, "warnings": [], "errors": ["not a .vmat_c file (version check failed)"]}, "vmat_c"
        mat = parse_vmat_c(path)
        if not mat.shader:
            warnings.append("no shader name")
        if not mat.texture_params:
            warnings.append("no texture params")
        _check_texture_refs(base_dir, mat.texture_params, warnings)
        stats = {
            "name": mat.name, "shader": mat.shader,
            "texture_params": len(mat.texture_params),
            "int_params": len(mat.int_params),
            "float_params": len(mat.float_params),
            "vector_params": len(mat.vector_params),
        }
        return {"stats": stats, "warnings": warnings, "errors": errors}, "vmat_c"

    from viewer.vmt_parser import parse_vmt
    shader, params = parse_vmt(path)
    if not shader:
        errors.append("no shader name")
    base_keys = [k for k in params if "basetexture" in k or "diffuse" in k or "albedo" in k]
    if not base_keys:
        warnings.append("no $basetexture-like param (untextured or vertex-lit fallback?)")
    _check_texture_refs(base_dir, {k: v for k, v in params.items() if v and ("/" in v or v.lower().endswith((".vtf", ".vtex_c")))}, warnings)
    stats = {"shader": shader, "params": len(params), "keys": sorted(params)}
    return {"stats": stats, "warnings": warnings, "errors": errors}, "vmt"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check Source material files headlessly.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    return run_checks(args, check_one, EXTENSIONS)


if __name__ == "__main__":
    sys.exit(main())
