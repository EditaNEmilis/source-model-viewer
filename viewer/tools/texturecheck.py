"""Check Source texture files headlessly (no GUI).

  .vtf      -> parse_vtf (decodes largest mip to RGBA, reports alpha statistics)
  .vtex_c   -> parse_vtex_c (Source 2; compressed BC formats keep blocks, no RGBA)

Usage:
  python viewer/tools/texturecheck.py path/to/texture.vtf
  python viewer/tools/texturecheck.py materials/ -r
  python viewer/tools/texturecheck.py materials/ -r --strict
  python -m viewer.tools.texturecheck path/to/texture.vtex_c

Exit code: 0 ok, 1 check failed, 2 no files matched.
"""
import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viewer.tools._common import add_common_args, is_power_of_two, run_checks

EXTENSIONS = ("vtf", "vtex_c")


def _alpha_stats(rgba):
    if rgba is None or len(rgba) == 0:
        return {}
    alpha = rgba[:, :, 3]
    return {
        "alpha_min": int(alpha.min()),
        "alpha_max": int(alpha.max()),
        "alpha_mean": round(float(alpha.mean()), 2),
        "transparent_px": int((alpha == 0).sum()),
        "opaque_px": int((alpha == 255).sum()),
        "semi_px": int(((alpha > 0) & (alpha < 255)).sum()),
    }


def check_one(path):
    lower = path.lower()
    if lower.endswith(".vtex_c"):
        from viewer.vtex_parser import is_vtex_c_file, parse_vtex_c
        if not is_vtex_c_file(path):
            return {"stats": {}, "warnings": [], "errors": ["not a .vtex_c file (magic check failed)"]}, "vtex_c"
        tex = parse_vtex_c(path)
        warnings = []
        errors = []
        if not tex.width or not tex.height:
            errors.append("zero width/height")
        if tex.rgba is None and not tex.compressed_mips:
            warnings.append("no decoded RGBA and no compressed mips (unsupported format?)")
        stats = {
            "width": tex.width, "height": tex.height,
            "format": tex.format_name, "format_id": tex.format,
            "mip_levels": tex.mip_levels, "version": tex.version,
            "compression": tex.compression,
            "compressed_mips": len(tex.compressed_mips) if tex.compressed_mips else 0,
        }
        stats.update(_alpha_stats(tex.rgba))
        if tex.rgba is not None and tuple(tex.rgba.shape[:2]) != (tex.height, tex.width):
            errors.append(f"decoded RGBA shape {tex.rgba.shape} != ({tex.height}, {tex.width})")
        return {"stats": stats, "warnings": warnings, "errors": errors}, "vtex_c"

    from viewer.vtf_parser import FORMAT_NAMES, parse_vtf
    info, rgba = parse_vtf(path)
    warnings = []
    errors = []
    if not info.width or not info.height:
        errors.append("zero width/height")
    if not (is_power_of_two(info.width) and is_power_of_two(info.height)):
        warnings.append("non-power-of-two dimensions (fine for most Source, breaks old fixed-function)")
    if info.high_res_format not in FORMAT_NAMES:
        errors.append(f"unknown high-res format id {info.high_res_format}")
    if rgba is not None and tuple(rgba.shape[:2]) != (info.height, info.width):
        errors.append(f"decoded RGBA shape {rgba.shape} != ({info.height}, {info.width})")
    stats = {
        "version": f"{info.version_major}.{info.version_minor}",
        "width": info.width, "height": info.height, "depth": info.depth,
        "format": FORMAT_NAMES.get(info.high_res_format, "UNKNOWN"),
        "frames": info.frames, "mipmaps": info.mipmap_count,
        "flags": info.flags, "header_size": info.header_size,
    }
    stats.update(_alpha_stats(rgba))
    return {"stats": stats, "warnings": warnings, "errors": errors}, "vtf"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check Source texture files headlessly.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    return run_checks(args, check_one, EXTENSIONS)


if __name__ == "__main__":
    sys.exit(main())
