"""Run every check script over the bundled test files.

Test files live in viewer/tools/testfiles/ (Suzanne models in
.smd/.mdl/.dmx, plus one .vtf per texture format). This is the
quick smoke test after parser changes:

  python viewer/tools/run_testfiles.py
  python viewer/tools/run_testfiles.py --strict
  python -m viewer.tools.run_testfiles

Exit code: 0 all passed, 1 at least one failure.
"""
import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TESTFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testfiles")


def discover(subdir, extensions):
    base = os.path.join(TESTFILES, subdir)
    if not os.path.isdir(base):
        return []
    out = []
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            if os.path.splitext(name)[1].lower().lstrip(".") in extensions:
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def run_section(title, check_one, paths, strict=False):
    print(f"\n== {title} ({len(paths)} file(s)) ==")
    failures = 0
    for path in paths:
        try:
            result, label = check_one(path)
        except Exception as error:
            result = {"stats": {}, "warnings": [], "errors": [f"{type(error).__name__}: {error}"]}
            label = ""
        failed = bool(result.get("errors")) or (strict and bool(result.get("warnings")))
        tag = "FAIL" if result.get("errors") else ("WARN" if result.get("warnings") else "OK")
        rel = os.path.relpath(path, TESTFILES)
        print(f"[{tag}] {rel}" + (f" ({label})" if label else ""))
        for w in result.get("warnings", []):
            print(f"       warning: {w}")
        for e in result.get("errors", []):
            print(f"       error: {e}")
        if failed:
            failures += 1
    return failures, len(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run checks over bundled test files.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args(argv)

    from viewer.tools import materialcheck, modelcheck, texturecheck

    total_failures = 0
    total_files = 0

    sections = [
        ("models", modelcheck.check_one,
         discover("models", {"smd", "vta", "mdl", "vmdl_c", "dmx"})),
        ("textures", texturecheck.check_one,
         discover("materials", {"vtf", "vtex_c"})),
        ("materials", materialcheck.check_one,
         discover("materials", {"vmt", "vmat_c"})),
    ]
    for title, check_one, paths in sections:
        if not paths:
            print(f"\n== {title} (0 file(s), skipped) ==")
            continue
        failures, count = run_section(title, check_one, paths, strict=args.strict)
        total_failures += failures
        total_files += count

    print(f"\n{total_files} file(s) checked, {total_failures} failure(s)")
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
