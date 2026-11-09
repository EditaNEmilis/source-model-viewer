Tools

Headless testing and debugging scripts. No GUI, no PySide6, so they run
in CI or on a dev machine without a display. They reuse the same
parsers as the viewer (viewer/*_parser.py) and the same dispatch as
MainWindow.open_model.

Run from the repo root, either as a script or as a module:

    python viewer/tools/modelcheck.py path/to/model.smd
    python -m viewer.tools.modelcheck path/to/model.mdl
    python viewer/tools/texturecheck.py path/to/texture.vtf
    python viewer/tools/materialcheck.py path/to/mat.vmt
    python viewer/tools/run_testfiles.py

Scripts

- modelcheck.py: .smd/.vta, .mdl (Source 1, GoldSrc, HL2 beta),
  .vmdl_c (Source 2), .dmx. Reports bones, vertices, triangles,
  frames, materials, bounds, and flags non-finite values, broken bone
  parents or cycles, out-of-range triangle indices, degenerate
  triangles, bad skin weights, and UVs outside 0..1.
- texturecheck.py: .vtf and .vtex_c. Reports format, size, mips,
  flags, decoded alpha statistics, and flags unknown formats,
  zero size, and decoded shape mismatches.
- materialcheck.py: .vmt and .vmat_c. Reports shader and param
  counts, and flags missing shader, missing diffuse texture param,
  and texture refs not found next to the material.
- run_testfiles.py: smoke test over viewer/tools/testfiles/
  (Suzanne models and one .vtf per format).

Common flags

- paths: one or more files or folders.
- -r / --recursive: recurse into subfolders.
- -q / --quiet: only print failures.
- --strict: warnings count as failures.
- --json: machine-readable output, one JSON object per file.

Exit codes: 0 = ok, 1 = a check failed, 2 = no files matched.

Adding a check

Parser changes usually only need a new check_one() case in the
matching script. Shared model sanity checks live in _common.py
(check_smd_model), so new formats that parse to SmdModel get them
for free.
