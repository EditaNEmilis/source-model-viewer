Skyboxes for the viewport background. Pick one in Settings, Viewport tab. With no skybox selected the viewport uses the solid background color.

Each subfolder here is one skybox:

Cubemap set: six faces sharing a filename prefix, ending in
rt, lf, bk, ft, up or dn, with or without an underscore, for example
sky_day01/sky_day01_01rt.vtf, sky_day01_01lf.vtf, and so on.
Faces can be VTF files or common images (.png, .jpg, .bmp, .webp, .tga).

Panorama: a single image file in the folder (or loose in this
directory), treated as an equirectangular panorama and converted
to cube faces on load.

Coordinates are Z-up: rt is -X, lf is +X, ft is +Y, bk is -Y,
up is +Z, dn is -Z. This mapping was verified against sky_day01:
with it, the shared edges of adjacent faces match exactly.
