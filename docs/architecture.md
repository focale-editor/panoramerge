# Panoramerge architecture

Panoramerge separates photometric storage from registration evidence. The
returned raster stays in the caller's canonical RGB or CMYK component space;
an 8-bit luminance proxy is derived solely for feature detection and seam
costs.

## Pipeline

1. Validate source count, aggregate area, and a common canonical pixel format.
2. Decode canonical samples once to premultiplied float components.
3. Prewarp each source onto its requested camera surface. Planar, affine, and
   translation modes preserve the original sampling plane.
4. Build a half-resolution Gaussian luminance pyramid. At every level, detect
   FAST-9 corners, rank them with a Harris response, distribute them spatially,
   compute intensity-centroid orientation, and extract rotated BRIEF bits.
5. Match likely source pairs by Hamming distance. The ratio test rejects
   ambiguous textures; the mutual check prevents many-to-one correspondences.
6. Estimate a source-to-source transform with deterministic RANSAC. Projective
   and affine least-squares refinement uses Hartley-normalized coordinates.
7. Treat accepted transforms as a weighted overlap graph. Choose the most
   supported source as reference and construct a maximum-quality spanning tree.
8. Transform projected source corners to determine one bounded output canvas,
   then inverse-warp sources with bilinear premultiplied sampling.
9. Optionally equalize reliable RGB overlap means. CMYK plates and HDR samples
   are not implicitly exposure-normalized.
10. Find a minimum-cost colour/gradient seam along the dominant overlap axis.
11. Blend the two sides through Gaussian ownership masks and Laplacian process
    bands. Repeat in spanning-tree order and encode the canonical format once.
12. Crop transparent borders when requested and update every returned source
    transform to the cropped origin.
13. When editable output was requested, crop every projected source and seam
    mask to the same final canvas and transfer them in composition order.

## Coordinate conventions

Feature and transform coordinates address pixel centres with `(0, 0)` at the
centre of the top-left pixel. `ProjectiveTransform.compose(other)` returns
`this * other`: `other` is applied first. Pairwise transforms map the lower
source index to the higher index. `PanoramaSourceTransform.transform` first
applies the nonlinear camera projection and then the projective canvas mapping.

## Projection models

Perspective mode leaves source pixels planar and estimates a homography.
Cylindrical and spherical modes first use the pinhole focal length supplied in
`PanoramaStitcherOptions`; absent metadata, the fallback is `0.8 * max(width,
height)`. Affine mode is suitable for scans that may rotate, scale, or shear.
Translation mode is the strict repositioning path.

Automatic mode uses perspective for two sources and cylindrical projection for
three or more. Applications with focal-length metadata should choose an
explicit curved projection and provide `focalLengthPixels`.

## Failure semantics

Expected registration failures throw `PanoramaException`. Its stable code
distinguishes invalid inputs, insufficient features, disconnected overlap
graphs, unstable transforms, excessive outputs, and excessive estimated
floating working sets. Diagnostic
source indices identify the images that need user attention. Programmer errors
in component-level APIs continue to use `ArgumentError` or `RangeError`.

## Resource behaviour

The high-level stitcher accepts 2–32 ordered sources. For up to six sources it
tests all pairs. Larger jobs test neighbours inside `maximumPairDistance`,
which bounds the otherwise quadratic descriptor search. Registration, warping,
seaming, and blending are deterministic for a fixed option set.

`stitchAsync` uses `Isolate.run`; buffers are transferable by the Dart runtime
but the returned canonical raster remains ordinary managed memory. The
pixel and canonical-byte limits default to Focale's aggregate-input and
single-raster bounds. A separate working-set estimate guards the larger
floating pyramids. High-depth and CMYK rasters consume more working memory
because pyramid arithmetic uses one float per process component.
The estimate also accounts for retained floating projections and seam masks,
canonical source rasters, and the peak copies of quantized one-byte masks when
`includeSourceLayers` is enabled. The default flattened workflow therefore
pays no editable-output allocation cost.

Feature evidence is area-downsampled when a source exceeds
`registrationMaximumDimension` (2400 pixels by default). Returned keypoint
coordinates and RANSAC thresholds are scaled back to full projection pixels,
so this performance bound does not change output geometry.
