# Panoramerge

Panoramerge is a pure Dart panorama engine. It takes already decoded,
overlapping rasters and returns one registered, seamed, multiband-blended
canonical raster. It has no Flutter, codec, OpenCV, FFI, or platform runtime
dependency.

The implementation includes:

- binomial Gaussian and Laplacian image pyramids;
- multiscale FAST-9 corner detection, intensity-centroid orientation, and a
  deterministic 256-bit rotated BRIEF descriptor;
- brute-force Hamming matching with a nearest-neighbour ratio test and mutual
  cross-check;
- deterministic RANSAC with normalized homography, affine, and translation
  solvers;
- perspective, cylindrical, and spherical source projection;
- a weighted overlap graph and maximum-quality spanning registration tree;
- bilinear inverse warping and RGB exposure compensation;
- minimum-cost vertical or horizontal seams found by dynamic programming; and
- Gaussian-mask, Laplacian-pyramid multiband blending.

## Canonical pixels

`PanoramaRaster` samples are interleaved, premultiplied, and little-endian:

| Colour model | Components                          | Supported depths       |
|--------------|-------------------------------------|------------------------|
| RGB          | red, green, blue, alpha             | uint8, uint16, float32 |
| CMYK         | cyan, magenta, yellow, black, alpha | uint8, uint16, float32 |

Integer samples are unsigned normalized values. Alpha and CMYK ink are bounded
to 0–1. Float32 RGB components may exceed 1, so HDR values survive projection
and blending. Geometry and pyramid operations run on floating premultiplied
components and quantize only when the final raster is encoded.

Colour profiles remain an application concern. All sources passed to one
stitch must use the same pixel format and colour interpretation. Panoramerge
does not rewrite process plates; it uses RGB luminance or a deterministic
generic device-CMYK conversion only to derive registration and seam evidence.
Hosts that already enforce this canonical contract can adopt owned buffers with
`PanoramaRaster.takeTrustedBytes`, avoiding a copy and a second full sample
scan.

## Usage

The package does not decode files. Pass canonical bytes produced by the host
application or construct an RGBA8 raster from straight bytes:

```dart
import 'dart:typed_data';

import 'package:panoramerge/panoramerge.dart';

Future<PanoramaResult> mergeDecodedImages(
  Uint8List firstRgba,
  Uint8List secondRgba, {
  required int width,
  required int height,
}) {
  final PanoramaRaster first = PanoramaRaster.fromStraightRgba8(
    width: width,
    height: height,
    bytes: firstRgba,
  );
  final PanoramaRaster second = PanoramaRaster.fromStraightRgba8(
    width: width,
    height: height,
    bytes: secondRgba,
  );

  return const PanoramaStitcher(
    options: PanoramaStitcherOptions(
      projection: PanoramaProjection.automatic,
      cropTransparentBorders: true,
    ),
  ).stitchAsync([first, second]);
}
```

`stitchAsync` moves CPU work to a Dart isolate. `stitch` is the synchronous
equivalent and accepts a progress callback. A successful `PanoramaResult`
contains the output raster, each original-source-to-output mapping, feature
counts, pairwise match/inlier evidence, the chosen reference source, and the
composition order.
