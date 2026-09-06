import 'dart:developer' as developer;
import 'dart:typed_data';

import 'package:panoramerge/panoramerge.dart';

/// Builds a small panorama from two overlapping in-memory RGBA images.
Future<void> main() async {
  const int sceneWidth = 220;
  const int imageWidth = 150;
  const int imageHeight = 100;
  final Uint8List scene = _createTexturedScene(
    width: sceneWidth,
    height: imageHeight,
  );
  final List<PanoramaRaster> sources = [
    PanoramaRaster.fromStraightRgba8(
      width: imageWidth,
      height: imageHeight,
      bytes: _cropRgba(
        scene,
        sourceWidth: sceneWidth,
        left: 0,
        width: imageWidth,
        height: imageHeight,
      ),
    ),
    PanoramaRaster.fromStraightRgba8(
      width: imageWidth,
      height: imageHeight,
      bytes: _cropRgba(
        scene,
        sourceWidth: sceneWidth,
        left: 70,
        width: imageWidth,
        height: imageHeight,
      ),
    ),
  ];

  final PanoramaResult result = await const PanoramaStitcher().stitchAsync(
    sources,
  );
  final PairwiseRegistrationDiagnostic registration = result.diagnostics.pairwiseRegistrations.single;

  developer.log(
    'Created a ${result.raster.width} x ${result.raster.height} panorama '
    'from ${sources.length} images with ${registration.inlierCount} inliers.',
    name: 'panoramerge.example',
  );
}

/// Creates deterministic image detail that the feature detector can match.
Uint8List _createTexturedScene({required int width, required int height}) {
  final Uint8List bytes = Uint8List(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int offset = (y * width + x) * 4;
      final int pattern = (x * 73 ^ y * 151 ^ x * y * 17 ^ (x ~/ 9) * 199) & 255;
      bytes[offset] = pattern;
      bytes[offset + 1] = (pattern * 3 + x * 5 + y) & 255;
      bytes[offset + 2] = (pattern * 7 + y * 11 + x) & 255;
      bytes[offset + 3] = 255;
    }
  }
  return bytes;
}

/// Copies one tightly packed rectangular view from an RGBA scene.
Uint8List _cropRgba(
  Uint8List source, {
  required int sourceWidth,
  required int left,
  required int width,
  required int height,
}) {
  final Uint8List output = Uint8List(width * height * 4);
  final int rowLength = width * 4;
  for (int y = 0; y < height; y++) {
    final int sourceOffset = (y * sourceWidth + left) * 4;
    final int destinationOffset = y * rowLength;
    output.setRange(
      destinationOffset,
      destinationOffset + rowLength,
      source,
      sourceOffset,
    );
  }
  return output;
}
