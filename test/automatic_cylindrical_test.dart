import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/panoramerge.dart';
import 'package:test/test.dart';

void main() {
  test('automatic mode connects a three-view cylindrical sweep', () {
    // Arrange.
    const int sourceWidth = 180;
    const int sourceHeight = 120;
    const double focalLength = 160;
    final Uint8List world = _texturedWorld(width: 340, height: sourceHeight);
    final List<PanoramaRaster> sources = [
      for (final int offset in const [0, 65, 130])
        _cylindricalView(
          world,
          worldWidth: 340,
          width: sourceWidth,
          height: sourceHeight,
          focalLength: focalLength,
          worldOffset: offset,
        ),
    ];
    const PanoramaStitcher stitcher = PanoramaStitcher(
      options: PanoramaStitcherOptions(
        projection: PanoramaProjection.automatic,
        focalLengthPixels: focalLength,
        registrationMaximumDimension: 120,
        minimumPairMatches: 6,
        minimumPairInliers: 6,
        featureDetector: OrbFeatureDetectorOptions(
          maximumFeatures: 700,
          pyramidLevels: 3,
          distributionCellSize: 24,
        ),
        featureMatcher: FeatureMatcherOptions(
          maximumHammingDistance: 86,
          ratioThreshold: 0.85,
        ),
        blender: MultibandBlenderOptions(
          maximumLevels: 4,
          minimumLevelDimension: 12,
        ),
      ),
    );

    // Act.
    final PanoramaResult result = stitcher.stitch(sources);

    // Assert.
    expect(result.projection, PanoramaProjection.cylindrical);
    expect(result.diagnostics.compositionOrder, hasLength(3));
    expect(
      result.diagnostics.pairwiseRegistrations.where((pair) => pair.accepted),
      hasLength(greaterThanOrEqualTo(2)),
    );
    expect(result.raster.width, inInclusiveRange(270, 300));
    expect(result.raster.height, inInclusiveRange(105, 122));
  });
}

PanoramaRaster _cylindricalView(
  Uint8List world, {
  required int worldWidth,
  required int width,
  required int height,
  required double focalLength,
  required int worldOffset,
}) {
  final ProjectionMapping mapping = ProjectionMapping(
    projection: PanoramaProjection.cylindrical,
    width: width,
    height: height,
    focalLengthPixels: focalLength,
  );
  final Uint8List output = Uint8List(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final Point2 projected = mapping.project(Point2(x: x.toDouble(), y: y.toDouble()));
      final int worldX = (projected.x + worldOffset).round().clamp(0, worldWidth - 1);
      final int worldY = projected.y.round().clamp(0, height - 1);
      final int sourceOffset = (worldY * worldWidth + worldX) * 4;
      final int destinationOffset = (y * width + x) * 4;
      output.setRange(destinationOffset, destinationOffset + 4, world, sourceOffset);
    }
  }
  return PanoramaRaster.fromStraightRgba8(width: width, height: height, bytes: output);
}

Uint8List _texturedWorld({required int width, required int height}) {
  final Uint8List output = Uint8List(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int offset = (y * width + x) * 4;
      final int wave = ((math.sin(x * 0.17) + math.cos(y * 0.23)) * 35).round();
      final int hash = (x * 61 ^ y * 149 ^ x * y * 13) & 255;
      output[offset] = (hash + wave).clamp(0, 255);
      output[offset + 1] = (hash * 3 + x * 7 - wave).clamp(0, 255);
      output[offset + 2] = (hash * 5 + y * 9 + wave).clamp(0, 255);
      output[offset + 3] = 255;
    }
  }
  return output;
}
