import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/panoramerge.dart';
import 'package:test/test.dart';

void main() {
  test('curved mappings round-trip finite image points', () {
    // Arrange.
    const List<PanoramaProjection> projections = [
      PanoramaProjection.cylindrical,
      PanoramaProjection.spherical,
    ];

    // Act and assert.
    for (final PanoramaProjection projection in projections) {
      final ProjectionMapping mapping = ProjectionMapping(
        projection: projection,
        width: 320,
        height: 180,
        focalLengthPixels: 260,
      );
      for (final Point2 point in const [
        Point2(x: 10, y: 20),
        Point2(x: 160, y: 90),
        Point2(x: 300, y: 155),
      ]) {
        final Point2 roundTrip = mapping.unproject(mapping.project(point));
        expect(roundTrip.x, closeTo(point.x, 1e-9));
        expect(roundTrip.y, closeTo(point.y, 1e-9));
      }
    }
  });

  test('curved unprojection leaves rays behind the camera uncovered', () {
    // Arrange.
    const int width = 400;
    const int height = 200;
    // A focal length this short makes the canvas span more than half a turn,
    // so its outer columns ask for rays that face away from the camera. The
    // tangent wraps there and would otherwise resample mirrored content.
    const double focalLength = 60;
    final Uint8List opaque = Uint8List(width * height * 4);
    for (int pixel = 0; pixel < width * height; pixel++) {
      opaque[pixel * 4] = 200;
      opaque[pixel * 4 + 1] = 120;
      opaque[pixel * 4 + 2] = 40;
      opaque[pixel * 4 + 3] = 255;
    }
    final PanoramaRaster source = PanoramaRaster.fromStraightRgba8(width: width, height: height, bytes: opaque);
    const ProjectionMapping cylindrical = ProjectionMapping(
      projection: PanoramaProjection.cylindrical,
      width: width,
      height: height,
      focalLengthPixels: focalLength,
    );
    const ProjectionMapping spherical = ProjectionMapping(
      projection: PanoramaProjection.spherical,
      width: width,
      height: height,
      focalLengthPixels: focalLength,
    );

    // Act.
    final Uint8List projected = const ImageProjector()
        .project(
          source,
          projection: PanoramaProjection.cylindrical,
          focalLengthPixels: focalLength,
        )
        .raster
        .toStraightRgba8();

    // Assert.
    int wrappedColumns = 0;
    for (int x = 0; x < width; x++) {
      if (((x - (width - 1) / 2) / focalLength).abs() < math.pi / 2) {
        continue;
      }
      wrappedColumns++;
      expect(cylindrical.unproject(Point2(x: x.toDouble(), y: height / 2)).isFinite, isFalse);
      for (int y = 0; y < height; y++) {
        expect(projected[(y * width + x) * 4 + 3], 0, reason: 'column $x lies beyond a quarter turn');
      }
    }
    expect(wrappedColumns, greaterThan(0));
    // Spherical mapping wraps on the vertical axis as well.
    int wrappedRows = 0;
    for (int y = 0; y < height; y++) {
      if (((y - (height - 1) / 2) / focalLength).abs() < math.pi / 2) {
        continue;
      }
      wrappedRows++;
      expect(spherical.unproject(Point2(x: width / 2, y: y.toDouble())).isFinite, isFalse);
    }
    expect(wrappedRows, greaterThan(0));
  });

  test('blending leaves canvas beyond the incoming reach untouched', () {
    // Arrange.
    // The blender only reconstructs the incoming footprint widened by the
    // reach of its pyramids, so a canvas much wider than the incoming image
    // must come back carrying the existing samples bit for bit.
    const int width = 400;
    const int height = 60;
    const int incomingLeft = 250;
    const MultibandBlenderOptions blenderOptions = MultibandBlenderOptions(maximumLevels: 3, minimumLevelDimension: 8);
    const int reach = 2 << 3;
    final FloatRaster existing = _covered(width: width, height: height, from: 0, to: 299, tint: 0.8);
    final FloatRaster incoming = _covered(width: width, height: height, from: incomingLeft, to: 399, tint: 0.3);
    final SeamMask seam = const DynamicProgrammingSeamFinder().findFloat(
      existing,
      incoming,
      colorModel: PanoramaColorModel.rgb,
    );

    // Act.
    final FloatRaster blended = const MultibandBlender(options: blenderOptions).blendFloat(
      existing,
      incoming,
      seamMask: seam,
    );

    // Assert.
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < incomingLeft - reach; x++) {
        for (int channel = 0; channel < 4; channel++) {
          final int index = (y * width + x) * 4 + channel;
          expect(blended.components[index], existing.components[index], reason: 'pixel $x, $y channel $channel');
        }
      }
    }
    // The incoming image still owns its own side of the seam.
    expect(blended.components[((height ~/ 2) * width + 380) * 4 + 3], closeTo(1, 1e-6));
  });

  test('seam finder and multiband blender preserve opposite image interiors', () {
    // Arrange.
    final PanoramaRaster existing = _coveredRaster(
      width: 64,
      height: 40,
      firstCoveredX: 0,
      lastCoveredX: 47,
      red: 255,
      blue: 0,
    );
    final PanoramaRaster incoming = _coveredRaster(
      width: 64,
      height: 40,
      firstCoveredX: 16,
      lastCoveredX: 63,
      red: 0,
      blue: 255,
    );
    const DynamicProgrammingSeamFinder finder = DynamicProgrammingSeamFinder();

    // Act.
    final SeamMask seam = finder.find(existing, incoming);
    final PanoramaRaster blended = const MultibandBlender(
      options: MultibandBlenderOptions(maximumLevels: 4, minimumLevelDimension: 8),
    ).blend(existing, incoming, seamMask: seam);
    final Uint8List rgba = blended.toStraightRgba8();

    // Assert.
    expect(seam.orientation, SeamOrientation.vertical);
    expect(seam.path, hasLength(40));
    expect(_pixel(rgba, 64, 4, 20), [255, 0, 0, 255]);
    expect(_pixel(rgba, 64, 59, 20), [0, 0, 255, 255]);
    final List<int> boundary = _pixel(rgba, 64, 32, 20);
    expect(boundary[0], inInclusiveRange(1, 254));
    expect(boundary[2], inInclusiveRange(1, 254));
    expect(boundary[3], 255);
  });

  test('seam finder traverses horizontal overlaps from left to right', () {
    // Arrange.
    final PanoramaRaster existing = _coveredHorizontalRaster(
      width: 40,
      height: 64,
      firstCoveredY: 0,
      lastCoveredY: 47,
      green: 255,
    );
    final PanoramaRaster incoming = _coveredHorizontalRaster(
      width: 40,
      height: 64,
      firstCoveredY: 16,
      lastCoveredY: 63,
      green: 64,
    );

    // Act.
    final SeamMask seam = const DynamicProgrammingSeamFinder().find(
      existing,
      incoming,
    );

    // Assert.
    expect(seam.orientation, SeamOrientation.horizontal);
    expect(seam.path, hasLength(40));
    expect(seam.incomingWeights[4 * 40 + 20], 0);
    expect(seam.incomingWeights[59 * 40 + 20], 1);
  });
}

PanoramaRaster _coveredRaster({
  required int width,
  required int height,
  required int firstCoveredX,
  required int lastCoveredX,
  required int red,
  required int blue,
}) {
  final Uint8List straight = Uint8List(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = firstCoveredX; x <= lastCoveredX; x++) {
      final int offset = (y * width + x) * 4;
      straight[offset] = red;
      straight[offset + 2] = blue;
      straight[offset + 3] = 255;
    }
  }
  return PanoramaRaster.fromStraightRgba8(width: width, height: height, bytes: straight);
}

PanoramaRaster _coveredHorizontalRaster({
  required int width,
  required int height,
  required int firstCoveredY,
  required int lastCoveredY,
  required int green,
}) {
  final Uint8List straight = Uint8List(width * height * 4);
  for (int y = firstCoveredY; y <= lastCoveredY; y++) {
    for (int x = 0; x < width; x++) {
      final int offset = (y * width + x) * 4;
      straight[offset + 1] = green;
      straight[offset + 3] = 255;
    }
  }
  return PanoramaRaster.fromStraightRgba8(
    width: width,
    height: height,
    bytes: straight,
  );
}

List<int> _pixel(Uint8List bytes, int width, int x, int y) {
  final int offset = (y * width + x) * 4;
  return bytes.sublist(offset, offset + 4);
}

/// Builds a premultiplied raster covering one horizontal band of columns.
FloatRaster _covered({
  required int width,
  required int height,
  required int from,
  required int to,
  required double tint,
}) {
  final Float32List components = Float32List(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = from; x <= to; x++) {
      final int offset = (y * width + x) * 4;
      final int hash = (x * 61 ^ y * 149) & 255;
      components[offset] = hash / 255 * tint;
      components[offset + 1] = (hash * 3 & 255) / 255 * tint;
      components[offset + 2] = (hash * 5 & 255) / 255 * tint;
      components[offset + 3] = 1;
    }
  }
  return FloatRaster(width: width, height: height, channelCount: 4, components: components);
}
