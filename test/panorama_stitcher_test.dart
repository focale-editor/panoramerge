import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/panoramerge.dart';
import 'package:test/test.dart';

void main() {
  test('stitches translated crops through the complete pipeline', () {
    // Arrange.
    final List<PanoramaRaster> sources = _translatedSources(PanoramaPixelFormat.rgba8);
    final PanoramaStitcher stitcher = _stitcher(PanoramaProjection.translation);
    final List<PanoramaProgress> progress = [];

    // Act.
    final PanoramaResult result = stitcher.stitch(
      sources,
      onProgress: progress.add,
    );

    // Assert.
    expect(result.raster.width, inInclusiveRange(218, 222));
    expect(result.raster.height, inInclusiveRange(99, 102));
    expect(result.diagnostics.featureCounts.every((count) => count > 20), isTrue);
    expect(result.diagnostics.pairwiseRegistrations.single.accepted, isTrue);
    expect(result.diagnostics.pairwiseRegistrations.single.inlierCount, greaterThan(10));
    expect(progress.last.stage, PanoramaStage.encoding);
    final Point2 mappedLeft = result.sourceTransforms[0].transform(const Point2(x: 100, y: 50));
    final Point2 mappedRight = result.sourceTransforms[1].transform(const Point2(x: 30, y: 50));
    expect(mappedLeft.x, closeTo(mappedRight.x, 1));
    expect(mappedLeft.y, closeTo(mappedRight.y, 1));
  });

  test('stitchAsync preserves RGB16 through projective registration', () async {
    // Arrange.
    const PanoramaPixelFormat format = PanoramaPixelFormat(
      sampleDepth: PanoramaSampleDepth.uint16,
    );
    final List<PanoramaRaster> sources = _translatedSources(format);
    final PanoramaStitcher stitcher = _stitcher(PanoramaProjection.perspective);

    // Act.
    final PanoramaResult result = await stitcher.stitchAsync(sources);

    // Assert.
    expect(result.raster.pixelFormat, format);
    expect(result.raster.width, inInclusiveRange(218, 222));
    expect(result.raster.bytes.lengthInBytes, result.raster.width * result.raster.height * 8);
    expect(result.diagnostics.pairwiseRegistrations.single.inlierCount, greaterThan(10));
  });

  test('ORB and homography align rotated perspective views', () {
    // Arrange.
    final List<PanoramaRaster> sources = _rotatedPerspectiveSources();

    // Act.
    final PanoramaResult result = _stitcher(
      PanoramaProjection.perspective,
    ).stitch(sources);

    // Assert.
    final PairwiseRegistrationDiagnostic registration = result.diagnostics.pairwiseRegistrations.single;
    expect(registration.accepted, isTrue);
    expect(registration.inlierCount, greaterThan(12));
    const Point2 pointInFirst = Point2(x: 130, y: 60);
    const double angle = 0.06;
    final double cosine = math.cos(angle);
    final double sine = math.sin(angle);
    const double worldX = 140;
    const double worldY = 90;
    final Point2 pointInSecond = Point2(
      x: 89.5 + cosine * (worldX - 175) + sine * (worldY - 92),
      y: 64.5 - sine * (worldX - 175) + cosine * (worldY - 92),
    );
    final Point2 mappedFirst = result.sourceTransforms[0].transform(
      pointInFirst,
    );
    final Point2 mappedSecond = result.sourceTransforms[1].transform(
      pointInSecond,
    );
    expect(mappedFirst.x, closeTo(mappedSecond.x, 2.5));
    expect(mappedFirst.y, closeTo(mappedSecond.y, 2.5));
  });

  test('stitches CMYK8 process plates without converting the output to RGB', () {
    // Arrange.
    const PanoramaPixelFormat format = PanoramaPixelFormat(
      colorModel: PanoramaColorModel.cmyk,
    );
    final List<PanoramaRaster> sources = _translatedSources(format);

    // Act.
    final PanoramaResult result = _stitcher(
      PanoramaProjection.translation,
    ).stitch(sources);

    // Assert.
    expect(result.raster.pixelFormat, format);
    expect(result.raster.width, inInclusiveRange(218, 222));
    expect(result.raster.bytes.lengthInBytes, result.raster.width * result.raster.height * 5);
    expect(result.diagnostics.pairwiseRegistrations.single.inlierCount, greaterThan(10));
  });

  test('preserves every remaining canonical high-depth format', () {
    // Arrange.
    const List<PanoramaPixelFormat> formats = [
      PanoramaPixelFormat(sampleDepth: PanoramaSampleDepth.float32),
      PanoramaPixelFormat(
        colorModel: PanoramaColorModel.cmyk,
        sampleDepth: PanoramaSampleDepth.uint16,
      ),
      PanoramaPixelFormat(
        colorModel: PanoramaColorModel.cmyk,
        sampleDepth: PanoramaSampleDepth.float32,
      ),
    ];

    for (final PanoramaPixelFormat format in formats) {
      // Act.
      final PanoramaResult result = _stitcher(
        PanoramaProjection.translation,
      ).stitch(_translatedSources(format));

      // Assert.
      expect(result.raster.pixelFormat, format, reason: '$format was changed');
      expect(result.raster.width, inInclusiveRange(218, 222));
      expect(
        result.raster.bytes.lengthInBytes,
        result.raster.width * result.raster.height * format.bytesPerPixel,
      );
    }
  });

  test('rejects mixed canonical pixel formats before registration', () {
    // Arrange.
    final PanoramaRaster rgb8 = _solidRaster(
      PanoramaPixelFormat.rgba8,
      width: 50,
      height: 50,
    );
    final PanoramaRaster rgb16 = _solidRaster(
      const PanoramaPixelFormat(sampleDepth: PanoramaSampleDepth.uint16),
      width: 50,
      height: 50,
    );

    // Act and assert.
    expect(
      () => const PanoramaStitcher().stitch([rgb8, rgb16]),
      throwsA(
        isA<PanoramaException>().having(
          (error) => error.code,
          'code',
          PanoramaFailureCode.invalidInput,
        ),
      ),
    );
  });
}

PanoramaStitcher _stitcher(PanoramaProjection projection) => PanoramaStitcher(
  options: PanoramaStitcherOptions(
    projection: projection,
    minimumPairMatches: 6,
    minimumPairInliers: 6,
    featureDetector: const OrbFeatureDetectorOptions(
      maximumFeatures: 600,
      pyramidLevels: 3,
      distributionCellSize: 24,
    ),
    featureMatcher: const FeatureMatcherOptions(
      maximumHammingDistance: 72,
      ratioThreshold: 0.82,
    ),
    blender: const MultibandBlenderOptions(
      maximumLevels: 4,
      minimumLevelDimension: 12,
    ),
  ),
);

List<PanoramaRaster> _translatedSources(PanoramaPixelFormat format) {
  final Uint8List world = _texturedWorld(width: 220, height: 100);
  final List<PanoramaRaster> rgba8 = [
    PanoramaRaster.fromStraightRgba8(
      width: 150,
      height: 100,
      bytes: _crop(world, sourceWidth: 220, left: 0, width: 150, height: 100),
    ),
    PanoramaRaster.fromStraightRgba8(
      width: 150,
      height: 100,
      bytes: _crop(world, sourceWidth: 220, left: 70, width: 150, height: 100),
    ),
  ];
  if (format == PanoramaPixelFormat.rgba8) {
    return rgba8;
  }
  if (format.colorModel == PanoramaColorModel.cmyk) {
    return [for (final PanoramaRaster raster in rgba8) _toCmyk(raster, format)];
  }
  return [
    for (final PanoramaRaster raster in rgba8)
      PanoramaRaster.fromPremultipliedComponents(
        width: raster.width,
        height: raster.height,
        pixelFormat: format,
        components: raster.toPremultipliedComponents(),
      ),
  ];
}

List<PanoramaRaster> _rotatedPerspectiveSources() {
  const int worldWidth = 280;
  const int sourceWidth = 180;
  const int sourceHeight = 130;
  const double angle = 0.06;
  final Uint8List world = _texturedWorld(
    width: worldWidth,
    height: 190,
  );
  final PanoramaRaster first = PanoramaRaster.fromStraightRgba8(
    width: sourceWidth,
    height: sourceHeight,
    bytes: _crop(
      world,
      sourceWidth: worldWidth,
      left: 10,
      top: 30,
      width: sourceWidth,
      height: sourceHeight,
    ),
  );
  final Uint8List secondBytes = Uint8List(sourceWidth * sourceHeight * 4);
  final double cosine = math.cos(angle);
  final double sine = math.sin(angle);
  for (int y = 0; y < sourceHeight; y++) {
    for (int x = 0; x < sourceWidth; x++) {
      final double worldX = 175 + cosine * (x - 89.5) - sine * (y - 64.5);
      final double worldY = 92 + sine * (x - 89.5) + cosine * (y - 64.5);
      final int left = worldX.floor();
      final int top = worldY.floor();
      final int right = left + 1;
      final int bottom = top + 1;
      final double fractionX = worldX - left;
      final double fractionY = worldY - top;
      final int destinationOffset = (y * sourceWidth + x) * 4;
      for (int channel = 0; channel < 3; channel++) {
        final double upper = world[(top * worldWidth + left) * 4 + channel] * (1 - fractionX) + world[(top * worldWidth + right) * 4 + channel] * fractionX;
        final double lower = world[(bottom * worldWidth + left) * 4 + channel] * (1 - fractionX) + world[(bottom * worldWidth + right) * 4 + channel] * fractionX;
        secondBytes[destinationOffset + channel] = (upper * (1 - fractionY) + lower * fractionY).round();
      }
      secondBytes[destinationOffset + 3] = 255;
    }
  }
  return [
    first,
    PanoramaRaster.fromStraightRgba8(
      width: sourceWidth,
      height: sourceHeight,
      bytes: secondBytes,
    ),
  ];
}

PanoramaRaster _toCmyk(PanoramaRaster source, PanoramaPixelFormat format) {
  final Float32List rgba = source.toPremultipliedComponents();
  final Float32List cmyka = Float32List(source.pixelCount * 5);
  for (int pixel = 0; pixel < source.pixelCount; pixel++) {
    final int rgbOffset = pixel * 4;
    final int cmykOffset = pixel * 5;
    final double alpha = rgba[rgbOffset + 3];
    final double red = alpha > 0 ? rgba[rgbOffset] / alpha : 0;
    final double green = alpha > 0 ? rgba[rgbOffset + 1] / alpha : 0;
    final double blue = alpha > 0 ? rgba[rgbOffset + 2] / alpha : 0;
    final double black = 1 - [red, green, blue].reduce((first, second) => first > second ? first : second);
    final double denominator = 1 - black;
    cmyka[cmykOffset] = (denominator > 1e-8 ? (1 - red - black) / denominator : 0) * alpha;
    cmyka[cmykOffset + 1] = (denominator > 1e-8 ? (1 - green - black) / denominator : 0) * alpha;
    cmyka[cmykOffset + 2] = (denominator > 1e-8 ? (1 - blue - black) / denominator : 0) * alpha;
    cmyka[cmykOffset + 3] = black * alpha;
    cmyka[cmykOffset + 4] = alpha;
  }
  return PanoramaRaster.fromPremultipliedComponents(
    width: source.width,
    height: source.height,
    pixelFormat: format,
    components: cmyka,
  );
}

Uint8List _texturedWorld({required int width, required int height}) {
  final Uint8List output = Uint8List(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int offset = (y * width + x) * 4;
      final int hash = (x * 73 ^ y * 151 ^ x * y * 17 ^ (x ~/ 9) * 199) & 255;
      output[offset] = hash;
      output[offset + 1] = (hash * 3 + x * 5 + y) & 255;
      output[offset + 2] = (hash * 7 + y * 11 + x) & 255;
      output[offset + 3] = 255;
    }
  }
  return output;
}

Uint8List _crop(
  Uint8List source, {
  required int sourceWidth,
  required int left,
  int top = 0,
  required int width,
  required int height,
}) {
  final Uint8List output = Uint8List(width * height * 4);
  for (int y = 0; y < height; y++) {
    final int sourceOffset = ((top + y) * sourceWidth + left) * 4;
    final int destinationOffset = y * width * 4;
    output.setRange(
      destinationOffset,
      destinationOffset + width * 4,
      source,
      sourceOffset,
    );
  }
  return output;
}

PanoramaRaster _solidRaster(
  PanoramaPixelFormat format, {
  required int width,
  required int height,
}) {
  final Float32List components = Float32List(width * height * format.channelCount);
  for (int pixel = 0; pixel < width * height; pixel++) {
    final int offset = pixel * format.channelCount;
    for (int channel = 0; channel < format.colorModel.processChannelCount; channel++) {
      components[offset + channel] = 0.5;
    }
    components[offset + format.alphaChannelIndex] = 1;
  }
  return PanoramaRaster.fromPremultipliedComponents(
    width: width,
    height: height,
    pixelFormat: format,
    components: components,
  );
}
