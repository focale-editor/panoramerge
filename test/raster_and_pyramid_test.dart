import 'dart:typed_data';

import 'package:panoramerge/panoramerge.dart';
import 'package:test/test.dart';

void main() {
  group('canonical raster', () {
    test('round-trips premultiplied RGB16 components', () {
      // Arrange.
      const PanoramaPixelFormat format = PanoramaPixelFormat(
        sampleDepth: PanoramaSampleDepth.uint16,
      );
      final Float32List components = Float32List.fromList([
        0.1,
        0.2,
        0.3,
        0.5,
        0.75,
        0.5,
        0.25,
        1,
      ]);

      // Act.
      final PanoramaRaster raster = PanoramaRaster.fromPremultipliedComponents(
        width: 2,
        height: 1,
        pixelFormat: format,
        components: components,
      );
      final Float32List decoded = raster.toPremultipliedComponents();

      // Assert.
      for (int index = 0; index < components.length; index++) {
        expect(decoded[index], closeTo(components[index], 1 / 65535));
      }
    });

    test('retains finite HDR RGB values above display white', () {
      // Arrange.
      const PanoramaPixelFormat format = PanoramaPixelFormat(
        sampleDepth: PanoramaSampleDepth.float32,
      );
      final Float32List components = Float32List.fromList([2.5, 1.25, 0.5, 1]);

      // Act.
      final PanoramaRaster raster = PanoramaRaster.fromPremultipliedComponents(
        width: 1,
        height: 1,
        pixelFormat: format,
        components: components,
      );

      // Assert.
      expect(raster.toPremultipliedComponents(), orderedEquals(components));
      expect(raster.toStraightRgba8(), orderedEquals([255, 255, 128, 255]));
    });

    test('derives visible luminance without altering CMYK plates', () {
      // Arrange.
      const PanoramaPixelFormat format = PanoramaPixelFormat(
        colorModel: PanoramaColorModel.cmyk,
        sampleDepth: PanoramaSampleDepth.uint8,
      );
      final Float32List components = Float32List.fromList([
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
      ]);
      final PanoramaRaster raster = PanoramaRaster.fromPremultipliedComponents(
        width: 2,
        height: 1,
        pixelFormat: format,
        components: components,
      );

      // Act.
      final GrayImage gray = raster.toGrayscale();

      // Assert.
      expect(gray.bytes[0], inInclusiveRange(52, 56));
      expect(gray.bytes[1], inInclusiveRange(180, 184));
      expect(raster.toPremultipliedComponents(), orderedEquals(components));
    });

    test('trusted ownership still validates dimensions and byte length', () {
      // Arrange.
      final Uint8List validBytes = Uint8List.fromList([16, 32, 48, 255]);

      // Act.
      final PanoramaRaster raster = PanoramaRaster.takeTrustedBytes(
        width: 1,
        height: 1,
        pixelFormat: PanoramaPixelFormat.rgba8,
        bytes: validBytes,
      );

      // Assert.
      expect(raster.bytes, orderedEquals(validBytes));
      expect(
        () => PanoramaRaster.takeTrustedBytes(
          width: 2,
          height: 1,
          pixelFormat: PanoramaPixelFormat.rgba8,
          bytes: validBytes,
        ),
        throwsArgumentError,
      );
    });
  });

  test('Gaussian pyramid halves dimensions after low-pass filtering', () {
    // Arrange.
    final Uint8List bytes = Uint8List(64 * 64);
    for (int y = 0; y < 64; y++) {
      for (int x = 0; x < 64; x++) {
        if ((x + y).isEven) {
          bytes[y * 64 + x] = 255;
        }
      }
    }
    final GrayImage source = GrayImage(width: 64, height: 64, bytes: bytes);

    // Act.
    final GaussianPyramid pyramid = GaussianPyramid.build(
      source,
      maximumLevels: 3,
      minimumDimension: 8,
    );

    // Assert.
    expect(pyramid.levels.map((level) => (level.width, level.height)), [
      (64, 64),
      (32, 32),
      (16, 16),
    ]);
    expect(pyramid.levels[1].bytes[10 * 32 + 10], inInclusiveRange(126, 129));
  });
}
