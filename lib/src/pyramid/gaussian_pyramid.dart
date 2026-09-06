import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/src/image/panorama_raster.dart';

/// A half-resolution Gaussian image pyramid.
final class GaussianPyramid {
  /// Full-resolution image followed by successively downsampled levels.
  final List<GrayImage> levels;

  /// Builds a pyramid using the separable binomial kernel `[1 4 6 4 1]`.
  factory GaussianPyramid.build(
    GrayImage source, {
    int maximumLevels = 5,
    int minimumDimension = 32,
  }) {
    if (maximumLevels < 1) {
      throw RangeError.range(maximumLevels, 1, null, 'maximumLevels');
    }
    if (minimumDimension < 2) {
      throw RangeError.range(minimumDimension, 2, null, 'minimumDimension');
    }
    final List<GrayImage> levels = [source];
    while (levels.length < maximumLevels) {
      final GrayImage previous = levels.last;
      if (math.min(previous.width, previous.height) < minimumDimension * 2) {
        break;
      }
      levels.add(_downsample(previous));
    }
    return GaussianPyramid._(List<GrayImage>.unmodifiable(levels));
  }

  /// Stores already validated pyramid levels.
  const GaussianPyramid._(this.levels);

  /// Applies one five-tap Gaussian blur without changing image dimensions.
  static GrayImage blur(GrayImage source) {
    final int width = source.width;
    final int height = source.height;
    final Uint16List horizontal = Uint16List(width * height);
    for (int y = 0; y < height; y++) {
      final int row = y * width;
      for (int x = 0; x < width; x++) {
        horizontal[row + x] =
            source.sampleClamped(x - 2, y) + source.sampleClamped(x - 1, y) * 4 + source.sampleClamped(x, y) * 6 + source.sampleClamped(x + 1, y) * 4 + source.sampleClamped(x + 2, y);
      }
    }
    final Uint8List output = Uint8List(width * height);
    for (int y = 0; y < height; y++) {
      final int y0 = (y - 2).clamp(0, height - 1);
      final int y1 = (y - 1).clamp(0, height - 1);
      final int y3 = (y + 1).clamp(0, height - 1);
      final int y4 = (y + 2).clamp(0, height - 1);
      for (int x = 0; x < width; x++) {
        final int value = horizontal[y0 * width + x] + horizontal[y1 * width + x] * 4 + horizontal[y * width + x] * 6 + horizontal[y3 * width + x] * 4 + horizontal[y4 * width + x];
        output[y * width + x] = (value + 128) >> 8;
      }
    }
    return GrayImage.takeBytes(width: width, height: height, bytes: output);
  }

  /// Blurs and halves one pyramid level.
  static GrayImage _downsample(GrayImage source) {
    final GrayImage blurred = blur(source);
    final int width = math.max(1, source.width ~/ 2);
    final int height = math.max(1, source.height ~/ 2);
    final Uint8List output = Uint8List(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        output[y * width + x] = blurred.bytes[y * 2 * source.width + x * 2];
      }
    }
    return GrayImage.takeBytes(width: width, height: height, bytes: output);
  }
}
