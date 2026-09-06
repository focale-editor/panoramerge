import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/src/image/panorama_raster.dart';

/// Mutable floating-point premultiplied storage used between pipeline stages.
final class FloatRaster {
  /// Horizontal pixel extent.
  final int width;

  /// Vertical pixel extent.
  final int height;

  /// Number of process-plus-alpha components per pixel.
  final int channelCount;

  /// Interleaved premultiplied samples.
  final Float32List components;

  /// Creates validated floating raster storage.
  FloatRaster({
    required this.width,
    required this.height,
    required this.channelCount,
    required this.components,
  }) {
    if (width < 1 || height < 1 || channelCount < 2 || components.length != width * height * channelCount) {
      throw ArgumentError('Floating raster dimensions do not match their components.');
    }
  }

  /// Creates zero-filled floating raster storage.
  factory FloatRaster.empty({
    required int width,
    required int height,
    required int channelCount,
  }) => FloatRaster(
    width: width,
    height: height,
    channelCount: channelCount,
    components: Float32List(width * height * channelCount),
  );

  /// Decodes one canonical raster.
  factory FloatRaster.fromRaster(PanoramaRaster raster) => FloatRaster(
    width: raster.width,
    height: raster.height,
    channelCount: raster.pixelFormat.channelCount,
    components: raster.toPremultipliedComponents(),
  );

  /// Index of the alpha component.
  int get alphaChannelIndex => channelCount - 1;

  /// Returns alpha at the supplied integer pixel coordinate.
  double alphaAt(int x, int y) => components[(y * width + x) * channelCount + alphaChannelIndex];

  /// Derives visible luminance from an RGB or CMYK pixel.
  double luminanceAt(int x, int y, PanoramaColorModel colorModel) {
    final (double red, double green, double blue) = visibleRgbAt(x, y, colorModel);
    return math.max(0, red * 0.2126 + green * 0.7152 + blue * 0.0722);
  }

  /// Converts one premultiplied process pixel to straight visible RGB.
  (double, double, double) visibleRgbAt(int x, int y, PanoramaColorModel colorModel) {
    final int offset = (y * width + x) * channelCount;
    final double alpha = components[offset + alphaChannelIndex].clamp(0.0, 1.0);
    if (alpha <= 1e-8) {
      return (0, 0, 0);
    }
    final double red;
    final double green;
    final double blue;
    if (colorModel == PanoramaColorModel.rgb) {
      red = components[offset] / alpha;
      green = components[offset + 1] / alpha;
      blue = components[offset + 2] / alpha;
    } else {
      final double cyan = (components[offset] / alpha).clamp(0.0, 1.0);
      final double magenta = (components[offset + 1] / alpha).clamp(0.0, 1.0);
      final double yellow = (components[offset + 2] / alpha).clamp(0.0, 1.0);
      final double black = (components[offset + 3] / alpha).clamp(0.0, 1.0);
      red = (1 - cyan) * (1 - black);
      green = (1 - magenta) * (1 - black);
      blue = (1 - yellow) * (1 - black);
    }
    return (red, green, blue);
  }

  /// Derives an alpha-weighted 8-bit luminance raster.
  GrayImage toGrayscale(PanoramaColorModel colorModel) {
    final Uint8List output = Uint8List(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int pixel = y * width + x;
        final double alpha = components[pixel * channelCount + alphaChannelIndex].clamp(0.0, 1.0);
        output[pixel] = (luminanceAt(x, y, colorModel).clamp(0.0, 1.0) * alpha * 255).round();
      }
    }
    return GrayImage.takeBytes(width: width, height: height, bytes: output);
  }
}
