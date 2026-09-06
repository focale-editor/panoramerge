import 'dart:math' as math;
import 'dart:typed_data';

/// Selects the authored process channels stored by a panorama raster.
enum PanoramaColorModel {
  /// Red, green, and blue process components.
  rgb(processChannelCount: 3),

  /// Cyan, magenta, yellow, and black process components.
  cmyk(processChannelCount: 4);

  /// Number of process components, excluding alpha.
  final int processChannelCount;

  /// Creates a process-channel model.
  const PanoramaColorModel({required this.processChannelCount});
}

/// Selects the canonical scalar representation of each sample.
enum PanoramaSampleDepth {
  /// Unsigned normalized 8-bit integer samples.
  uint8(bits: 8, bytesPerSample: 1, floatingPoint: false),

  /// Unsigned normalized little-endian 16-bit integer samples.
  uint16(bits: 16, bytesPerSample: 2, floatingPoint: false),

  /// Little-endian IEEE-754 32-bit floating-point samples.
  float32(bits: 32, bytesPerSample: 4, floatingPoint: true);

  /// User-facing number of bits per sample.
  final int bits;

  /// Number of bytes occupied by one sample.
  final int bytesPerSample;

  /// Whether the representation stores floating-point values.
  final bool floatingPoint;

  /// Creates a sample-depth description.
  const PanoramaSampleDepth({
    required this.bits,
    required this.bytesPerSample,
    required this.floatingPoint,
  });
}

/// Describes interleaved premultiplied process-plus-alpha samples.
final class PanoramaPixelFormat {
  /// Conventional premultiplied RGBA8 storage.
  static const PanoramaPixelFormat rgba8 = PanoramaPixelFormat();

  /// Process colour model represented by each pixel.
  final PanoramaColorModel colorModel;

  /// Scalar representation used by every component.
  final PanoramaSampleDepth sampleDepth;

  /// Creates a canonical pixel format.
  const PanoramaPixelFormat({
    this.colorModel = PanoramaColorModel.rgb,
    this.sampleDepth = PanoramaSampleDepth.uint8,
  });

  /// Total components per pixel, including alpha.
  int get channelCount => colorModel.processChannelCount + 1;

  /// Index of the final alpha component.
  int get alphaChannelIndex => channelCount - 1;

  /// Bytes retained by one complete pixel.
  int get bytesPerPixel => channelCount * sampleDepth.bytesPerSample;

  /// Returns the exact byte count for [width] by [height] pixels.
  int byteLength({required int width, required int height}) => width * height * bytesPerPixel;

  @override
  bool operator ==(Object other) => identical(this, other) || other is PanoramaPixelFormat && colorModel == other.colorModel && sampleDepth == other.sampleDepth;

  @override
  int get hashCode => Object.hash(colorModel, sampleDepth);

  @override
  String toString() => '${colorModel.name}/${sampleDepth.bits}-bit';
}

/// An immutable canonical raster accepted and produced by the stitcher.
///
/// Samples are interleaved, premultiplied, and little-endian. RGB float
/// process samples may exceed one to retain scene-referred high dynamic range;
/// alpha and CMYK ink samples remain bounded to zero through one.
final class PanoramaRaster {
  /// Horizontal pixel extent.
  final int width;

  /// Vertical pixel extent.
  final int height;

  /// Process model and scalar representation of [bytes].
  final PanoramaPixelFormat pixelFormat;

  /// Canonical interleaved premultiplied samples.
  final Uint8List bytes;

  /// Creates a raster and defensively copies [bytes].
  factory PanoramaRaster({
    required int width,
    required int height,
    required PanoramaPixelFormat pixelFormat,
    required Uint8List bytes,
  }) => PanoramaRaster._(
    width: width,
    height: height,
    pixelFormat: pixelFormat,
    bytes: Uint8List.fromList(bytes).asUnmodifiableView(),
  );

  /// Takes ownership of [bytes] without copying their storage.
  ///
  /// The caller must not retain a mutable alias to the supplied list.
  factory PanoramaRaster.takeBytes({
    required int width,
    required int height,
    required PanoramaPixelFormat pixelFormat,
    required Uint8List bytes,
  }) => PanoramaRaster._(
    width: width,
    height: height,
    pixelFormat: pixelFormat,
    bytes: bytes.asUnmodifiableView(),
  );

  /// Takes ownership of canonical bytes supplied by a trusted raster engine.
  ///
  /// Dimensions and byte length are still checked, but the linear scan for
  /// finite, bounded, premultiplied samples is skipped. The caller must have
  /// already enforced those invariants and must not mutate the backing storage
  /// for the lifetime of this raster.
  factory PanoramaRaster.takeTrustedBytes({
    required int width,
    required int height,
    required PanoramaPixelFormat pixelFormat,
    required Uint8List bytes,
  }) => PanoramaRaster._(
    width: width,
    height: height,
    pixelFormat: pixelFormat,
    bytes: bytes.asUnmodifiableView(),
    validateCanonicalSamples: false,
  );

  /// Converts tightly packed straight-alpha RGBA8 bytes to canonical storage.
  factory PanoramaRaster.fromStraightRgba8({
    required int width,
    required int height,
    required Uint8List bytes,
  }) {
    if (width < 1 || height < 1 || bytes.lengthInBytes != width * height * 4) {
      throw ArgumentError('Straight RGBA dimensions do not match their bytes.');
    }
    final Uint8List premultiplied = Uint8List(bytes.lengthInBytes);
    for (int offset = 0; offset < bytes.lengthInBytes; offset += 4) {
      final int alpha = bytes[offset + 3];
      premultiplied[offset] = (bytes[offset] * alpha + 127) ~/ 255;
      premultiplied[offset + 1] = (bytes[offset + 1] * alpha + 127) ~/ 255;
      premultiplied[offset + 2] = (bytes[offset + 2] * alpha + 127) ~/ 255;
      premultiplied[offset + 3] = alpha;
    }
    return PanoramaRaster.takeBytes(
      width: width,
      height: height,
      pixelFormat: PanoramaPixelFormat.rgba8,
      bytes: premultiplied,
    );
  }

  /// Encodes native premultiplied components in [pixelFormat].
  factory PanoramaRaster.fromPremultipliedComponents({
    required int width,
    required int height,
    required PanoramaPixelFormat pixelFormat,
    required Float32List components,
  }) {
    final int expectedComponents = width * height * pixelFormat.channelCount;
    if (width < 1 || height < 1 || components.length != expectedComponents) {
      throw ArgumentError.value(
        components.length,
        'components',
        'Component count does not match the raster dimensions.',
      );
    }
    final Uint8List output = Uint8List(pixelFormat.byteLength(width: width, height: height));
    final PanoramaSampleBuffer destination = PanoramaSampleBuffer(
      output,
      pixelFormat: pixelFormat,
    );
    for (int pixel = 0; pixel < width * height; pixel++) {
      final int componentOffset = pixel * pixelFormat.channelCount;
      final double rawAlpha = components[componentOffset + pixelFormat.alphaChannelIndex];
      final double alpha = (rawAlpha.isFinite ? rawAlpha : 0.0).clamp(0.0, 1.0);
      for (int channel = 0; channel < pixelFormat.colorModel.processChannelCount; channel++) {
        final double raw = components[componentOffset + channel];
        final double finite = raw.isFinite ? math.max(0.0, raw) : 0;
        final double canonical = pixelFormat.colorModel == PanoramaColorModel.cmyk || !pixelFormat.sampleDepth.floatingPoint ? math.min(alpha, finite) : finite;
        destination.writeUnchecked(pixel, channel, canonical);
      }
      destination.writeUnchecked(pixel, pixelFormat.alphaChannelIndex, alpha);
    }
    return PanoramaRaster.takeBytes(
      width: width,
      height: height,
      pixelFormat: pixelFormat,
      bytes: output,
    );
  }

  /// Validates and stores one raster.
  PanoramaRaster._({
    required this.width,
    required this.height,
    required this.pixelFormat,
    required this.bytes,
    bool validateCanonicalSamples = true,
  }) {
    if (width < 1 || height < 1) {
      throw ArgumentError('Raster dimensions must be positive.');
    }
    if (width > 0x3fffffff ~/ height || width * height > 0x3fffffff ~/ pixelFormat.bytesPerPixel) {
      throw ArgumentError('Raster dimensions exceed the supported allocation range.');
    }
    final int expected = pixelFormat.byteLength(width: width, height: height);
    if (bytes.lengthInBytes != expected) {
      throw ArgumentError.value(
        bytes.lengthInBytes,
        'bytes',
        'Expected $expected canonical bytes for a ${width}x$height $pixelFormat raster.',
      );
    }
    if (validateCanonicalSamples) {
      _validateCanonicalSamples();
    }
  }

  /// Number of pixels in the raster.
  int get pixelCount => width * height;

  /// Decodes every canonical sample to premultiplied floating-point storage.
  Float32List toPremultipliedComponents() {
    final PanoramaSampleBuffer source = PanoramaSampleBuffer(bytes, pixelFormat: pixelFormat);
    final Float32List output = Float32List(pixelCount * pixelFormat.channelCount);
    for (int pixel = 0; pixel < pixelCount; pixel++) {
      final int offset = pixel * pixelFormat.channelCount;
      for (int channel = 0; channel < pixelFormat.channelCount; channel++) {
        output[offset + channel] = source.readUnchecked(pixel, channel);
      }
    }
    return output;
  }

  /// Derives alpha-weighted visible luminance for feature registration.
  GrayImage toGrayscale() {
    final PanoramaSampleBuffer source = PanoramaSampleBuffer(bytes, pixelFormat: pixelFormat);
    final Uint8List luminance = Uint8List(pixelCount);
    for (int pixel = 0; pixel < pixelCount; pixel++) {
      final double alpha = source.readUnchecked(pixel, pixelFormat.alphaChannelIndex).clamp(0.0, 1.0);
      if (alpha <= 1e-8) {
        continue;
      }
      final double red;
      final double green;
      final double blue;
      if (pixelFormat.colorModel == PanoramaColorModel.rgb) {
        red = source.readUnchecked(pixel, 0) / alpha;
        green = source.readUnchecked(pixel, 1) / alpha;
        blue = source.readUnchecked(pixel, 2) / alpha;
      } else {
        final double cyan = (source.readUnchecked(pixel, 0) / alpha).clamp(0.0, 1.0);
        final double magenta = (source.readUnchecked(pixel, 1) / alpha).clamp(0.0, 1.0);
        final double yellow = (source.readUnchecked(pixel, 2) / alpha).clamp(0.0, 1.0);
        final double black = (source.readUnchecked(pixel, 3) / alpha).clamp(0.0, 1.0);
        red = (1 - cyan) * (1 - black);
        green = (1 - magenta) * (1 - black);
        blue = (1 - yellow) * (1 - black);
      }
      final double gray = (red * 0.2126 + green * 0.7152 + blue * 0.0722).clamp(0.0, 1.0);
      luminance[pixel] = (gray * alpha * 255).round();
    }
    return GrayImage.takeBytes(width: width, height: height, bytes: luminance);
  }

  /// Produces a display-oriented straight-alpha RGBA8 conversion.
  ///
  /// CMYK uses a deterministic device conversion. RGB
  /// values above display white are clipped; this method is not a tone mapper.
  Uint8List toStraightRgba8() {
    final PanoramaSampleBuffer source = PanoramaSampleBuffer(bytes, pixelFormat: pixelFormat);
    final Uint8List output = Uint8List(pixelCount * 4);
    for (int pixel = 0; pixel < pixelCount; pixel++) {
      final double alpha = source.readUnchecked(pixel, pixelFormat.alphaChannelIndex).clamp(0.0, 1.0);
      final int offset = pixel * 4;
      output[offset + 3] = (alpha * 255).round();
      if (alpha <= 1e-8) {
        continue;
      }
      if (pixelFormat.colorModel == PanoramaColorModel.rgb) {
        for (int channel = 0; channel < 3; channel++) {
          output[offset + channel] = ((source.readUnchecked(pixel, channel) / alpha).clamp(0.0, 1.0) * 255).round();
        }
      } else {
        final double cyan = (source.readUnchecked(pixel, 0) / alpha).clamp(0.0, 1.0);
        final double magenta = (source.readUnchecked(pixel, 1) / alpha).clamp(0.0, 1.0);
        final double yellow = (source.readUnchecked(pixel, 2) / alpha).clamp(0.0, 1.0);
        final double black = (source.readUnchecked(pixel, 3) / alpha).clamp(0.0, 1.0);
        output[offset] = ((1 - cyan) * (1 - black) * 255).round();
        output[offset + 1] = ((1 - magenta) * (1 - black) * 255).round();
        output[offset + 2] = ((1 - yellow) * (1 - black) * 255).round();
      }
    }
    return output;
  }

  /// Crops this raster to the smallest rectangle containing nonzero alpha.
  PanoramaRaster cropToAlpha() {
    final PanoramaSampleBuffer source = PanoramaSampleBuffer(bytes, pixelFormat: pixelFormat);
    int left = width;
    int top = height;
    int right = 0;
    int bottom = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (source.readUnchecked(y * width + x, pixelFormat.alphaChannelIndex) <= 0) {
          continue;
        }
        left = math.min(left, x);
        top = math.min(top, y);
        right = math.max(right, x + 1);
        bottom = math.max(bottom, y + 1);
      }
    }
    if (right <= left || bottom <= top || left == 0 && top == 0 && right == width && bottom == height) {
      return this;
    }
    final int outputWidth = right - left;
    final int outputHeight = bottom - top;
    final int rowBytes = outputWidth * pixelFormat.bytesPerPixel;
    final Uint8List output = Uint8List(outputHeight * rowBytes);
    for (int y = 0; y < outputHeight; y++) {
      final int sourceStart = ((top + y) * width + left) * pixelFormat.bytesPerPixel;
      output.setRange(y * rowBytes, (y + 1) * rowBytes, bytes, sourceStart);
    }
    return PanoramaRaster.takeBytes(
      width: outputWidth,
      height: outputHeight,
      pixelFormat: pixelFormat,
      bytes: output,
    );
  }

  /// Rejects non-finite, out-of-range, or non-premultiplied input samples.
  void _validateCanonicalSamples() {
    final PanoramaSampleBuffer samples = PanoramaSampleBuffer(bytes, pixelFormat: pixelFormat);
    for (int pixel = 0; pixel < pixelCount; pixel++) {
      final double alpha = samples.readUnchecked(pixel, pixelFormat.alphaChannelIndex);
      if (!alpha.isFinite || alpha < 0 || alpha > 1) {
        throw FormatException('Pixel $pixel contains an invalid alpha sample.');
      }
      for (int channel = 0; channel < pixelFormat.colorModel.processChannelCount; channel++) {
        final double value = samples.readUnchecked(pixel, channel);
        final bool mustBeBoundedByAlpha = pixelFormat.colorModel == PanoramaColorModel.cmyk || !pixelFormat.sampleDepth.floatingPoint;
        if (!value.isFinite || value < 0 || mustBeBoundedByAlpha && value > alpha + 1e-6) {
          throw FormatException('Pixel $pixel contains an invalid premultiplied process sample.');
        }
      }
    }
  }
}

/// Provides allocation-free normalized access to canonical raster bytes.
final class PanoramaSampleBuffer {
  /// Canonical byte storage exposed by this view.
  final Uint8List bytes;

  /// Format used to interpret [bytes].
  final PanoramaPixelFormat pixelFormat;

  /// Cached multibyte accessor.
  final ByteData _data;

  /// Creates a view after validating the byte stride.
  PanoramaSampleBuffer(this.bytes, {required this.pixelFormat}) : _data = ByteData.sublistView(bytes) {
    if (bytes.lengthInBytes % pixelFormat.bytesPerPixel != 0) {
      throw ArgumentError('Canonical bytes do not contain complete pixels.');
    }
  }

  /// Number of complete pixels exposed by this view.
  int get pixelCount => bytes.lengthInBytes ~/ pixelFormat.bytesPerPixel;

  /// Reads one normalized sample after validating its coordinates.
  double read(int pixel, int channel) {
    _validatePosition(pixel, channel);
    return readUnchecked(pixel, channel);
  }

  /// Writes one normalized sample after validating its coordinates.
  void write(int pixel, int channel, double value) {
    _validatePosition(pixel, channel);
    writeUnchecked(pixel, channel, value);
  }

  /// Reads a sample whose coordinates were validated by an enclosing loop.
  double readUnchecked(int pixel, int channel) {
    final int offset = pixel * pixelFormat.bytesPerPixel + channel * pixelFormat.sampleDepth.bytesPerSample;
    return switch (pixelFormat.sampleDepth) {
      PanoramaSampleDepth.uint8 => bytes[offset] / 255,
      PanoramaSampleDepth.uint16 => _data.getUint16(offset, Endian.little) / 65535,
      PanoramaSampleDepth.float32 => _data.getFloat32(offset, Endian.little),
    };
  }

  /// Writes a sample whose coordinates were validated by an enclosing loop.
  void writeUnchecked(int pixel, int channel, double value) {
    final int offset = pixel * pixelFormat.bytesPerPixel + channel * pixelFormat.sampleDepth.bytesPerSample;
    final double finite = value.isFinite ? value : 0;
    switch (pixelFormat.sampleDepth) {
      case PanoramaSampleDepth.uint8:
        bytes[offset] = (finite.clamp(0.0, 1.0) * 255).round();
      case PanoramaSampleDepth.uint16:
        _data.setUint16(offset, (finite.clamp(0.0, 1.0) * 65535).round(), Endian.little);
      case PanoramaSampleDepth.float32:
        _data.setFloat32(offset, finite, Endian.little);
    }
  }

  /// Rejects a pixel or channel outside this view.
  void _validatePosition(int pixel, int channel) {
    if (pixel < 0 || pixel >= pixelCount) {
      throw RangeError.range(pixel, 0, pixelCount - 1, 'pixel');
    }
    if (channel < 0 || channel >= pixelFormat.channelCount) {
      throw RangeError.range(channel, 0, pixelFormat.channelCount - 1, 'channel');
    }
  }
}

/// An immutable tightly packed 8-bit grayscale raster.
final class GrayImage {
  /// Horizontal pixel extent.
  final int width;

  /// Vertical pixel extent.
  final int height;

  /// Luminance bytes in row-major order.
  final Uint8List bytes;

  /// Creates an image and defensively copies [bytes].
  factory GrayImage({
    required int width,
    required int height,
    required Uint8List bytes,
  }) => GrayImage._(
    width: width,
    height: height,
    bytes: Uint8List.fromList(bytes).asUnmodifiableView(),
  );

  /// Takes ownership of [bytes] without copying their storage.
  ///
  /// The caller must not retain a mutable alias to the supplied list.
  factory GrayImage.takeBytes({
    required int width,
    required int height,
    required Uint8List bytes,
  }) => GrayImage._(
    width: width,
    height: height,
    bytes: bytes.asUnmodifiableView(),
  );

  /// Validates and stores one grayscale raster.
  GrayImage._({required this.width, required this.height, required this.bytes}) {
    if (width < 1 || height < 1 || bytes.lengthInBytes != width * height) {
      throw ArgumentError('Grayscale dimensions do not match their byte storage.');
    }
  }

  /// Reads one pixel after clamping its coordinates to the image boundary.
  int sampleClamped(int x, int y) {
    final int clampedX = x.clamp(0, width - 1);
    final int clampedY = y.clamp(0, height - 1);
    return bytes[clampedY * width + clampedX];
  }

  /// Reads one bilinearly interpolated pixel after clamping to the boundary.
  double sampleBilinear(double x, double y) {
    final double clampedX = x.clamp(0, width - 1).toDouble();
    final double clampedY = y.clamp(0, height - 1).toDouble();
    final int x0 = clampedX.floor();
    final int y0 = clampedY.floor();
    final int x1 = math.min(x0 + 1, width - 1);
    final int y1 = math.min(y0 + 1, height - 1);
    final double fractionX = clampedX - x0;
    final double fractionY = clampedY - y0;
    final double top = bytes[y0 * width + x0] * (1 - fractionX) + bytes[y0 * width + x1] * fractionX;
    final double bottom = bytes[y1 * width + x0] * (1 - fractionX) + bytes[y1 * width + x1] * fractionX;
    return top * (1 - fractionY) + bottom * fractionY;
  }
}
