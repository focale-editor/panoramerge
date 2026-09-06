import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/src/image/float_raster.dart';
import 'package:panoramerge/src/image/panorama_raster.dart';
import 'package:panoramerge/src/seam/dynamic_programming_seam_finder.dart';

/// Controls Laplacian-pyramid depth during multiscale blending.
final class MultibandBlenderOptions {
  /// Largest number of pyramid levels, including full resolution.
  final int maximumLevels;

  /// Smallest side length retained before pyramid construction stops.
  final int minimumLevelDimension;

  /// Creates validated blending options.
  const MultibandBlenderOptions({
    this.maximumLevels = 6,
    this.minimumLevelDimension = 16,
  }) : assert(maximumLevels > 0, 'maximumLevels must be positive.'),
       assert(minimumLevelDimension > 1, 'minimumLevelDimension must exceed one.');
}

/// Blends aligned process channels through Gaussian masks and Laplacian bands.
final class MultibandBlender {
  /// Pyramid sizing used for every blend.
  final MultibandBlenderOptions options;

  /// Creates a blender with immutable [options].
  const MultibandBlender({this.options = const MultibandBlenderOptions()});

  /// Blends two canonical rasters according to incoming ownership [seamMask].
  PanoramaRaster blend(
    PanoramaRaster existing,
    PanoramaRaster incoming, {
    required SeamMask seamMask,
  }) {
    if (existing.width != incoming.width || existing.height != incoming.height || existing.pixelFormat != incoming.pixelFormat) {
      throw ArgumentError('Blend inputs must have equal dimensions and pixel formats.');
    }
    final FloatRaster result = blendFloat(
      FloatRaster.fromRaster(existing),
      FloatRaster.fromRaster(incoming),
      seamMask: seamMask,
    );
    return PanoramaRaster.fromPremultipliedComponents(
      width: result.width,
      height: result.height,
      pixelFormat: existing.pixelFormat,
      components: result.components,
    );
  }

  /// Blends floating pipeline rasters without intermediate quantization.
  ///
  /// Reconstruction is confined to the incoming footprint widened by
  /// [_pyramidReach], the furthest a band can travel while the pyramids are
  /// built and collapsed. Beyond it no incoming sample can reach the result,
  /// so those pixels are copied straight from [existing]. A sweep therefore
  /// costs one image rather than one canvas per merge, which matters most
  /// once the canvas has grown far wider than any single source.
  ///
  /// Confining the work does move the pyramid grid, so values inside the
  /// widened footprint differ from a whole-canvas blend by the rounding of
  /// that shift. The widening keeps the difference far from the seam.
  FloatRaster blendFloat(
    FloatRaster existing,
    FloatRaster incoming, {
    required SeamMask seamMask,
  }) {
    if (existing.width != incoming.width ||
        existing.height != incoming.height ||
        existing.channelCount != incoming.channelCount ||
        seamMask.width != existing.width ||
        seamMask.height != existing.height) {
      throw ArgumentError('Floating blend inputs and seam mask must share one layout.');
    }
    final _CanvasRegion? footprint = _incomingFootprint(incoming);
    if (footprint == null) {
      return FloatRaster(
        width: existing.width,
        height: existing.height,
        channelCount: existing.channelCount,
        components: Float32List.fromList(existing.components),
      );
    }
    final _CanvasRegion region = footprint.expanded(
      _pyramidReach,
      canvasWidth: existing.width,
      canvasHeight: existing.height,
    );
    if (region.width == existing.width && region.height == existing.height) {
      return _blendRegion(existing, incoming, seamMask.incomingWeights);
    }
    final FloatRaster blended = _blendRegion(
      _cropRaster(existing, region),
      _cropRaster(incoming, region),
      _cropWeights(seamMask.incomingWeights, existing.width, region),
    );
    final int channelCount = existing.channelCount;
    final Float32List output = Float32List.fromList(existing.components);
    final int rowLength = region.width * channelCount;
    for (int y = 0; y < region.height; y++) {
      final int destination = ((region.top + y) * existing.width + region.left) * channelCount;
      output.setRange(destination, destination + rowLength, blended.components, y * rowLength);
    }
    return FloatRaster(
      width: existing.width,
      height: existing.height,
      channelCount: channelCount,
      components: output,
    );
  }

  /// Returns the smallest region enclosing incoming coverage, or `null`.
  _CanvasRegion? _incomingFootprint(FloatRaster incoming) {
    int left = incoming.width;
    int top = incoming.height;
    int right = -1;
    int bottom = -1;
    for (int y = 0; y < incoming.height; y++) {
      for (int x = 0; x < incoming.width; x++) {
        if (incoming.alphaAt(x, y) <= _weightEpsilon) {
          continue;
        }
        left = math.min(left, x);
        top = math.min(top, y);
        right = math.max(right, x);
        bottom = math.max(bottom, y);
      }
    }
    return right < left ? null : _CanvasRegion(left: left, top: top, width: right - left + 1, height: bottom - top + 1);
  }

  /// Copies one region out of interleaved floating storage.
  FloatRaster _cropRaster(FloatRaster source, _CanvasRegion region) {
    final int channelCount = source.channelCount;
    final int rowLength = region.width * channelCount;
    final Float32List values = Float32List(region.height * rowLength);
    for (int y = 0; y < region.height; y++) {
      final int start = ((region.top + y) * source.width + region.left) * channelCount;
      values.setRange(y * rowLength, (y + 1) * rowLength, source.components, start);
    }
    return FloatRaster(
      width: region.width,
      height: region.height,
      channelCount: channelCount,
      components: values,
    );
  }

  /// Copies one region out of a canvas-wide ownership mask.
  Float32List _cropWeights(Float32List ownership, int canvasWidth, _CanvasRegion region) {
    final Float32List values = Float32List(region.width * region.height);
    for (int y = 0; y < region.height; y++) {
      final int start = (region.top + y) * canvasWidth + region.left;
      values.setRange(y * region.width, (y + 1) * region.width, ownership, start);
    }
    return values;
  }

  /// Reconstructs one complete pair of aligned rasters through Laplacian bands.
  FloatRaster _blendRegion(
    FloatRaster existing,
    FloatRaster incoming,
    Float32List incomingOwnership,
  ) {
    final _FloatLevel existingBase = _FloatLevel(
      width: existing.width,
      height: existing.height,
      channelCount: existing.channelCount,
      values: existing.components,
    );
    final _FloatLevel incomingBase = _FloatLevel(
      width: incoming.width,
      height: incoming.height,
      channelCount: incoming.channelCount,
      values: incoming.components,
    );
    final _FloatLevel existingWeights = _ownershipWeights(
      existing,
      incoming,
      incomingOwnership,
      incomingLayer: false,
    );
    final _FloatLevel incomingWeights = _ownershipWeights(
      existing,
      incoming,
      incomingOwnership,
      incomingLayer: true,
    );
    final int levelCount = _levelCount(existing.width, existing.height);
    final List<_FloatLevel> existingGaussian = _gaussianPyramid(existingBase, levelCount);
    final List<_FloatLevel> incomingGaussian = _gaussianPyramid(incomingBase, levelCount);
    final List<_FloatLevel> existingWeightGaussian = _gaussianPyramid(existingWeights, levelCount);
    final List<_FloatLevel> incomingWeightGaussian = _gaussianPyramid(incomingWeights, levelCount);
    final List<_FloatLevel> existingLaplacian = _laplacianPyramid(existingGaussian);
    final List<_FloatLevel> incomingLaplacian = _laplacianPyramid(incomingGaussian);
    final List<_FloatLevel> blendedBands = [];
    for (int level = 0; level < levelCount; level++) {
      final _FloatLevel first = existingLaplacian[level];
      final _FloatLevel second = incomingLaplacian[level];
      final _FloatLevel firstWeight = existingWeightGaussian[level];
      final _FloatLevel secondWeight = incomingWeightGaussian[level];
      final Float32List values = Float32List(first.values.length);
      for (int pixel = 0; pixel < first.width * first.height; pixel++) {
        final double weightA = firstWeight.values[pixel];
        final double weightB = secondWeight.values[pixel];
        final double sum = weightA + weightB;
        if (sum <= _weightEpsilon) {
          continue;
        }
        final int offset = pixel * first.channelCount;
        for (int channel = 0; channel < first.channelCount; channel++) {
          values[offset + channel] = (first.values[offset + channel] * weightA + second.values[offset + channel] * weightB) / sum;
        }
      }
      blendedBands.add(
        _FloatLevel(
          width: first.width,
          height: first.height,
          channelCount: first.channelCount,
          values: values,
        ),
      );
    }
    _FloatLevel reconstructed = blendedBands.last;
    for (int level = blendedBands.length - 2; level >= 0; level--) {
      final _FloatLevel expanded = _resizeBilinear(
        reconstructed,
        blendedBands[level].width,
        blendedBands[level].height,
      );
      final _FloatLevel band = blendedBands[level];
      final Float32List values = Float32List(band.values.length);
      for (int index = 0; index < values.length; index++) {
        values[index] = expanded.values[index] + band.values[index];
      }
      reconstructed = _FloatLevel(
        width: band.width,
        height: band.height,
        channelCount: band.channelCount,
        values: values,
      );
    }
    final int alphaChannel = reconstructed.channelCount - 1;
    for (int pixel = 0; pixel < reconstructed.width * reconstructed.height; pixel++) {
      final int offset = pixel * reconstructed.channelCount;
      final double exactCoverage = math
          .max(
            existing.components[offset + existing.alphaChannelIndex],
            incoming.components[offset + incoming.alphaChannelIndex],
          )
          .clamp(0.0, 1.0);
      reconstructed.values[offset + alphaChannel] = exactCoverage;
      for (int channel = 0; channel < alphaChannel; channel++) {
        reconstructed.values[offset + channel] = exactCoverage <= _weightEpsilon ? 0 : math.max(0, reconstructed.values[offset + channel]);
      }
    }
    return FloatRaster(
      width: reconstructed.width,
      height: reconstructed.height,
      channelCount: reconstructed.channelCount,
      components: reconstructed.values,
    );
  }

  /// Builds one scalar ownership image from alpha and a hard seam mask.
  _FloatLevel _ownershipWeights(
    FloatRaster existing,
    FloatRaster incoming,
    Float32List incomingOwnership, {
    required bool incomingLayer,
  }) {
    final Float32List values = Float32List(existing.width * existing.height);
    for (int pixel = 0; pixel < values.length; pixel++) {
      final double alphaA = existing.components[pixel * existing.channelCount + existing.alphaChannelIndex].clamp(0.0, 1.0);
      final double alphaB = incoming.components[pixel * incoming.channelCount + incoming.alphaChannelIndex].clamp(0.0, 1.0);
      final double ownership = incomingOwnership[pixel].clamp(0.0, 1.0);
      values[pixel] = incomingLayer ? alphaB * (alphaA > _weightEpsilon ? ownership : 1) : alphaA * (alphaB > _weightEpsilon ? 1 - ownership : 1);
    }
    return _FloatLevel(
      width: existing.width,
      height: existing.height,
      channelCount: 1,
      values: values,
    );
  }

  /// Calculates the number of levels supported by image dimensions.
  int _levelCount(int width, int height) {
    int count = 1;
    int currentWidth = width;
    int currentHeight = height;
    while (count < options.maximumLevels && math.min(currentWidth, currentHeight) >= options.minimumLevelDimension * 2) {
      currentWidth = math.max(1, currentWidth ~/ 2);
      currentHeight = math.max(1, currentHeight ~/ 2);
      count++;
    }
    return count;
  }

  /// Builds a Gaussian pyramid from arbitrary interleaved channels.
  List<_FloatLevel> _gaussianPyramid(_FloatLevel base, int levelCount) {
    final List<_FloatLevel> result = [base];
    while (result.length < levelCount) {
      result.add(_downsample(result.last));
    }
    return result;
  }

  /// Converts Gaussian levels into difference bands and one coarse residual.
  List<_FloatLevel> _laplacianPyramid(List<_FloatLevel> gaussian) {
    final List<_FloatLevel> result = [];
    for (int level = 0; level < gaussian.length - 1; level++) {
      final _FloatLevel current = gaussian[level];
      final _FloatLevel expanded = _resizeBilinear(
        gaussian[level + 1],
        current.width,
        current.height,
      );
      final Float32List difference = Float32List(current.values.length);
      for (int index = 0; index < difference.length; index++) {
        difference[index] = current.values[index] - expanded.values[index];
      }
      result.add(
        _FloatLevel(
          width: current.width,
          height: current.height,
          channelCount: current.channelCount,
          values: difference,
        ),
      );
    }
    result.add(gaussian.last);
    return result;
  }

  /// Applies a five-tap Gaussian and retains every second sample.
  _FloatLevel _downsample(_FloatLevel source) {
    final int outputWidth = math.max(1, source.width ~/ 2);
    final int outputHeight = math.max(1, source.height ~/ 2);
    final Float32List horizontal = Float32List(source.values.length);
    for (int y = 0; y < source.height; y++) {
      for (int x = 0; x < source.width; x++) {
        final int outputOffset = (y * source.width + x) * source.channelCount;
        for (int channel = 0; channel < source.channelCount; channel++) {
          double sum = 0;
          for (int tap = -2; tap <= 2; tap++) {
            final int sampleX = (x + tap).clamp(0, source.width - 1);
            sum += source.values[(y * source.width + sampleX) * source.channelCount + channel] * _kernel[tap + 2];
          }
          horizontal[outputOffset + channel] = sum / 16;
        }
      }
    }
    final Float32List output = Float32List(outputWidth * outputHeight * source.channelCount);
    for (int y = 0; y < outputHeight; y++) {
      final int sourceY = y * 2;
      for (int x = 0; x < outputWidth; x++) {
        final int sourceX = x * 2;
        final int outputOffset = (y * outputWidth + x) * source.channelCount;
        for (int channel = 0; channel < source.channelCount; channel++) {
          double sum = 0;
          for (int tap = -2; tap <= 2; tap++) {
            final int sampleY = (sourceY + tap).clamp(0, source.height - 1);
            sum += horizontal[(sampleY * source.width + sourceX) * source.channelCount + channel] * _kernel[tap + 2];
          }
          output[outputOffset + channel] = sum / 16;
        }
      }
    }
    return _FloatLevel(
      width: outputWidth,
      height: outputHeight,
      channelCount: source.channelCount,
      values: output,
    );
  }

  /// Resizes interleaved floating samples with bilinear interpolation.
  _FloatLevel _resizeBilinear(_FloatLevel source, int width, int height) {
    if (source.width == width && source.height == height) {
      return source;
    }
    final Float32List output = Float32List(width * height * source.channelCount);
    final double horizontalScale = width > 1 ? (source.width - 1) / (width - 1) : 0;
    final double verticalScale = height > 1 ? (source.height - 1) / (height - 1) : 0;
    for (int y = 0; y < height; y++) {
      final double sourceY = y * verticalScale;
      final int y0 = sourceY.floor();
      final int y1 = math.min(y0 + 1, source.height - 1);
      final double fractionY = sourceY - y0;
      for (int x = 0; x < width; x++) {
        final double sourceX = x * horizontalScale;
        final int x0 = sourceX.floor();
        final int x1 = math.min(x0 + 1, source.width - 1);
        final double fractionX = sourceX - x0;
        final int outputOffset = (y * width + x) * source.channelCount;
        final int topLeft = (y0 * source.width + x0) * source.channelCount;
        final int topRight = (y0 * source.width + x1) * source.channelCount;
        final int bottomLeft = (y1 * source.width + x0) * source.channelCount;
        final int bottomRight = (y1 * source.width + x1) * source.channelCount;
        for (int channel = 0; channel < source.channelCount; channel++) {
          final double top = source.values[topLeft + channel] * (1 - fractionX) + source.values[topRight + channel] * fractionX;
          final double bottom = source.values[bottomLeft + channel] * (1 - fractionX) + source.values[bottomRight + channel] * fractionX;
          output[outputOffset + channel] = top * (1 - fractionY) + bottom * fractionY;
        }
      }
    }
    return _FloatLevel(
      width: width,
      height: height,
      channelCount: source.channelCount,
      values: output,
    );
  }

  /// Full-resolution pixels a single band can travel through the pyramids.
  ///
  /// One extra level widens the reach by the five-tap radius of two on the
  /// way down and by the bilinear pair on the way back up, so the reach obeys
  /// `reach(n) = 2 * (reach(n - 1) + 1) + 2` and settles at `2^(n + 1) - 4`.
  /// The shift below rounds that up to `2^(n + 1)`.
  int get _pyramidReach => 2 << math.min(options.maximumLevels, _largestReachExponent);

  /// Five-tap binomial Gaussian kernel.
  static const List<double> _kernel = [1, 4, 6, 4, 1];

  /// Small denominator below which a pyramid pixel has no owner.
  static const double _weightEpsilon = 1e-8;

  /// Level count past which a wider reach already spans any real canvas.
  static const int _largestReachExponent = 24;
}

/// One axis-aligned pixel region of the blending canvas.
final class _CanvasRegion {
  /// Leftmost column in the region.
  final int left;

  /// Topmost row in the region.
  final int top;

  /// Number of columns in the region.
  final int width;

  /// Number of rows in the region.
  final int height;

  /// Creates a region.
  const _CanvasRegion({
    required this.left,
    required this.top,
    required this.width,
    required this.height,
  });

  /// Returns this region grown by [margin] and clipped to the canvas.
  _CanvasRegion expanded(
    int margin, {
    required int canvasWidth,
    required int canvasHeight,
  }) {
    final int grownLeft = math.max(0, left - margin);
    final int grownTop = math.max(0, top - margin);
    return _CanvasRegion(
      left: grownLeft,
      top: grownTop,
      width: math.min(canvasWidth, left + width + margin) - grownLeft,
      height: math.min(canvasHeight, top + height + margin) - grownTop,
    );
  }
}

/// One interleaved floating image-pyramid level.
final class _FloatLevel {
  /// Level width.
  final int width;

  /// Level height.
  final int height;

  /// Interleaved channel count.
  final int channelCount;

  /// Level samples.
  final Float32List values;

  /// Creates one level.
  const _FloatLevel({
    required this.width,
    required this.height,
    required this.channelCount,
    required this.values,
  });
}
