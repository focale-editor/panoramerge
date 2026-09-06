import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/src/image/float_raster.dart';
import 'package:panoramerge/src/image/panorama_raster.dart';

/// Direction in which a minimum-cost seam traverses an overlap.
enum SeamOrientation {
  /// No overlap required a seam.
  none,

  /// The path runs from top to bottom and chooses one x coordinate per row.
  vertical,

  /// The path runs from left to right and chooses one y coordinate per column.
  horizontal,
}

/// Controls photometric costs used by dynamic seam selection.
final class SeamFinderOptions {
  /// Contribution of local edge disagreement to the colour cost.
  final double gradientWeight;

  /// Cost assigned where a proposed seam leaves the true overlap.
  final double invalidPixelCost;

  /// Creates validated seam-finder options.
  const SeamFinderOptions({
    this.gradientWeight = 0.25,
    this.invalidPixelCost = 1000000,
  }) : assert(gradientWeight >= 0, 'gradientWeight cannot be negative.'),
       assert(invalidPixelCost > 0, 'invalidPixelCost must be positive.');
}

/// Incoming-image ownership weights selected around one overlap seam.
final class SeamMask {
  /// Mask width.
  final int width;

  /// Mask height.
  final int height;

  /// Per-pixel incoming ownership in the range zero through one.
  final Float32List incomingWeights;

  /// Direction used by the selected path.
  final SeamOrientation orientation;

  /// Chosen x or y coordinate for each row or column in the overlap.
  final List<int> path;

  /// Creates an immutable seam result while taking ownership of [incomingWeights].
  SeamMask.takeWeights({
    required this.width,
    required this.height,
    required Float32List incomingWeights,
    required this.orientation,
    required List<int> path,
  }) : incomingWeights = incomingWeights.asUnmodifiableView(),
       path = List<int>.unmodifiable(path) {
    if (width < 1 || height < 1 || incomingWeights.length != width * height) {
      throw ArgumentError('Seam-mask dimensions do not match their weights.');
    }
  }
}

/// Finds low-error overlap seams through dynamic programming.
final class DynamicProgrammingSeamFinder {
  /// Photometric cost tuning.
  final SeamFinderOptions options;

  /// Creates a seam finder with immutable [options].
  const DynamicProgrammingSeamFinder({this.options = const SeamFinderOptions()});

  /// Finds incoming ownership between two canonical, canvas-aligned rasters.
  SeamMask find(PanoramaRaster existing, PanoramaRaster incoming) {
    if (existing.width != incoming.width || existing.height != incoming.height || existing.pixelFormat != incoming.pixelFormat) {
      throw ArgumentError('Seam inputs must have equal dimensions and pixel formats.');
    }
    return findFloat(
      FloatRaster.fromRaster(existing),
      FloatRaster.fromRaster(incoming),
      colorModel: existing.pixelFormat.colorModel,
    );
  }

  /// Finds incoming ownership without quantizing pipeline-native samples.
  SeamMask findFloat(
    FloatRaster existing,
    FloatRaster incoming, {
    required PanoramaColorModel colorModel,
  }) {
    if (existing.width != incoming.width || existing.height != incoming.height || existing.channelCount != incoming.channelCount) {
      throw ArgumentError('Seam inputs must use identical floating raster layouts.');
    }
    final int width = existing.width;
    final int height = existing.height;
    final Float32List weights = Float32List(width * height);
    final Uint8List coverage = Uint8List(width * height);
    int overlapLeft = width;
    int overlapTop = height;
    int overlapRight = -1;
    int overlapBottom = -1;
    double existingCenterX = 0;
    double existingCenterY = 0;
    double incomingCenterX = 0;
    double incomingCenterY = 0;
    int existingCount = 0;
    int incomingCount = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int pixel = y * width + x;
        final bool hasExisting = existing.alphaAt(x, y) > _alphaEpsilon;
        final bool hasIncoming = incoming.alphaAt(x, y) > _alphaEpsilon;
        if (hasExisting) {
          existingCenterX += x;
          existingCenterY += y;
          existingCount++;
        }
        if (hasIncoming) {
          incomingCenterX += x;
          incomingCenterY += y;
          incomingCount++;
        }
        coverage[pixel] = (hasExisting ? _existingCovered : 0) | (hasIncoming ? _incomingCovered : 0);
        if (hasIncoming && !hasExisting) {
          weights[pixel] = 1;
        }
        if (hasIncoming && hasExisting) {
          overlapLeft = math.min(overlapLeft, x);
          overlapTop = math.min(overlapTop, y);
          overlapRight = math.max(overlapRight, x);
          overlapBottom = math.max(overlapBottom, y);
        }
      }
    }
    if (overlapRight < overlapLeft || overlapBottom < overlapTop) {
      return SeamMask.takeWeights(
        width: width,
        height: height,
        incomingWeights: weights,
        orientation: SeamOrientation.none,
        path: const [],
      );
    }
    final double existingX = existingCount == 0 ? width / 2 : existingCenterX / existingCount;
    final double existingY = existingCount == 0 ? height / 2 : existingCenterY / existingCount;
    final double incomingX = incomingCount == 0 ? width / 2 : incomingCenterX / incomingCount;
    final double incomingY = incomingCount == 0 ? height / 2 : incomingCenterY / incomingCount;
    final int overlapWidth = overlapRight - overlapLeft + 1;
    final int overlapHeight = overlapBottom - overlapTop + 1;
    final Float64List costs = _overlapCosts(
      existing,
      incoming,
      colorModel,
      coverage,
      left: overlapLeft,
      top: overlapTop,
      overlapWidth: overlapWidth,
      overlapHeight: overlapHeight,
      canvasWidth: width,
    );
    final bool vertical = (incomingX - existingX).abs() >= (incomingY - existingY).abs();
    if (vertical) {
      final List<int> path = _verticalPath(
        costs,
        overlapWidth: overlapWidth,
        overlapHeight: overlapHeight,
        origin: overlapLeft,
      );
      final bool incomingOwnsGreater = incomingX >= existingX;
      for (int y = overlapTop; y <= overlapBottom; y++) {
        final int seamX = path[y - overlapTop];
        for (int x = overlapLeft; x <= overlapRight; x++) {
          final int pixel = y * width + x;
          if (coverage[pixel] != _bothCovered) {
            continue;
          }
          weights[pixel] = incomingOwnsGreater == (x >= seamX) ? 1 : 0;
        }
      }
      return SeamMask.takeWeights(
        width: width,
        height: height,
        incomingWeights: weights,
        orientation: SeamOrientation.vertical,
        path: path,
      );
    }
    final List<int> path = _horizontalPath(
      costs,
      overlapWidth: overlapWidth,
      overlapHeight: overlapHeight,
      origin: overlapTop,
    );
    final bool incomingOwnsGreater = incomingY >= existingY;
    for (int x = overlapLeft; x <= overlapRight; x++) {
      final int seamY = path[x - overlapLeft];
      for (int y = overlapTop; y <= overlapBottom; y++) {
        final int pixel = y * width + x;
        if (coverage[pixel] != _bothCovered) {
          continue;
        }
        weights[pixel] = incomingOwnsGreater == (y >= seamY) ? 1 : 0;
      }
    }
    return SeamMask.takeWeights(
      width: width,
      height: height,
      incomingWeights: weights,
      orientation: SeamOrientation.horizontal,
      path: path,
    );
  }

  /// Measures visible-colour and edge disagreement across the whole overlap.
  ///
  /// Both recurrences read every cell once but reach each luminance sample
  /// from up to four neighbours, so converting premultiplied samples inside
  /// the recurrence repeated the same divisions. Luminance is derived once
  /// here, over the overlap plus a one-pixel halo that supplies the central
  /// differences, and the recurrences then read plain cost values.
  Float64List _overlapCosts(
    FloatRaster existing,
    FloatRaster incoming,
    PanoramaColorModel colorModel,
    Uint8List coverage, {
    required int left,
    required int top,
    required int overlapWidth,
    required int overlapHeight,
    required int canvasWidth,
  }) {
    final int haloLeft = math.max(0, left - 1);
    final int haloTop = math.max(0, top - 1);
    final int haloWidth = math.min(existing.width - 1, left + overlapWidth) - haloLeft + 1;
    final int haloHeight = math.min(existing.height - 1, top + overlapHeight) - haloTop + 1;
    final Float64List existingLuminance = Float64List(haloWidth * haloHeight);
    final Float64List incomingLuminance = Float64List(haloWidth * haloHeight);
    for (int y = 0; y < haloHeight; y++) {
      for (int x = 0; x < haloWidth; x++) {
        final int index = y * haloWidth + x;
        existingLuminance[index] = existing.luminanceAt(haloLeft + x, haloTop + y, colorModel);
        incomingLuminance[index] = incoming.luminanceAt(haloLeft + x, haloTop + y, colorModel);
      }
    }
    final int offsetX = left - haloLeft;
    final int offsetY = top - haloTop;
    final Float64List existingGradient = _gradientPlane(
      existingLuminance,
      haloWidth: haloWidth,
      haloHeight: haloHeight,
      offsetX: offsetX,
      offsetY: offsetY,
      overlapWidth: overlapWidth,
      overlapHeight: overlapHeight,
    );
    final Float64List incomingGradient = _gradientPlane(
      incomingLuminance,
      haloWidth: haloWidth,
      haloHeight: haloHeight,
      offsetX: offsetX,
      offsetY: offsetY,
      overlapWidth: overlapWidth,
      overlapHeight: overlapHeight,
    );
    final Float64List costs = Float64List(overlapWidth * overlapHeight);
    for (int y = 0; y < overlapHeight; y++) {
      final int canvasY = top + y;
      for (int x = 0; x < overlapWidth; x++) {
        final int cell = y * overlapWidth + x;
        final int canvasX = left + x;
        if (coverage[canvasY * canvasWidth + canvasX] != _bothCovered) {
          costs[cell] = options.invalidPixelCost;
          continue;
        }
        final (double existingRed, double existingGreen, double existingBlue) = existing.visibleRgbAt(canvasX, canvasY, colorModel);
        final (double incomingRed, double incomingGreen, double incomingBlue) = incoming.visibleRgbAt(canvasX, canvasY, colorModel);
        final double colourDifference = ((existingRed - incomingRed).abs() + (existingGreen - incomingGreen).abs() + (existingBlue - incomingBlue).abs()) / 3;
        costs[cell] = colourDifference + options.gradientWeight * (existingGradient[cell] - incomingGradient[cell]).abs() + _minimumCellCost;
      }
    }
    return costs;
  }

  /// Converts one halo luminance plane into overlap gradient magnitudes.
  ///
  /// Clamping to the halo edge reproduces clamping to the image edge, because
  /// the halo only stops short of a neighbour where the image itself does.
  Float64List _gradientPlane(
    Float64List luminance, {
    required int haloWidth,
    required int haloHeight,
    required int offsetX,
    required int offsetY,
    required int overlapWidth,
    required int overlapHeight,
  }) {
    final Float64List result = Float64List(overlapWidth * overlapHeight);
    for (int y = 0; y < overlapHeight; y++) {
      final int haloY = y + offsetY;
      final int row = haloY * haloWidth;
      final int above = math.max(0, haloY - 1) * haloWidth;
      final int below = math.min(haloHeight - 1, haloY + 1) * haloWidth;
      for (int x = 0; x < overlapWidth; x++) {
        final int haloX = x + offsetX;
        final double horizontal = luminance[row + math.min(haloWidth - 1, haloX + 1)] - luminance[row + math.max(0, haloX - 1)];
        final double vertical = luminance[below + haloX] - luminance[above + haloX];
        result[y * overlapWidth + x] = math.sqrt(horizontal * horizontal + vertical * vertical);
      }
    }
    return result;
  }

  /// Finds a top-to-bottom path choosing one x coordinate per overlap row.
  List<int> _verticalPath(
    Float64List costs, {
    required int overlapWidth,
    required int overlapHeight,
    required int origin,
  }) {
    final Float64List previous = Float64List(overlapWidth);
    final Float64List current = Float64List(overlapWidth);
    final Int8List predecessors = Int8List(overlapWidth * overlapHeight);
    for (int x = 0; x < overlapWidth; x++) {
      previous[x] = costs[x] + _centreBias(x, overlapWidth);
    }
    for (int y = 1; y < overlapHeight; y++) {
      final int row = y * overlapWidth;
      for (int x = 0; x < overlapWidth; x++) {
        int previousX = x;
        double best = previous[x];
        if (x > 0 && previous[x - 1] < best) {
          previousX = x - 1;
          best = previous[x - 1];
        }
        if (x + 1 < overlapWidth && previous[x + 1] < best) {
          previousX = x + 1;
          best = previous[x + 1];
        }
        current[x] = best + costs[row + x] + _centreBias(x, overlapWidth);
        predecessors[row + x] = previousX - x;
      }
      previous.setAll(0, current);
    }
    int bestX = 0;
    for (int x = 1; x < overlapWidth; x++) {
      if (previous[x] < previous[bestX]) {
        bestX = x;
      }
    }
    final List<int> path = List<int>.filled(overlapHeight, origin);
    for (int y = overlapHeight - 1; y >= 0; y--) {
      path[y] = origin + bestX;
      if (y > 0) {
        bestX += predecessors[y * overlapWidth + bestX];
      }
    }
    return path;
  }

  /// Finds a left-to-right path choosing one y coordinate per overlap column.
  List<int> _horizontalPath(
    Float64List costs, {
    required int overlapWidth,
    required int overlapHeight,
    required int origin,
  }) {
    final Float64List previous = Float64List(overlapHeight);
    final Float64List current = Float64List(overlapHeight);
    final Int8List predecessors = Int8List(overlapWidth * overlapHeight);
    for (int y = 0; y < overlapHeight; y++) {
      previous[y] = costs[y * overlapWidth] + _centreBias(y, overlapHeight);
    }
    for (int x = 1; x < overlapWidth; x++) {
      for (int y = 0; y < overlapHeight; y++) {
        int previousY = y;
        double best = previous[y];
        if (y > 0 && previous[y - 1] < best) {
          previousY = y - 1;
          best = previous[y - 1];
        }
        if (y + 1 < overlapHeight && previous[y + 1] < best) {
          previousY = y + 1;
          best = previous[y + 1];
        }
        current[y] = best + costs[y * overlapWidth + x] + _centreBias(y, overlapHeight);
        predecessors[y * overlapWidth + x] = previousY - y;
      }
      previous.setAll(0, current);
    }
    int bestY = 0;
    for (int y = 1; y < overlapHeight; y++) {
      if (previous[y] < previous[bestY]) {
        bestY = y;
      }
    }
    final List<int> path = List<int>.filled(overlapWidth, origin);
    for (int x = overlapWidth - 1; x >= 0; x--) {
      path[x] = origin + bestY;
      if (x > 0) {
        bestY += predecessors[bestY * overlapWidth + x];
      }
    }
    return path;
  }

  /// Breaks equal-cost ties toward the overlap centre without masking evidence.
  double _centreBias(int position, int extent) => (position - (extent - 1) / 2).abs() * 1e-9;

  /// Alpha threshold below which a pixel is treated as uncovered.
  static const double _alphaEpsilon = 1 / 65535;

  /// Coverage flag marking a pixel the existing composite already fills.
  static const int _existingCovered = 1;

  /// Coverage flag marking a pixel the incoming image fills.
  static const int _incomingCovered = 2;

  /// Coverage flags of a pixel both images fill, so a seam may cross it.
  static const int _bothCovered = _existingCovered | _incomingCovered;

  /// Floor keeping every valid overlap cell above zero cost.
  static const double _minimumCellCost = 1e-6;
}
