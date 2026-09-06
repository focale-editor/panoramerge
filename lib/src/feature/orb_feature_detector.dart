import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/src/core/geometry.dart';
import 'package:panoramerge/src/feature/feature.dart';
import 'package:panoramerge/src/image/panorama_raster.dart';
import 'package:panoramerge/src/pyramid/gaussian_pyramid.dart';

/// Controls scale-space corner detection and binary description.
final class OrbFeatureDetectorOptions {
  /// Maximum retained descriptors across all pyramid levels.
  final int maximumFeatures;

  /// Maximum number of half-resolution pyramid levels.
  final int pyramidLevels;

  /// Primary FAST intensity threshold in byte units.
  final int fastThreshold;

  /// Lower threshold used when an image contains few primary corners.
  final int fallbackFastThreshold;

  /// Full-resolution grid size used for spatial feature distribution.
  final int distributionCellSize;

  /// Creates validated detector options.
  const OrbFeatureDetectorOptions({
    this.maximumFeatures = 1500,
    this.pyramidLevels = 5,
    this.fastThreshold = 18,
    this.fallbackFastThreshold = 7,
    this.distributionCellSize = 48,
  }) : assert(maximumFeatures > 0, 'maximumFeatures must be positive.'),
       assert(pyramidLevels > 0, 'pyramidLevels must be positive.'),
       assert(fastThreshold > 0 && fastThreshold < 256, 'fastThreshold must be between 1 and 255.'),
       assert(fallbackFastThreshold > 0 && fallbackFastThreshold <= fastThreshold, 'fallbackFastThreshold must not exceed fastThreshold.'),
       assert(distributionCellSize > 0, 'distributionCellSize must be positive.');
}

/// Detects oriented FAST corners and describes them with rotated BRIEF bits.
///
/// The implementation follows the ORB family while keeping the complete
/// implementation in Dart: Gaussian scale space, FAST-9, intensity-centroid
/// orientation, and a deterministic 256-bit BRIEF sampling pattern.
final class OrbFeatureDetector {
  /// Detector tuning used for every extraction.
  final OrbFeatureDetectorOptions options;

  /// Creates a detector with immutable [options].
  const OrbFeatureDetector({this.options = const OrbFeatureDetectorOptions()});

  /// Extracts features from a canonical RGB or CMYK raster.
  ImageFeatures detect(PanoramaRaster image) => detectGrayscale(image.toGrayscale());

  /// Extracts features from an already prepared luminance raster.
  ImageFeatures detectGrayscale(GrayImage image) {
    final GaussianPyramid pyramid = GaussianPyramid.build(
      image,
      maximumLevels: options.pyramidLevels,
      minimumDimension: _minimumLevelDimension,
    );
    final List<_CornerCandidate> candidates = [];
    for (int octave = 0; octave < pyramid.levels.length; octave++) {
      final GrayImage level = pyramid.levels[octave];
      List<_LevelCorner> corners = _detectLevel(level, options.fastThreshold);
      final int desiredPerLevel = math.max(24, options.maximumFeatures ~/ pyramid.levels.length);
      if (corners.length < desiredPerLevel ~/ 2 && options.fallbackFastThreshold < options.fastThreshold) {
        corners = _detectLevel(level, options.fallbackFastThreshold);
      }
      corners.sort((first, second) => second.response.compareTo(first.response));
      final int levelLimit = math.min(corners.length, desiredPerLevel * 3);
      final double scale = math.pow(2, octave).toDouble();
      for (int index = 0; index < levelLimit; index++) {
        final _LevelCorner corner = corners[index];
        candidates.add(
          _CornerCandidate(
            x: corner.x,
            y: corner.y,
            octave: octave,
            scale: scale,
            response: corner.response / scale,
          ),
        );
      }
    }
    final List<_CornerCandidate> selected = _distribute(candidates, image.width, image.height);
    final List<FeatureKeypoint> keypoints = [];
    final List<BinaryDescriptor> descriptors = [];
    for (final _CornerCandidate candidate in selected) {
      final GrayImage level = pyramid.levels[candidate.octave];
      final double orientation = _orientation(level, candidate.x, candidate.y);
      keypoints.add(
        FeatureKeypoint(
          position: Point2(x: candidate.x * candidate.scale, y: candidate.y * candidate.scale),
          octave: candidate.octave,
          scale: candidate.scale,
          orientation: orientation,
          response: candidate.response,
        ),
      );
      descriptors.add(_describe(level, candidate.x, candidate.y, orientation));
    }
    return ImageFeatures(keypoints: keypoints, descriptors: descriptors);
  }

  /// Finds nonmaximal FAST-9 corners on one pyramid level.
  List<_LevelCorner> _detectLevel(GrayImage image, int threshold) {
    if (image.width < _minimumLevelDimension || image.height < _minimumLevelDimension) {
      return const [];
    }
    final Float64List responses = Float64List(image.width * image.height);
    for (int y = _descriptorBorder; y < image.height - _descriptorBorder; y++) {
      for (int x = _descriptorBorder; x < image.width - _descriptorBorder; x++) {
        final int fastScore = _fastScore(image, x, y, threshold);
        if (fastScore == 0) {
          continue;
        }
        final double harris = _harrisResponse(image, x, y);
        if (harris > 0) {
          responses[y * image.width + x] = harris + fastScore * 1e6;
        }
      }
    }
    final List<_LevelCorner> result = [];
    for (int y = _descriptorBorder; y < image.height - _descriptorBorder; y++) {
      for (int x = _descriptorBorder; x < image.width - _descriptorBorder; x++) {
        final double response = responses[y * image.width + x];
        if (response <= 0) {
          continue;
        }
        bool maximum = true;
        for (int neighborY = y - 1; neighborY <= y + 1 && maximum; neighborY++) {
          for (int neighborX = x - 1; neighborX <= x + 1; neighborX++) {
            if ((neighborX != x || neighborY != y) && responses[neighborY * image.width + neighborX] > response) {
              maximum = false;
              break;
            }
          }
        }
        if (maximum) {
          result.add(_LevelCorner(x: x, y: y, response: response));
        }
      }
    }
    return result;
  }

  /// Returns a thresholded FAST-9 score, or zero for non-corners.
  int _fastScore(GrayImage image, int x, int y, int threshold) {
    final int center = image.bytes[y * image.width + x];
    int bestScore = 0;
    for (int polarity = -1; polarity <= 1; polarity += 2) {
      for (int start = 0; start < _circle.length; start++) {
        int minimumDifference = 256;
        bool contiguous = true;
        for (int step = 0; step < 9; step++) {
          final _IntegerOffset offset = _circle[(start + step) % _circle.length];
          final int sample = image.bytes[(y + offset.y) * image.width + x + offset.x];
          final int difference = (sample - center) * polarity;
          if (difference <= threshold) {
            contiguous = false;
            break;
          }
          minimumDifference = math.min(minimumDifference, difference);
        }
        if (contiguous) {
          bestScore = math.max(bestScore, minimumDifference);
        }
      }
    }
    return bestScore;
  }

  /// Computes a local Harris response for stable corner ranking.
  ///
  /// Only [_detectLevel] calls this, and it keeps [x] and [y] at least
  /// [_descriptorBorder] pixels from every edge. The five-by-five window and
  /// its central differences reach three pixels at most, so every sample is
  /// already inside the image and needs no per-access clamping.
  double _harrisResponse(GrayImage image, int x, int y) {
    final Uint8List bytes = image.bytes;
    final int width = image.width;
    double xx = 0;
    double xy = 0;
    double yy = 0;
    for (int offsetY = -2; offsetY <= 2; offsetY++) {
      final int row = (y + offsetY) * width + x;
      for (int offsetX = -2; offsetX <= 2; offsetX++) {
        final int sample = row + offsetX;
        final double gradientX = (bytes[sample + 1] - bytes[sample - 1]).toDouble();
        final double gradientY = (bytes[sample + width] - bytes[sample - width]).toDouble();
        xx += gradientX * gradientX;
        xy += gradientX * gradientY;
        yy += gradientY * gradientY;
      }
    }
    final double determinant = xx * yy - xy * xy;
    final double trace = xx + yy;
    return determinant - 0.04 * trace * trace;
  }

  /// Spreads high-response candidates across a full-resolution grid.
  List<_CornerCandidate> _distribute(List<_CornerCandidate> candidates, int width, int height) {
    candidates.sort((first, second) => second.response.compareTo(first.response));
    final int columns = math.max(1, (width / options.distributionCellSize).ceil());
    final int rows = math.max(1, (height / options.distributionCellSize).ceil());
    final List<List<_CornerCandidate>> buckets = List<List<_CornerCandidate>>.generate(columns * rows, (_) => []);
    for (final _CornerCandidate candidate in candidates) {
      final int cellX = math.min(columns - 1, (candidate.x * candidate.scale / options.distributionCellSize).floor());
      final int cellY = math.min(rows - 1, (candidate.y * candidate.scale / options.distributionCellSize).floor());
      buckets[cellY * columns + cellX].add(candidate);
    }
    final List<_CornerCandidate> selected = [];
    int round = 0;
    bool added = true;
    while (selected.length < options.maximumFeatures && added) {
      added = false;
      for (final List<_CornerCandidate> bucket in buckets) {
        if (round < bucket.length) {
          selected.add(bucket[round]);
          added = true;
          if (selected.length == options.maximumFeatures) {
            break;
          }
        }
      }
      round++;
    }
    selected.sort((first, second) => second.response.compareTo(first.response));
    return selected;
  }

  /// Estimates dominant patch orientation through its intensity centroid.
  double _orientation(GrayImage image, int x, int y) {
    double momentX = 0;
    double momentY = 0;
    for (int offsetY = -_orientationRadius; offsetY <= _orientationRadius; offsetY++) {
      final int horizontalRadius = math.sqrt(_orientationRadius * _orientationRadius - offsetY * offsetY).floor();
      for (int offsetX = -horizontalRadius; offsetX <= horizontalRadius; offsetX++) {
        final int intensity = image.bytes[(y + offsetY) * image.width + x + offsetX];
        momentX += offsetX * intensity;
        momentY += offsetY * intensity;
      }
    }
    return math.atan2(momentY, momentX);
  }

  /// Computes the rotated deterministic 256-bit BRIEF descriptor.
  BinaryDescriptor _describe(GrayImage image, int x, int y, double orientation) {
    final double cosine = math.cos(orientation);
    final double sine = math.sin(orientation);
    final Uint8List descriptor = Uint8List(32);
    for (int bit = 0; bit < _briefPattern.length; bit++) {
      final _OffsetPair pair = _briefPattern[bit];
      final double firstX = x + pair.firstX * cosine - pair.firstY * sine;
      final double firstY = y + pair.firstX * sine + pair.firstY * cosine;
      final double secondX = x + pair.secondX * cosine - pair.secondY * sine;
      final double secondY = y + pair.secondX * sine + pair.secondY * cosine;
      if (image.sampleBilinear(firstX, firstY) < image.sampleBilinear(secondX, secondY)) {
        descriptor[bit >> 3] |= 1 << (bit & 7);
      }
    }
    return BinaryDescriptor.takeBytes(bytes: descriptor);
  }

  /// Minimum side required by FAST and rotated descriptor patches.
  static const int _minimumLevelDimension = 48;

  /// Safe edge distance for every rotated BRIEF sample.
  static const int _descriptorBorder = 18;

  /// Radius of the intensity-centroid patch.
  static const int _orientationRadius = 9;

  /// Bresenham circle of radius three used by FAST.
  static const List<_IntegerOffset> _circle = [
    _IntegerOffset(x: 0, y: -3),
    _IntegerOffset(x: 1, y: -3),
    _IntegerOffset(x: 2, y: -2),
    _IntegerOffset(x: 3, y: -1),
    _IntegerOffset(x: 3, y: 0),
    _IntegerOffset(x: 3, y: 1),
    _IntegerOffset(x: 2, y: 2),
    _IntegerOffset(x: 1, y: 3),
    _IntegerOffset(x: 0, y: 3),
    _IntegerOffset(x: -1, y: 3),
    _IntegerOffset(x: -2, y: 2),
    _IntegerOffset(x: -3, y: 1),
    _IntegerOffset(x: -3, y: 0),
    _IntegerOffset(x: -3, y: -1),
    _IntegerOffset(x: -2, y: -2),
    _IntegerOffset(x: -1, y: -3),
  ];

  /// Fixed pseudo-random BRIEF pairs generated inside a radius-12 disk.
  static final List<_OffsetPair> _briefPattern = _createBriefPattern();

  /// Generates a platform-independent descriptor sampling pattern.
  static List<_OffsetPair> _createBriefPattern() {
    int state = 0x6d2b79f5;
    (int, int) nextOffset() {
      while (true) {
        state = (state * 1664525 + 1013904223) & 0xffffffff;
        final int first = ((state >>> 16) % 25) - 12;
        state = (state * 1664525 + 1013904223) & 0xffffffff;
        final int second = ((state >>> 16) % 25) - 12;
        if (first * first + second * second <= 144) {
          return (first, second);
        }
      }
    }

    final List<_OffsetPair> pattern = [];
    while (pattern.length < 256) {
      final (int firstX, int firstY) = nextOffset();
      final (int secondX, int secondY) = nextOffset();
      if (firstX == secondX && firstY == secondY) {
        continue;
      }
      pattern.add(
        _OffsetPair(
          firstX: firstX,
          firstY: firstY,
          secondX: secondX,
          secondY: secondY,
        ),
      );
    }
    return List<_OffsetPair>.unmodifiable(pattern);
  }
}

/// One integer image-space offset.
final class _IntegerOffset {
  /// Horizontal displacement.
  final int x;

  /// Vertical displacement.
  final int y;

  /// Creates an offset.
  const _IntegerOffset({required this.x, required this.y});
}

/// One pair of BRIEF sampling offsets.
final class _OffsetPair {
  /// First horizontal offset.
  final int firstX;

  /// First vertical offset.
  final int firstY;

  /// Second horizontal offset.
  final int secondX;

  /// Second vertical offset.
  final int secondY;

  /// Creates a sampling pair.
  const _OffsetPair({
    required this.firstX,
    required this.firstY,
    required this.secondX,
    required this.secondY,
  });
}

/// One raw corner at a pyramid level.
final class _LevelCorner {
  /// Horizontal level coordinate.
  final int x;

  /// Vertical level coordinate.
  final int y;

  /// Combined corner response.
  final double response;

  /// Creates a level corner.
  const _LevelCorner({required this.x, required this.y, required this.response});
}

/// One corner annotated with its source pyramid scale.
final class _CornerCandidate {
  /// Horizontal level coordinate.
  final int x;

  /// Vertical level coordinate.
  final int y;

  /// Pyramid level index.
  final int octave;

  /// Full-resolution pixels represented by a level pixel.
  final double scale;

  /// Scale-adjusted corner response.
  final double response;

  /// Creates a candidate.
  const _CornerCandidate({
    required this.x,
    required this.y,
    required this.octave,
    required this.scale,
    required this.response,
  });
}
