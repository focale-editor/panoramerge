import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/src/core/deterministic_random.dart';
import 'package:panoramerge/src/core/geometry.dart';

/// Selects the transformation family estimated from feature correspondences.
enum GeometricModel {
  /// Eight-degree-of-freedom projective homography.
  homography(minimumSampleSize: 4),

  /// Six-degree-of-freedom affine transform.
  affine(minimumSampleSize: 3),

  /// Two-dimensional translation.
  translation(minimumSampleSize: 1);

  /// Number of correspondences required by a minimal solver.
  final int minimumSampleSize;

  /// Creates a geometric model description.
  const GeometricModel({required this.minimumSampleSize});
}

/// One point correspondence used by geometric estimation.
final class PointMatch {
  /// Point in the coordinate system being transformed.
  final Point2 source;

  /// Corresponding point in the destination coordinate system.
  final Point2 destination;

  /// Optional descriptor distance retained for diagnostics.
  final int descriptorDistance;

  /// Creates one point correspondence.
  const PointMatch({
    required this.source,
    required this.destination,
    this.descriptorDistance = 0,
  });
}

/// Controls robust geometric-model estimation.
final class RansacOptions {
  /// Maximum reprojection distance for an inlier, measured in pixels.
  final double inlierThreshold;

  /// Hard iteration limit before the best model is refined.
  final int maximumIterations;

  /// Probability that adaptive sampling has included an outlier-free subset.
  final double confidence;

  /// Deterministic pseudo-random seed.
  final int randomSeed;

  /// Whether consensus uses forward and inverse transfer error together.
  final bool symmetricTransferError;

  /// Creates validated RANSAC options.
  const RansacOptions({
    this.inlierThreshold = 3,
    this.maximumIterations = 2000,
    this.confidence = 0.995,
    this.randomSeed = 0x51f15e,
    this.symmetricTransferError = true,
  }) : assert(inlierThreshold > 0, 'inlierThreshold must be positive.'),
       assert(maximumIterations > 0, 'maximumIterations must be positive.'),
       assert(confidence > 0 && confidence < 1, 'confidence must be in (0, 1).');
}

/// A robust transform together with the correspondences that support it.
final class GeometricEstimate {
  /// Refined transformation from source to destination coordinates.
  final ProjectiveTransform transform;

  /// Indices of accepted correspondences in the estimator input.
  final List<int> inlierIndices;

  /// Root-mean-square forward reprojection error in pixels.
  final double rootMeanSquareError;

  /// Number of random hypotheses evaluated.
  final int iterations;

  /// Creates an immutable estimation result.
  GeometricEstimate({
    required this.transform,
    required List<int> inlierIndices,
    required this.rootMeanSquareError,
    required this.iterations,
  }) : inlierIndices = List<int>.unmodifiable(inlierIndices);
}

/// Fits translation, affine, or projective transforms with deterministic RANSAC.
final class RansacGeometricEstimator {
  /// Robust-fitting thresholds and limits.
  final RansacOptions options;

  /// Creates an estimator with immutable [options].
  const RansacGeometricEstimator({this.options = const RansacOptions()});

  /// Estimates [model], returning `null` when no nonsingular consensus exists.
  GeometricEstimate? estimate(
    List<PointMatch> matches, {
    GeometricModel model = GeometricModel.homography,
  }) {
    final int sampleSize = model.minimumSampleSize;
    if (matches.length < sampleSize || matches.any((match) => !match.source.isFinite || !match.destination.isFinite)) {
      return null;
    }
    if (matches.length == sampleSize) {
      final ProjectiveTransform? exact = _fit(matches, model);
      if (exact == null) {
        return null;
      }
      final List<int> inliers = _inliers(exact, matches);
      if (inliers.length < sampleSize) {
        return null;
      }
      return GeometricEstimate(
        transform: exact,
        inlierIndices: inliers,
        rootMeanSquareError: _rootMeanSquareError(exact, matches, inliers),
        iterations: 1,
      );
    }

    final DeterministicRandom random = DeterministicRandom(options.randomSeed);
    ProjectiveTransform? bestTransform;
    List<int> bestInliers = const [];
    double bestSquaredError = double.infinity;
    int adaptiveLimit = options.maximumIterations;
    int iterations = 0;
    while (iterations < adaptiveLimit) {
      iterations++;
      final List<PointMatch> sample = _sample(matches, sampleSize, random);
      final ProjectiveTransform? candidate = _fit(sample, model);
      if (candidate == null) {
        continue;
      }
      final List<int> inliers = _inliers(candidate, matches);
      if (inliers.length < sampleSize) {
        continue;
      }
      final double squaredError = _totalSquaredError(candidate, matches, inliers);
      if (inliers.length > bestInliers.length || inliers.length == bestInliers.length && squaredError < bestSquaredError) {
        bestTransform = candidate;
        bestInliers = inliers;
        bestSquaredError = squaredError;
        final double ratio = inliers.length / matches.length;
        final double successProbability = math.pow(ratio, sampleSize).toDouble();
        if (successProbability >= 1 - 1e-12) {
          adaptiveLimit = iterations;
        } else if (successProbability > 1e-12) {
          final double estimate = math.log(1 - options.confidence) / math.log(1 - successProbability);
          if (estimate.isFinite) {
            adaptiveLimit = math.max(iterations, math.min(adaptiveLimit, estimate.ceil()));
          }
        }
      }
    }
    if (bestTransform == null || bestInliers.length < sampleSize) {
      return null;
    }

    ProjectiveTransform refined =
        _fit(
          [for (final int index in bestInliers) matches[index]],
          model,
        ) ??
        bestTransform;
    List<int> refinedInliers = _inliers(refined, matches);
    if (refinedInliers.length >= sampleSize) {
      refined =
          _fit(
            [for (final int index in refinedInliers) matches[index]],
            model,
          ) ??
          refined;
      refinedInliers = _inliers(refined, matches);
    }
    return GeometricEstimate(
      transform: refined,
      inlierIndices: refinedInliers,
      rootMeanSquareError: _rootMeanSquareError(refined, matches, refinedInliers),
      iterations: iterations,
    );
  }

  /// Chooses a model-specific least-squares fit.
  ProjectiveTransform? _fit(List<PointMatch> matches, GeometricModel model) => switch (model) {
    GeometricModel.homography => _fitHomography(matches),
    GeometricModel.affine => _fitAffine(matches),
    GeometricModel.translation => _fitTranslation(matches),
  };

  /// Fits a normalized projective transform with eight unknowns.
  ProjectiveTransform? _fitHomography(List<PointMatch> matches) {
    if (matches.length < 4 || !_hasTwoDimensionalSupport(matches.map((match) => match.source)) || !_hasTwoDimensionalSupport(matches.map((match) => match.destination))) {
      return null;
    }
    final _NormalizedPoints source = _normalize(matches.map((match) => match.source).toList(growable: false));
    final _NormalizedPoints destination = _normalize(matches.map((match) => match.destination).toList(growable: false));
    final List<Float64List> normal = List<Float64List>.generate(8, (_) => Float64List(8));
    final Float64List rightHandSide = Float64List(8);
    for (int index = 0; index < matches.length; index++) {
      final Point2 from = source.points[index];
      final Point2 to = destination.points[index];
      final Float64List rowX = Float64List.fromList([
        from.x,
        from.y,
        1,
        0,
        0,
        0,
        -from.x * to.x,
        -from.y * to.x,
      ]);
      final Float64List rowY = Float64List.fromList([
        0,
        0,
        0,
        from.x,
        from.y,
        1,
        -from.x * to.y,
        -from.y * to.y,
      ]);
      _accumulateNormalEquations(normal, rightHandSide, rowX, to.x);
      _accumulateNormalEquations(normal, rightHandSide, rowY, to.y);
    }
    final Float64List? solution = _solveLinearSystem(normal, rightHandSide);
    if (solution == null) {
      return null;
    }
    final ProjectiveTransform normalized = ProjectiveTransform(
      m00: solution[0],
      m01: solution[1],
      m02: solution[2],
      m10: solution[3],
      m11: solution[4],
      m12: solution[5],
      m20: solution[6],
      m21: solution[7],
      m22: 1,
    );
    final ProjectiveTransform? destinationInverse = destination.transform.inverse();
    if (destinationInverse == null) {
      return null;
    }
    final ProjectiveTransform result = destinationInverse.compose(normalized).compose(source.transform).normalized();
    return result.isFinite && result.inverse() != null ? result : null;
  }

  /// Fits a normalized affine transform with six unknowns.
  ProjectiveTransform? _fitAffine(List<PointMatch> matches) {
    if (matches.length < 3 || !_hasTwoDimensionalSupport(matches.map((match) => match.source)) || !_hasTwoDimensionalSupport(matches.map((match) => match.destination))) {
      return null;
    }
    final _NormalizedPoints source = _normalize(matches.map((match) => match.source).toList(growable: false));
    final _NormalizedPoints destination = _normalize(matches.map((match) => match.destination).toList(growable: false));
    final List<Float64List> normal = List<Float64List>.generate(6, (_) => Float64List(6));
    final Float64List rightHandSide = Float64List(6);
    for (int index = 0; index < matches.length; index++) {
      final Point2 from = source.points[index];
      final Point2 to = destination.points[index];
      final Float64List rowX = Float64List.fromList([from.x, from.y, 1, 0, 0, 0]);
      final Float64List rowY = Float64List.fromList([0, 0, 0, from.x, from.y, 1]);
      _accumulateNormalEquations(normal, rightHandSide, rowX, to.x);
      _accumulateNormalEquations(normal, rightHandSide, rowY, to.y);
    }
    final Float64List? solution = _solveLinearSystem(normal, rightHandSide);
    if (solution == null) {
      return null;
    }
    final ProjectiveTransform normalized = ProjectiveTransform(
      m00: solution[0],
      m01: solution[1],
      m02: solution[2],
      m10: solution[3],
      m11: solution[4],
      m12: solution[5],
      m20: 0,
      m21: 0,
      m22: 1,
    );
    final ProjectiveTransform? destinationInverse = destination.transform.inverse();
    if (destinationInverse == null) {
      return null;
    }
    final ProjectiveTransform result = destinationInverse.compose(normalized).compose(source.transform).normalized();
    return result.isFinite && result.inverse() != null ? result : null;
  }

  /// Fits a robust median translation.
  ProjectiveTransform? _fitTranslation(List<PointMatch> matches) {
    if (matches.isEmpty) {
      return null;
    }
    final List<double> horizontal = [for (final PointMatch match in matches) match.destination.x - match.source.x]..sort();
    final List<double> vertical = [for (final PointMatch match in matches) match.destination.y - match.source.y]..sort();
    final double x = _median(horizontal);
    final double y = _median(vertical);
    return x.isFinite && y.isFinite ? ProjectiveTransform.translation(x: x, y: y) : null;
  }

  /// Accumulates one row into normal equations `A^T A` and `A^T b`.
  void _accumulateNormalEquations(
    List<Float64List> normal,
    Float64List rightHandSide,
    Float64List row,
    double target,
  ) {
    for (int first = 0; first < row.length; first++) {
      rightHandSide[first] += row[first] * target;
      for (int second = 0; second < row.length; second++) {
        normal[first][second] += row[first] * row[second];
      }
    }
  }

  /// Solves a dense square system through partial-pivot Gaussian elimination.
  Float64List? _solveLinearSystem(List<Float64List> matrix, Float64List values) {
    final int size = values.length;
    final List<Float64List> rows = List<Float64List>.generate(size, (row) {
      final Float64List augmented = Float64List(size + 1);
      augmented.setRange(0, size, matrix[row]);
      augmented[size] = values[row];
      return augmented;
    });
    for (int column = 0; column < size; column++) {
      int pivot = column;
      double magnitude = rows[column][column].abs();
      for (int row = column + 1; row < size; row++) {
        final double candidate = rows[row][column].abs();
        if (candidate > magnitude) {
          pivot = row;
          magnitude = candidate;
        }
      }
      if (!magnitude.isFinite || magnitude < 1e-10) {
        return null;
      }
      if (pivot != column) {
        final Float64List swap = rows[column];
        rows[column] = rows[pivot];
        rows[pivot] = swap;
      }
      final double divisor = rows[column][column];
      for (int entry = column; entry <= size; entry++) {
        rows[column][entry] /= divisor;
      }
      for (int row = 0; row < size; row++) {
        if (row == column) {
          continue;
        }
        final double factor = rows[row][column];
        if (factor.abs() < 1e-20) {
          continue;
        }
        for (int entry = column; entry <= size; entry++) {
          rows[row][entry] -= factor * rows[column][entry];
        }
      }
    }
    final Float64List solution = Float64List(size);
    for (int row = 0; row < size; row++) {
      solution[row] = rows[row][size];
      if (!solution[row].isFinite) {
        return null;
      }
    }
    return solution;
  }

  /// Hartley-normalizes points to zero mean and mean distance `sqrt(2)`.
  _NormalizedPoints _normalize(List<Point2> points) {
    double centerX = 0;
    double centerY = 0;
    for (final Point2 point in points) {
      centerX += point.x;
      centerY += point.y;
    }
    centerX /= points.length;
    centerY /= points.length;
    double meanDistance = 0;
    for (final Point2 point in points) {
      meanDistance += math.sqrt((point.x - centerX) * (point.x - centerX) + (point.y - centerY) * (point.y - centerY));
    }
    meanDistance /= points.length;
    final double scale = meanDistance > 1e-12 ? math.sqrt2 / meanDistance : 1;
    final ProjectiveTransform transform = ProjectiveTransform(
      m00: scale,
      m01: 0,
      m02: -centerX * scale,
      m10: 0,
      m11: scale,
      m12: -centerY * scale,
      m20: 0,
      m21: 0,
      m22: 1,
    );
    return _NormalizedPoints(
      points: [for (final Point2 point in points) transform.transform(point)],
      transform: transform,
    );
  }

  /// Returns whether a point set contains a non-collinear triangle.
  bool _hasTwoDimensionalSupport(Iterable<Point2> values) {
    final List<Point2> points = values.toList(growable: false);
    if (points.length < 3) {
      return false;
    }
    final Point2 first = points.first;
    for (int secondIndex = 1; secondIndex < points.length - 1; secondIndex++) {
      final Point2 second = points[secondIndex];
      for (int thirdIndex = secondIndex + 1; thirdIndex < points.length; thirdIndex++) {
        final Point2 third = points[thirdIndex];
        final double twiceArea = (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x);
        if (twiceArea.abs() > 1e-6) {
          return true;
        }
      }
    }
    return false;
  }

  /// Selects distinct correspondences without replacement.
  List<PointMatch> _sample(
    List<PointMatch> matches,
    int count,
    DeterministicRandom random,
  ) {
    final Set<int> indices = {};
    while (indices.length < count) {
      indices.add(random.nextInt(matches.length));
    }
    return [for (final int index in indices) matches[index]];
  }

  /// Collects correspondences within the forward reprojection threshold.
  List<int> _inliers(ProjectiveTransform transform, List<PointMatch> matches) {
    final double squaredThreshold = options.inlierThreshold * options.inlierThreshold;
    final ProjectiveTransform? inverse = options.symmetricTransferError ? transform.inverse() : null;
    if (options.symmetricTransferError && inverse == null) {
      return const [];
    }
    final List<int> result = [];
    for (int index = 0; index < matches.length; index++) {
      final Point2 projected = transform.transform(matches[index].source);
      double squaredError = projected.squaredDistanceTo(matches[index].destination);
      if (inverse != null) {
        final Point2 reverseProjected = inverse.transform(matches[index].destination);
        squaredError = (squaredError + reverseProjected.squaredDistanceTo(matches[index].source)) / 2;
      }
      if (projected.isFinite && squaredError.isFinite && squaredError <= squaredThreshold) {
        result.add(index);
      }
    }
    return result;
  }

  /// Sums forward squared errors for model comparison.
  double _totalSquaredError(
    ProjectiveTransform transform,
    List<PointMatch> matches,
    List<int> indices,
  ) {
    double result = 0;
    for (final int index in indices) {
      result += transform.transform(matches[index].source).squaredDistanceTo(matches[index].destination);
    }
    return result;
  }

  /// Computes forward root-mean-square reprojection error.
  double _rootMeanSquareError(
    ProjectiveTransform transform,
    List<PointMatch> matches,
    List<int> indices,
  ) => indices.isEmpty ? double.infinity : math.sqrt(_totalSquaredError(transform, matches, indices) / indices.length);

  /// Returns the median of an already sorted nonempty list.
  double _median(List<double> sorted) {
    final int middle = sorted.length ~/ 2;
    return sorted.length.isOdd ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
}

/// Normalized points and the original-to-normalized transform.
final class _NormalizedPoints {
  /// Normalized point coordinates.
  final List<Point2> points;

  /// Transform mapping original coordinates to [points].
  final ProjectiveTransform transform;

  /// Creates normalized point storage.
  const _NormalizedPoints({required this.points, required this.transform});
}
