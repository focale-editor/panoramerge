import 'dart:typed_data';

import 'package:panoramerge/src/feature/feature.dart';

/// One correspondence between a query and train feature set.
final class FeatureMatch {
  /// Index in the query feature set.
  final int queryIndex;

  /// Index in the train feature set.
  final int trainIndex;

  /// Hamming distance between the two descriptors.
  final int distance;

  /// Creates a descriptor correspondence.
  const FeatureMatch({
    required this.queryIndex,
    required this.trainIndex,
    required this.distance,
  });
}

/// Controls ambiguity rejection in brute-force binary matching.
final class FeatureMatcherOptions {
  /// Largest accepted descriptor distance in bits.
  final int maximumHammingDistance;

  /// Required best-to-second-best distance ratio.
  final double ratioThreshold;

  /// Whether the train descriptor must independently choose the query.
  final bool crossCheck;

  /// Creates validated matcher options.
  const FeatureMatcherOptions({
    this.maximumHammingDistance = 80,
    this.ratioThreshold = 0.8,
    this.crossCheck = true,
  }) : assert(maximumHammingDistance >= 0, 'maximumHammingDistance cannot be negative.'),
       assert(ratioThreshold > 0 && ratioThreshold <= 1, 'ratioThreshold must be in (0, 1].');
}

/// Matches binary descriptors by exhaustive Hamming search.
final class BruteForceFeatureMatcher {
  /// Ambiguity and distance thresholds.
  final FeatureMatcherOptions options;

  /// Creates a matcher with immutable [options].
  const BruteForceFeatureMatcher({this.options = const FeatureMatcherOptions()});

  /// Returns sorted, ratio-tested, optionally mutual correspondences.
  ///
  /// Hamming distance is symmetric, so one pass over the query-by-train
  /// distances feeds both the forward nearest and runner-up search and, when
  /// [FeatureMatcherOptions.crossCheck] is set, the reverse nearest search
  /// that the mutual test needs. Cross-checking therefore costs no extra
  /// descriptor comparisons.
  List<FeatureMatch> match(ImageFeatures query, ImageFeatures train) {
    if (query.descriptors.isEmpty || train.descriptors.isEmpty) {
      return const [];
    }
    final int descriptorLength = query.descriptors.first.bytes.length;
    if (query.descriptors.any((descriptor) => descriptor.bytes.length != descriptorLength) || train.descriptors.any((descriptor) => descriptor.bytes.length != descriptorLength)) {
      throw ArgumentError('Every binary descriptor must use the same length.');
    }
    final bool crossCheck = options.crossCheck;
    final int queryCount = query.descriptors.length;
    final int trainCount = train.descriptors.length;
    final Int32List nearestTrain = Int32List(queryCount);
    final Int32List nearestDistance = Int32List(queryCount);
    final Int32List runnerUpDistance = Int32List(queryCount);
    final Int32List nearestQuery = Int32List(crossCheck ? trainCount : 0)..fillRange(0, crossCheck ? trainCount : 0, -1);
    final Int32List nearestQueryDistance = Int32List(crossCheck ? trainCount : 0)..fillRange(0, crossCheck ? trainCount : 0, _unreachableDistance);
    for (int queryIndex = 0; queryIndex < queryCount; queryIndex++) {
      final BinaryDescriptor descriptor = query.descriptors[queryIndex];
      int nearest = -1;
      int nearestValue = _unreachableDistance;
      int runnerUpValue = _unreachableDistance;
      for (int trainIndex = 0; trainIndex < trainCount; trainIndex++) {
        final int distance = descriptor.distanceTo(train.descriptors[trainIndex]);
        if (distance < nearestValue) {
          runnerUpValue = nearestValue;
          nearestValue = distance;
          nearest = trainIndex;
        } else if (distance < runnerUpValue) {
          runnerUpValue = distance;
        }
        if (crossCheck && distance < nearestQueryDistance[trainIndex]) {
          nearestQueryDistance[trainIndex] = distance;
          nearestQuery[trainIndex] = queryIndex;
        }
      }
      nearestTrain[queryIndex] = nearest;
      nearestDistance[queryIndex] = nearestValue;
      runnerUpDistance[queryIndex] = runnerUpValue;
    }
    final List<FeatureMatch> result = [];
    for (int queryIndex = 0; queryIndex < queryCount; queryIndex++) {
      final int bestIndex = nearestTrain[queryIndex];
      final int bestDistance = nearestDistance[queryIndex];
      final int secondDistance = runnerUpDistance[queryIndex];
      final bool ratioAccepted = secondDistance == _unreachableDistance || bestDistance < secondDistance * options.ratioThreshold;
      if (bestIndex >= 0 && bestDistance <= options.maximumHammingDistance && ratioAccepted && (!crossCheck || nearestQuery[bestIndex] == queryIndex)) {
        result.add(
          FeatureMatch(
            queryIndex: queryIndex,
            trainIndex: bestIndex,
            distance: bestDistance,
          ),
        );
      }
    }
    result.sort((first, second) {
      final int distanceOrder = first.distance.compareTo(second.distance);
      return distanceOrder != 0 ? distanceOrder : first.queryIndex.compareTo(second.queryIndex);
    });
    return List<FeatureMatch>.unmodifiable(result);
  }

  /// Sentinel standing in for a distance no descriptor pair can reach.
  static const int _unreachableDistance = 1 << 30;
}
