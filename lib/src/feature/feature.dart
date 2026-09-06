import 'dart:typed_data';

import 'package:panoramerge/src/core/geometry.dart';

/// One oriented scale-space feature location.
final class FeatureKeypoint {
  /// Location expressed in full-resolution image coordinates.
  final Point2 position;

  /// Pyramid level in which the point was detected.
  final int octave;

  /// Full-resolution pixels represented by one octave pixel.
  final double scale;

  /// Intensity-centroid orientation in radians.
  final double orientation;

  /// Corner response used to rank competing detections.
  final double response;

  /// Creates one feature location.
  const FeatureKeypoint({
    required this.position,
    required this.octave,
    required this.scale,
    required this.orientation,
    required this.response,
  });
}

/// A compact binary descriptor compared through Hamming distance.
final class BinaryDescriptor {
  /// Descriptor bits packed from least to most significant within each byte.
  final Uint8List bytes;

  /// Creates an immutable descriptor by copying [bytes].
  factory BinaryDescriptor({required Uint8List bytes}) {
    if (bytes.isEmpty) {
      throw ArgumentError.value(bytes, 'bytes', 'A descriptor cannot be empty.');
    }
    return BinaryDescriptor._(Uint8List.fromList(bytes).asUnmodifiableView());
  }

  /// Takes ownership of descriptor storage without copying it.
  ///
  /// The caller must not retain a mutable alias to [bytes].
  factory BinaryDescriptor.takeBytes({required Uint8List bytes}) {
    if (bytes.isEmpty) {
      throw ArgumentError.value(bytes, 'bytes', 'A descriptor cannot be empty.');
    }
    return BinaryDescriptor._(bytes.asUnmodifiableView());
  }

  /// Stores validated descriptor bytes.
  const BinaryDescriptor._(this.bytes);

  /// Returns the Hamming distance to [other].
  int distanceTo(BinaryDescriptor other) {
    if (bytes.length != other.bytes.length) {
      throw ArgumentError('Binary descriptors must have equal lengths.');
    }
    int distance = 0;
    for (int index = 0; index < bytes.length; index++) {
      distance += _populationCounts[bytes[index] ^ other.bytes[index]];
    }
    return distance;
  }

  /// Population count for every possible byte value.
  static final Uint8List _populationCounts = Uint8List.fromList([
    for (int value = 0; value < 256; value++) _countBits(value),
  ]);

  /// Counts set bits in one byte.
  static int _countBits(int input) {
    int value = input;
    int count = 0;
    while (value != 0) {
      value &= value - 1;
      count++;
    }
    return count;
  }
}

/// Keypoints and corresponding descriptors extracted from one image.
final class ImageFeatures {
  /// Oriented feature locations.
  final List<FeatureKeypoint> keypoints;

  /// Binary descriptors at the same indices as [keypoints].
  final List<BinaryDescriptor> descriptors;

  /// Creates an immutable aligned feature set.
  ImageFeatures({
    required List<FeatureKeypoint> keypoints,
    required List<BinaryDescriptor> descriptors,
  }) : keypoints = List<FeatureKeypoint>.unmodifiable(keypoints),
       descriptors = List<BinaryDescriptor>.unmodifiable(descriptors) {
    if (keypoints.length != descriptors.length) {
      throw ArgumentError('Every keypoint must have exactly one descriptor.');
    }
  }

  /// Number of described keypoints.
  int get length => keypoints.length;
}
