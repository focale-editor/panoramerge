import 'dart:typed_data';

import 'package:panoramerge/panoramerge.dart';
import 'package:test/test.dart';

void main() {
  test('ORB-family detector returns distributed repeatable features', () {
    // Arrange.
    final PanoramaRaster image = _texturedRaster(width: 160, height: 120);
    const OrbFeatureDetector detector = OrbFeatureDetector(
      options: OrbFeatureDetectorOptions(
        maximumFeatures: 300,
        pyramidLevels: 3,
        distributionCellSize: 32,
      ),
    );

    // Act.
    final ImageFeatures first = detector.detect(image);
    final ImageFeatures second = detector.detect(image);

    // Assert.
    expect(first.length, greaterThan(40));
    expect(first.length, lessThanOrEqualTo(300));
    expect(second.length, first.length);
    expect(
      second.keypoints.map((keypoint) => (keypoint.position.x, keypoint.position.y)),
      orderedEquals(first.keypoints.map((keypoint) => (keypoint.position.x, keypoint.position.y))),
    );
    for (int index = 0; index < first.descriptors.length; index++) {
      expect(second.descriptors[index].bytes, orderedEquals(first.descriptors[index].bytes));
    }
  });

  test('Hamming matcher applies ratio and mutual checks', () {
    // Arrange.
    final ImageFeatures query = _features([
      [0x00, 0x00],
      [0xff, 0x00],
      [0xaa, 0xaa],
    ]);
    final ImageFeatures train = _features([
      [0x00, 0x01],
      [0xfe, 0x00],
      [0x55, 0x55],
    ]);
    const BruteForceFeatureMatcher matcher = BruteForceFeatureMatcher(
      options: FeatureMatcherOptions(
        maximumHammingDistance: 4,
        ratioThreshold: 0.8,
      ),
    );

    // Act.
    final List<FeatureMatch> matches = matcher.match(query, train);

    // Assert.
    expect(matches, hasLength(2));
    expect(matches.map((match) => (match.queryIndex, match.trainIndex, match.distance)), [
      (0, 0, 1),
      (1, 1, 1),
    ]);
  });
}

ImageFeatures _features(List<List<int>> descriptors) => ImageFeatures(
  keypoints: [
    for (int index = 0; index < descriptors.length; index++)
      FeatureKeypoint(
        position: Point2(x: index.toDouble(), y: 0),
        octave: 0,
        scale: 1,
        orientation: 0,
        response: 1,
      ),
  ],
  descriptors: [
    for (final List<int> descriptor in descriptors) BinaryDescriptor(bytes: Uint8List.fromList(descriptor)),
  ],
);

PanoramaRaster _texturedRaster({required int width, required int height}) {
  final Uint8List straight = Uint8List(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int offset = (y * width + x) * 4;
      final int hash = (x * 73 ^ y * 151 ^ x * y * 17) & 255;
      straight[offset] = hash;
      straight[offset + 1] = (hash * 3 + x * 5) & 255;
      straight[offset + 2] = (hash * 7 + y * 11) & 255;
      straight[offset + 3] = 255;
    }
  }
  return PanoramaRaster.fromStraightRgba8(width: width, height: height, bytes: straight);
}
