import 'package:panoramerge/panoramerge.dart';
import 'package:test/test.dart';

void main() {
  test('RANSAC recovers a projective transform in the presence of outliers', () {
    // Arrange.
    const ProjectiveTransform expected = ProjectiveTransform(
      m00: 1.02,
      m01: 0.03,
      m02: 18,
      m10: -0.02,
      m11: 0.98,
      m12: -7,
      m20: 0.0003,
      m21: -0.0002,
      m22: 1,
    );
    final List<PointMatch> matches = [];
    for (int y = 0; y < 5; y++) {
      for (int x = 0; x < 7; x++) {
        final Point2 source = Point2(x: x * 31.0 + 3, y: y * 27.0 + 5);
        final Point2 projected = expected.transform(source);
        final double noise = ((x * 11 + y * 7) % 5 - 2) * 0.08;
        matches.add(
          PointMatch(
            source: source,
            destination: Point2(x: projected.x + noise, y: projected.y - noise),
          ),
        );
      }
    }
    matches.addAll(const [
      PointMatch(source: Point2(x: 0, y: 0), destination: Point2(x: 400, y: 20)),
      PointMatch(source: Point2(x: 50, y: 10), destination: Point2(x: -80, y: 300)),
      PointMatch(source: Point2(x: 100, y: 80), destination: Point2(x: 10, y: -200)),
      PointMatch(source: Point2(x: 180, y: 90), destination: Point2(x: 500, y: 500)),
    ]);
    const RansacGeometricEstimator estimator = RansacGeometricEstimator(
      options: RansacOptions(
        inlierThreshold: 1,
        maximumIterations: 800,
        randomSeed: 42,
      ),
    );

    // Act.
    final GeometricEstimate? result = estimator.estimate(matches);

    // Assert.
    expect(result, isNotNull);
    expect(result!.inlierIndices, hasLength(35));
    expect(result.rootMeanSquareError, lessThan(0.25));
    for (final Point2 point in const [
      Point2(x: 10, y: 20),
      Point2(x: 150, y: 75),
      Point2(x: 210, y: 130),
    ]) {
      final Point2 actual = result.transform.transform(point);
      final Point2 wanted = expected.transform(point);
      expect(actual.x, closeTo(wanted.x, 0.2));
      expect(actual.y, closeTo(wanted.y, 0.2));
    }
  });

  test('translation model uses median consensus refinement', () {
    // Arrange.
    final List<PointMatch> matches = [
      for (int index = 0; index < 20; index++)
        PointMatch(
          source: Point2(x: index * 4, y: index * 3),
          destination: Point2(x: index * 4 - 37, y: index * 3 + 6),
        ),
      const PointMatch(
        source: Point2(x: 5, y: 5),
        destination: Point2(x: 500, y: -300),
      ),
    ];

    // Act.
    final GeometricEstimate? result = const RansacGeometricEstimator().estimate(
      matches,
      model: GeometricModel.translation,
    );

    // Assert.
    expect(result, isNotNull);
    final Point2 mapped = result!.transform.transform(const Point2(x: 50, y: 40));
    expect(mapped.x, closeTo(13, 1e-9));
    expect(mapped.y, closeTo(46, 1e-9));
    expect(result.inlierIndices, hasLength(20));
  });

  test('affine model recovers rotation, scale, shear, and translation', () {
    // Arrange.
    const ProjectiveTransform expected = ProjectiveTransform(
      m00: 1.08,
      m01: -0.17,
      m02: 23,
      m10: 0.21,
      m11: 0.94,
      m12: -11,
      m20: 0,
      m21: 0,
      m22: 1,
    );
    final List<PointMatch> matches = [
      for (int y = 0; y < 4; y++)
        for (int x = 0; x < 5; x++)
          PointMatch(
            source: Point2(x: x * 29 + 4, y: y * 31 + 7),
            destination: expected.transform(
              Point2(x: x * 29 + 4, y: y * 31 + 7),
            ),
          ),
      const PointMatch(
        source: Point2(x: 40, y: 50),
        destination: Point2(x: -300, y: 700),
      ),
    ];

    // Act.
    final GeometricEstimate? result = const RansacGeometricEstimator().estimate(
      matches,
      model: GeometricModel.affine,
    );

    // Assert.
    expect(result, isNotNull);
    expect(result!.inlierIndices, hasLength(20));
    final Point2 mapped = result.transform.transform(
      const Point2(x: 91, y: 63),
    );
    final Point2 wanted = expected.transform(const Point2(x: 91, y: 63));
    expect(mapped.x, closeTo(wanted.x, 1e-7));
    expect(mapped.y, closeTo(wanted.y, 1e-7));
  });
}
