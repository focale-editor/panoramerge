import 'dart:math' as math;

/// A two-dimensional point represented with double-precision coordinates.
final class Point2 {
  /// Horizontal coordinate.
  final double x;

  /// Vertical coordinate.
  final double y;

  /// Creates a point from its Cartesian coordinates.
  const Point2({required this.x, required this.y});

  /// Returns the squared Euclidean distance to [other].
  double squaredDistanceTo(Point2 other) {
    final double deltaX = x - other.x;
    final double deltaY = y - other.y;
    return deltaX * deltaX + deltaY * deltaY;
  }

  /// Returns whether both coordinates contain finite values.
  bool get isFinite => x.isFinite && y.isFinite;

  @override
  String toString() => 'Point2(x: $x, y: $y)';
}

/// A double-precision axis-aligned rectangle.
final class Rectangle2 {
  /// Inclusive horizontal minimum.
  final double left;

  /// Inclusive vertical minimum.
  final double top;

  /// Exclusive horizontal maximum.
  final double right;

  /// Exclusive vertical maximum.
  final double bottom;

  /// Creates a rectangle from its bounds.
  const Rectangle2({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  /// Width of the rectangle.
  double get width => right - left;

  /// Height of the rectangle.
  double get height => bottom - top;

  /// Returns the smallest rectangle containing [points].
  factory Rectangle2.fromPoints(Iterable<Point2> points) {
    final Iterator<Point2> iterator = points.iterator;
    if (!iterator.moveNext()) {
      throw ArgumentError.value(points, 'points', 'At least one point is required.');
    }
    double left = iterator.current.x;
    double top = iterator.current.y;
    double right = iterator.current.x;
    double bottom = iterator.current.y;
    while (iterator.moveNext()) {
      final Point2 point = iterator.current;
      left = math.min(left, point.x);
      top = math.min(top, point.y);
      right = math.max(right, point.x);
      bottom = math.max(bottom, point.y);
    }
    return Rectangle2(left: left, top: top, right: right, bottom: bottom);
  }
}

/// A projective transform stored as a row-major 3 by 3 matrix.
final class ProjectiveTransform {
  /// First row, first column.
  final double m00;

  /// First row, second column.
  final double m01;

  /// First row, third column.
  final double m02;

  /// Second row, first column.
  final double m10;

  /// Second row, second column.
  final double m11;

  /// Second row, third column.
  final double m12;

  /// Third row, first column.
  final double m20;

  /// Third row, second column.
  final double m21;

  /// Third row, third column.
  final double m22;

  /// Creates a projective transform from its nine entries.
  const ProjectiveTransform({
    required this.m00,
    required this.m01,
    required this.m02,
    required this.m10,
    required this.m11,
    required this.m12,
    required this.m20,
    required this.m21,
    required this.m22,
  });

  /// Identity transform.
  const ProjectiveTransform.identity() : m00 = 1, m01 = 0, m02 = 0, m10 = 0, m11 = 1, m12 = 0, m20 = 0, m21 = 0, m22 = 1;

  /// Creates a two-dimensional translation.
  const ProjectiveTransform.translation({required double x, required double y}) : m00 = 1, m01 = 0, m02 = x, m10 = 0, m11 = 1, m12 = y, m20 = 0, m21 = 0, m22 = 1;

  /// Matrix determinant.
  double get determinant => m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);

  /// Maps [point] through this transform.
  Point2 transform(Point2 point) {
    final double denominator = m20 * point.x + m21 * point.y + m22;
    if (!denominator.isFinite || denominator.abs() < 1e-12) {
      return const Point2(x: double.nan, y: double.nan);
    }
    return Point2(
      x: (m00 * point.x + m01 * point.y + m02) / denominator,
      y: (m10 * point.x + m11 * point.y + m12) / denominator,
    );
  }

  /// Returns `this * other`, so [other] is applied first.
  ProjectiveTransform compose(ProjectiveTransform other) => ProjectiveTransform(
    m00: m00 * other.m00 + m01 * other.m10 + m02 * other.m20,
    m01: m00 * other.m01 + m01 * other.m11 + m02 * other.m21,
    m02: m00 * other.m02 + m01 * other.m12 + m02 * other.m22,
    m10: m10 * other.m00 + m11 * other.m10 + m12 * other.m20,
    m11: m10 * other.m01 + m11 * other.m11 + m12 * other.m21,
    m12: m10 * other.m02 + m11 * other.m12 + m12 * other.m22,
    m20: m20 * other.m00 + m21 * other.m10 + m22 * other.m20,
    m21: m20 * other.m01 + m21 * other.m11 + m22 * other.m21,
    m22: m20 * other.m02 + m21 * other.m12 + m22 * other.m22,
  );

  /// Returns the inverse transform, or `null` when the matrix is singular.
  ProjectiveTransform? inverse() {
    final double value = determinant;
    if (!value.isFinite || value.abs() < 1e-12) {
      return null;
    }
    final double inverseDeterminant = 1 / value;
    return ProjectiveTransform(
      m00: (m11 * m22 - m12 * m21) * inverseDeterminant,
      m01: (m02 * m21 - m01 * m22) * inverseDeterminant,
      m02: (m01 * m12 - m02 * m11) * inverseDeterminant,
      m10: (m12 * m20 - m10 * m22) * inverseDeterminant,
      m11: (m00 * m22 - m02 * m20) * inverseDeterminant,
      m12: (m02 * m10 - m00 * m12) * inverseDeterminant,
      m20: (m10 * m21 - m11 * m20) * inverseDeterminant,
      m21: (m01 * m20 - m00 * m21) * inverseDeterminant,
      m22: (m00 * m11 - m01 * m10) * inverseDeterminant,
    ).normalized();
  }

  /// Scales the matrix so its bottom-right entry is one when possible.
  ProjectiveTransform normalized() {
    final double divisor = m22.abs() >= 1e-12
        ? m22
        : math.sqrt(
            m00 * m00 + m01 * m01 + m02 * m02 + m10 * m10 + m11 * m11 + m12 * m12 + m20 * m20 + m21 * m21 + m22 * m22,
          );
    if (!divisor.isFinite || divisor.abs() < 1e-12) {
      return this;
    }
    return ProjectiveTransform(
      m00: m00 / divisor,
      m01: m01 / divisor,
      m02: m02 / divisor,
      m10: m10 / divisor,
      m11: m11 / divisor,
      m12: m12 / divisor,
      m20: m20 / divisor,
      m21: m21 / divisor,
      m22: m22 / divisor,
    );
  }

  /// Returns whether every matrix entry is finite.
  bool get isFinite => m00.isFinite && m01.isFinite && m02.isFinite && m10.isFinite && m11.isFinite && m12.isFinite && m20.isFinite && m21.isFinite && m22.isFinite;

  /// Returns the matrix entries in row-major order.
  List<double> toList() => [m00, m01, m02, m10, m11, m12, m20, m21, m22];

  @override
  String toString() => 'ProjectiveTransform(${toList()})';
}
