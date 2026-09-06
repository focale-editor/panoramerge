import 'dart:math' as math;

import 'package:panoramerge/src/core/geometry.dart';
import 'package:panoramerge/src/image/float_raster.dart';
import 'package:panoramerge/src/image/panorama_raster.dart';

/// Selects the camera surface and registration model used by the stitcher.
enum PanoramaProjection {
  /// Chooses perspective for two inputs and cylindrical for longer sweeps.
  automatic,

  /// Keeps each source on its original perspective plane.
  perspective,

  /// Maps horizontal camera rotation onto a cylinder.
  cylindrical,

  /// Maps horizontal and vertical camera rotation onto a sphere.
  spherical,

  /// Keeps pixels planar and restricts registration to an affine transform.
  affine,

  /// Keeps pixels planar and restricts registration to translation.
  translation,
}

/// Bidirectional mapping between source and projected pixel coordinates.
final class ProjectionMapping {
  /// Projection surface represented by this mapping.
  final PanoramaProjection projection;

  /// Source width used to locate the optical centre.
  final int width;

  /// Source height used to locate the optical centre.
  final int height;

  /// Camera focal length in pixels.
  final double focalLengthPixels;

  /// Creates a validated mapping.
  const ProjectionMapping({
    required this.projection,
    required this.width,
    required this.height,
    required this.focalLengthPixels,
  }) : assert(width > 0, 'width must be positive.'),
       assert(height > 0, 'height must be positive.'),
       assert(focalLengthPixels > 0, 'focalLengthPixels must be positive.');

  /// Maps an original source point onto the selected camera surface.
  Point2 project(Point2 point) {
    if (!_usesCurvedSurface) {
      return point;
    }
    final double centerX = (width - 1) / 2;
    final double centerY = (height - 1) / 2;
    final double horizontal = point.x - centerX;
    final double vertical = point.y - centerY;
    final double theta = math.atan(horizontal / focalLengthPixels);
    if (projection == PanoramaProjection.cylindrical) {
      return Point2(
        x: centerX + focalLengthPixels * theta,
        y: centerY + focalLengthPixels * vertical / math.sqrt(horizontal * horizontal + focalLengthPixels * focalLengthPixels),
      );
    }
    final double phi = math.atan(vertical / math.sqrt(horizontal * horizontal + focalLengthPixels * focalLengthPixels));
    return Point2(
      x: centerX + focalLengthPixels * theta,
      y: centerY + focalLengthPixels * phi,
    );
  }

  /// Maps one projected point back to the source perspective plane.
  ///
  /// Returns a non-finite point when the projected coordinate sits a quarter
  /// turn or more away from the optical axis. Such a ray points behind the
  /// camera, so no source pixel can supply it. Without this guard the tangent
  /// wraps and returns a valid-looking coordinate on the opposite side of the
  /// image, which resamples mirrored content instead of leaving the pixel
  /// uncovered. Callers treat a non-finite result as transparent.
  Point2 unproject(Point2 point) {
    if (!_usesCurvedSurface) {
      return point;
    }
    final double centerX = (width - 1) / 2;
    final double centerY = (height - 1) / 2;
    final double theta = (point.x - centerX) / focalLengthPixels;
    if (!(theta.abs() < _quarterTurn)) {
      return _behindCamera;
    }
    final double horizontal = focalLengthPixels * math.tan(theta);
    final double radius = math.sqrt(horizontal * horizontal + focalLengthPixels * focalLengthPixels);
    if (projection == PanoramaProjection.cylindrical) {
      return Point2(
        x: centerX + horizontal,
        y: centerY + (point.y - centerY) * radius / focalLengthPixels,
      );
    }
    final double phi = (point.y - centerY) / focalLengthPixels;
    if (!(phi.abs() < _quarterTurn)) {
      return _behindCamera;
    }
    return Point2(x: centerX + horizontal, y: centerY + math.tan(phi) * radius);
  }

  /// Whether the mapping changes the source sampling surface.
  bool get _usesCurvedSurface => projection == PanoramaProjection.cylindrical || projection == PanoramaProjection.spherical;

  /// Largest ray angle from the optical axis that still faces the camera.
  static const double _quarterTurn = math.pi / 2;

  /// Result marking a projected coordinate that no source pixel can supply.
  static const Point2 _behindCamera = Point2(x: double.nan, y: double.nan);
}

/// A projected canonical raster and its nonlinear coordinate mapping.
final class ProjectedImage {
  /// Resampled raster on the projection surface.
  final PanoramaRaster raster;

  /// Mapping used to construct [raster].
  final ProjectionMapping mapping;

  /// Creates a projected image result.
  const ProjectedImage({required this.raster, required this.mapping});
}

/// Floating projection result retained internally by the stitching pipeline.
final class ProjectedFloatImage {
  /// Native premultiplied samples without intermediate quantization.
  final FloatRaster raster;

  /// Canonical format to use for the eventual output.
  final PanoramaPixelFormat pixelFormat;

  /// Nonlinear coordinate mapping applied to the source.
  final ProjectionMapping mapping;

  /// Creates an internal floating projection result.
  const ProjectedFloatImage({
    required this.raster,
    required this.pixelFormat,
    required this.mapping,
  });
}

/// Resamples canonical rasters onto planar, cylindrical, or spherical surfaces.
final class ImageProjector {
  /// Creates a stateless projector.
  const ImageProjector();

  /// Projects [source] and encodes the result in the source pixel format.
  ProjectedImage project(
    PanoramaRaster source, {
    required PanoramaProjection projection,
    double? focalLengthPixels,
  }) {
    final ProjectedFloatImage projected = projectFloat(
      source,
      projection: projection,
      focalLengthPixels: focalLengthPixels,
    );
    return ProjectedImage(
      raster: PanoramaRaster.fromPremultipliedComponents(
        width: source.width,
        height: source.height,
        pixelFormat: source.pixelFormat,
        components: projected.raster.components,
      ),
      mapping: projected.mapping,
    );
  }

  /// Projects [source] while retaining unquantized floating samples.
  ///
  /// This method supports pipeline composition. Applications normally use
  /// [project] or the high-level panorama stitcher.
  ProjectedFloatImage projectFloat(
    PanoramaRaster source, {
    required PanoramaProjection projection,
    double? focalLengthPixels,
  }) {
    final PanoramaProjection resolved = projection == PanoramaProjection.automatic ? PanoramaProjection.perspective : projection;
    final double focal = focalLengthPixels ?? math.max(source.width, source.height) * 0.8;
    if (!focal.isFinite || focal <= 0) {
      throw ArgumentError.value(focalLengthPixels, 'focalLengthPixels', 'Focal length must be positive and finite.');
    }
    final ProjectionMapping mapping = ProjectionMapping(
      projection: resolved,
      width: source.width,
      height: source.height,
      focalLengthPixels: focal,
    );
    final FloatRaster input = FloatRaster.fromRaster(source);
    if (resolved != PanoramaProjection.cylindrical && resolved != PanoramaProjection.spherical) {
      return ProjectedFloatImage(
        raster: input,
        pixelFormat: source.pixelFormat,
        mapping: mapping,
      );
    }
    final FloatRaster output = FloatRaster.empty(
      width: source.width,
      height: source.height,
      channelCount: source.pixelFormat.channelCount,
    );
    for (int y = 0; y < source.height; y++) {
      for (int x = 0; x < source.width; x++) {
        final Point2 sample = mapping.unproject(Point2(x: x.toDouble(), y: y.toDouble()));
        _sampleBilinear(input, sample.x, sample.y, output, x, y);
      }
    }
    return ProjectedFloatImage(
      raster: output,
      pixelFormat: source.pixelFormat,
      mapping: mapping,
    );
  }

  /// Samples premultiplied components with transparent pixels outside input.
  void _sampleBilinear(
    FloatRaster source,
    double x,
    double y,
    FloatRaster destination,
    int destinationX,
    int destinationY,
  ) {
    if (!x.isFinite || !y.isFinite || x < 0 || y < 0 || x > source.width - 1 || y > source.height - 1) {
      return;
    }
    final int x0 = x.floor();
    final int y0 = y.floor();
    final int x1 = math.min(x0 + 1, source.width - 1);
    final int y1 = math.min(y0 + 1, source.height - 1);
    final double fractionX = x - x0;
    final double fractionY = y - y0;
    final int destinationOffset = (destinationY * destination.width + destinationX) * destination.channelCount;
    final int topLeft = (y0 * source.width + x0) * source.channelCount;
    final int topRight = (y0 * source.width + x1) * source.channelCount;
    final int bottomLeft = (y1 * source.width + x0) * source.channelCount;
    final int bottomRight = (y1 * source.width + x1) * source.channelCount;
    for (int channel = 0; channel < source.channelCount; channel++) {
      final double top = source.components[topLeft + channel] * (1 - fractionX) + source.components[topRight + channel] * fractionX;
      final double bottom = source.components[bottomLeft + channel] * (1 - fractionX) + source.components[bottomRight + channel] * fractionX;
      destination.components[destinationOffset + channel] = top * (1 - fractionY) + bottom * fractionY;
    }
  }
}
