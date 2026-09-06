import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:panoramerge/src/blending/multiband_blender.dart';
import 'package:panoramerge/src/core/geometry.dart';
import 'package:panoramerge/src/feature/feature.dart';
import 'package:panoramerge/src/feature/orb_feature_detector.dart';
import 'package:panoramerge/src/image/float_raster.dart';
import 'package:panoramerge/src/image/panorama_raster.dart';
import 'package:panoramerge/src/matching/brute_force_matcher.dart';
import 'package:panoramerge/src/projection/image_projector.dart';
import 'package:panoramerge/src/registration/geometric_estimator.dart';
import 'package:panoramerge/src/seam/dynamic_programming_seam_finder.dart';

/// Stages reported by synchronous panorama progress callbacks.
enum PanoramaStage {
  /// Camera-surface projection.
  projection,

  /// Scale-space feature extraction.
  featureDetection,

  /// Pairwise descriptor matching and robust fitting.
  pairwiseRegistration,

  /// Global overlap-graph assembly.
  globalRegistration,

  /// Inverse image resampling onto the output canvas.
  warping,

  /// Minimum-cost overlap seam selection.
  seamFinding,

  /// Laplacian-pyramid reconstruction.
  blending,

  /// Canonical output encoding.
  encoding,
}

/// One monotonic progress update from a synchronous stitch operation.
final class PanoramaProgress {
  /// Pipeline stage currently executing.
  final PanoramaStage stage;

  /// Completed work units in this stage.
  final int completed;

  /// Total work units in this stage.
  final int total;

  /// Creates a progress update.
  const PanoramaProgress({
    required this.stage,
    required this.completed,
    required this.total,
  });

  /// Completed fraction in the range zero through one.
  double get fraction => total <= 0 ? 1 : (completed / total).clamp(0.0, 1.0);
}

/// Receives progress from [PanoramaStitcher.stitch].
typedef PanoramaProgressCallback = void Function(PanoramaProgress progress);

/// Classifies recoverable failures from the stitching pipeline.
enum PanoramaFailureCode {
  /// Source count, dimensions, bytes, or formats are inconsistent.
  invalidInput,

  /// One or more sources do not contain enough stable features.
  insufficientFeatures,

  /// Accepted pairwise overlaps do not connect every source.
  disconnectedOverlapGraph,

  /// A fitted transform produces invalid or extreme output geometry.
  unstableGeometry,

  /// The required output allocation exceeds its configured bound.
  outputTooLarge,

  /// Estimated floating pipeline storage exceeds its configured bound.
  workingSetTooLarge,
}

/// A failure with a stable machine-readable [code].
final class PanoramaException implements Exception {
  /// Failure category.
  final PanoramaFailureCode code;

  /// Human-readable diagnostic message.
  final String message;

  /// Source indices most directly associated with the failure.
  final List<int> sourceIndices;

  /// Creates an immutable panorama failure.
  PanoramaException({
    required this.code,
    required this.message,
    List<int> sourceIndices = const [],
  }) : sourceIndices = List<int>.unmodifiable(sourceIndices);

  @override
  String toString() => 'PanoramaException(${code.name}): $message';
}

/// Controls the complete registration, seam, and blending pipeline.
final class PanoramaStitcherOptions {
  /// Camera surface or scan-motion model.
  final PanoramaProjection projection;

  /// Optional focal length shared by curved projections.
  final double? focalLengthPixels;

  /// Largest image side used for feature registration before downsampling.
  final int registrationMaximumDimension;

  /// Whether transparent outer rows and columns are removed.
  final bool cropTransparentBorders;

  /// Whether RGB overlap means are equalized before seam blending.
  final bool compensateExposure;

  /// Whether the result retains each aligned source and its editable seam mask.
  ///
  /// Disabled by default because retaining one panorama-sized raster per
  /// source can be substantially more expensive than returning the flattened
  /// composite alone.
  final bool includeSourceLayers;

  /// Maximum aggregate source area accepted by one operation.
  final int maximumInputPixels;

  /// Maximum aggregate canonical source bytes accepted by one operation.
  final int maximumInputBytes;

  /// Maximum output canvas area accepted before floating allocations.
  final int maximumOutputPixels;

  /// Maximum canonical output bytes accepted by one operation.
  final int maximumOutputBytes;

  /// Maximum estimated transient floating-point working set.
  final int maximumWorkingBytes;

  /// Largest ordered index gap considered for large source sets.
  final int maximumPairDistance;

  /// Source count below which every possible image pair is tested.
  final int exhaustivePairSearchLimit;

  /// Smallest descriptor-match count sent to RANSAC.
  final int minimumPairMatches;

  /// Smallest inlier count required for an overlap edge.
  final int minimumPairInliers;

  /// Smallest accepted inlier fraction among descriptor matches.
  final double minimumInlierRatio;

  /// Scale-space feature detector tuning.
  final OrbFeatureDetectorOptions featureDetector;

  /// Binary descriptor matcher tuning.
  final FeatureMatcherOptions featureMatcher;

  /// Robust geometric estimator tuning.
  final RansacOptions ransac;

  /// Dynamic seam cost tuning.
  final SeamFinderOptions seamFinder;

  /// Multiscale blend-pyramid tuning.
  final MultibandBlenderOptions blender;

  /// Creates validated stitcher options.
  const PanoramaStitcherOptions({
    this.projection = PanoramaProjection.automatic,
    this.focalLengthPixels,
    this.registrationMaximumDimension = 2400,
    this.cropTransparentBorders = true,
    this.compensateExposure = true,
    this.includeSourceLayers = false,
    this.maximumInputPixels = 400 * 1000 * 1000,
    this.maximumInputBytes = 1024 * 1024 * 1024,
    this.maximumOutputPixels = 200 * 1000 * 1000,
    this.maximumOutputBytes = 500 * 1024 * 1024,
    this.maximumWorkingBytes = 4 * 1024 * 1024 * 1024,
    this.maximumPairDistance = 2,
    this.exhaustivePairSearchLimit = 6,
    this.minimumPairMatches = 10,
    this.minimumPairInliers = 8,
    this.minimumInlierRatio = 0.2,
    this.featureDetector = const OrbFeatureDetectorOptions(),
    this.featureMatcher = const FeatureMatcherOptions(),
    this.ransac = const RansacOptions(),
    this.seamFinder = const SeamFinderOptions(),
    this.blender = const MultibandBlenderOptions(),
  }) : assert(focalLengthPixels == null || focalLengthPixels > 0, 'focalLengthPixels must be positive.'),
       assert(registrationMaximumDimension >= 48, 'registrationMaximumDimension must be at least 48.'),
       assert(maximumInputPixels > 0, 'maximumInputPixels must be positive.'),
       assert(maximumInputBytes > 0, 'maximumInputBytes must be positive.'),
       assert(maximumOutputPixels > 0, 'maximumOutputPixels must be positive.'),
       assert(maximumOutputBytes > 0, 'maximumOutputBytes must be positive.'),
       assert(maximumWorkingBytes > 0, 'maximumWorkingBytes must be positive.'),
       assert(maximumPairDistance > 0, 'maximumPairDistance must be positive.'),
       assert(exhaustivePairSearchLimit >= 2, 'exhaustivePairSearchLimit must be at least two.'),
       assert(minimumPairMatches > 0, 'minimumPairMatches must be positive.'),
       assert(minimumPairInliers > 0, 'minimumPairInliers must be positive.'),
       assert(minimumInlierRatio > 0 && minimumInlierRatio <= 1, 'minimumInlierRatio must be in (0, 1].');
}

/// Diagnostics for one tested source pair.
final class PairwiseRegistrationDiagnostic {
  /// First source index.
  final int firstSourceIndex;

  /// Second source index.
  final int secondSourceIndex;

  /// Ratio-tested descriptor correspondence count.
  final int matchCount;

  /// RANSAC consensus count.
  final int inlierCount;

  /// Forward root-mean-square inlier error in pixels.
  final double rootMeanSquareError;

  /// Whether this pair became an overlap-graph edge.
  final bool accepted;

  /// Creates one immutable pair diagnostic.
  const PairwiseRegistrationDiagnostic({
    required this.firstSourceIndex,
    required this.secondSourceIndex,
    required this.matchCount,
    required this.inlierCount,
    required this.rootMeanSquareError,
    required this.accepted,
  });
}

/// Maps one original source through its camera surface into the final canvas.
final class PanoramaSourceTransform {
  /// Nonlinear camera projection applied before registration.
  final ProjectionMapping projection;

  /// Projective mapping from projected coordinates to output pixels.
  final ProjectiveTransform projectedToPanorama;

  /// Creates one composed source mapping.
  const PanoramaSourceTransform({
    required this.projection,
    required this.projectedToPanorama,
  });

  /// Maps an original image coordinate into the returned panorama raster.
  Point2 transform(Point2 sourcePoint) => projectedToPanorama.transform(projection.project(sourcePoint));
}

/// Evidence retained from panorama registration and graph assembly.
final class PanoramaDiagnostics {
  /// Extracted feature count for each source.
  final List<int> featureCounts;

  /// All source pairs tested by the configured search window.
  final List<PairwiseRegistrationDiagnostic> pairwiseRegistrations;

  /// Source selected as the global coordinate reference.
  final int referenceSourceIndex;

  /// Order in which connected sources entered the composite.
  final List<int> compositionOrder;

  /// Creates immutable diagnostics.
  PanoramaDiagnostics({
    required List<int> featureCounts,
    required List<PairwiseRegistrationDiagnostic> pairwiseRegistrations,
    required this.referenceSourceIndex,
    required List<int> compositionOrder,
  }) : featureCounts = List<int>.unmodifiable(featureCounts),
       pairwiseRegistrations = List<PairwiseRegistrationDiagnostic>.unmodifiable(pairwiseRegistrations),
       compositionOrder = List<int>.unmodifiable(compositionOrder);
}

/// A canonical panorama and the registration metadata used to create it.
final class PanoramaResult {
  /// Final canonical process-colour raster.
  final PanoramaRaster raster;

  /// Projection chosen after resolving automatic mode.
  final PanoramaProjection projection;

  /// Original-source mappings aligned with the input list.
  final List<PanoramaSourceTransform> sourceTransforms;

  /// Aligned source rasters in bottom-to-top composition order.
  ///
  /// Empty unless [PanoramaStitcherOptions.includeSourceLayers] was enabled.
  /// The first entry needs no mask; each later entry owns the portion selected
  /// by its seam mask while lower entries remain visible underneath.
  final List<PanoramaSourceLayer> sourceLayers;

  /// Feature and overlap evidence useful for troubleshooting.
  final PanoramaDiagnostics diagnostics;

  /// Creates an immutable stitch result.
  PanoramaResult({
    required this.raster,
    required this.projection,
    required List<PanoramaSourceTransform> sourceTransforms,
    List<PanoramaSourceLayer> sourceLayers = const [],
    required this.diagnostics,
  }) : sourceTransforms = List<PanoramaSourceTransform>.unmodifiable(sourceTransforms),
       sourceLayers = List<PanoramaSourceLayer>.unmodifiable(sourceLayers);
}

/// One aligned panorama source and the mask that reveals it over lower layers.
final class PanoramaSourceLayer {
  /// Index of the corresponding raster in the original input list.
  final int sourceIndex;

  /// Source pixels projected, registered, and exposure-compensated on the
  /// final panorama canvas.
  final PanoramaRaster raster;

  /// One grayscale byte per output pixel, or `null` for the bottom layer.
  ///
  /// Zero hides the source and 255 reveals it. Intermediate values are
  /// retained when a seam implementation produces fractional ownership.
  final Uint8List? mask;

  /// Creates one immutable editable-layer result.
  PanoramaSourceLayer({
    required this.sourceIndex,
    required this.raster,
    Uint8List? mask,
  }) : mask = mask == null ? null : Uint8List.fromList(mask).asUnmodifiableView() {
    if (sourceIndex < 0) {
      throw ArgumentError.value(sourceIndex, 'sourceIndex', 'must not be negative');
    }
    if (this.mask case final Uint8List values when values.lengthInBytes != raster.pixelCount) {
      throw ArgumentError.value(values.lengthInBytes, 'mask', 'must contain one byte per raster pixel');
    }
  }
}

/// Orchestrates pure Dart projection, registration, seam finding, and blending.
final class PanoramaStitcher {
  /// Complete immutable pipeline configuration.
  final PanoramaStitcherOptions options;

  /// Creates a panorama stitcher.
  const PanoramaStitcher({this.options = const PanoramaStitcherOptions()});

  /// Runs the complete pipeline on the current isolate.
  ///
  /// Automatic mode assumes a rotating camera and tries a cylindrical surface
  /// for three or more sources. A set that moved instead of rotating, such as
  /// a flatbed scan or a drone strip, does not register on that surface, so a
  /// registration failure there is retried on the perspective plane and
  /// [PanoramaResult.projection] reports which surface succeeded. The retry
  /// replays the pipeline, so [onProgress] sees a second run of every stage.
  PanoramaResult stitch(
    List<PanoramaRaster> sources, {
    PanoramaProgressCallback? onProgress,
  }) {
    _validateSources(sources);
    if (options.projection != PanoramaProjection.automatic || sources.length <= 2) {
      return _stitchProjected(
        sources,
        options.projection == PanoramaProjection.automatic ? PanoramaProjection.perspective : options.projection,
        onProgress,
      );
    }
    try {
      return _stitchProjected(sources, PanoramaProjection.cylindrical, onProgress);
    } on PanoramaException catch (failure) {
      if (!_recoverableOnAnotherSurface.contains(failure.code)) {
        rethrow;
      }
      return _stitchProjected(sources, PanoramaProjection.perspective, onProgress);
    }
  }

  /// Runs the pipeline once on an already resolved camera surface.
  PanoramaResult _stitchProjected(
    List<PanoramaRaster> sources,
    PanoramaProjection resolvedProjection,
    PanoramaProgressCallback? onProgress,
  ) {
    const ImageProjector projector = ImageProjector();
    final List<ProjectedFloatImage> projected = [];
    for (int index = 0; index < sources.length; index++) {
      projected.add(
        projector.projectFloat(
          sources[index],
          projection: resolvedProjection,
          focalLengthPixels: options.focalLengthPixels,
        ),
      );
      _report(onProgress, PanoramaStage.projection, index + 1, sources.length);
    }

    final OrbFeatureDetector detector = OrbFeatureDetector(options: options.featureDetector);
    final List<_RegistrationFeatures> features = [];
    for (int index = 0; index < projected.length; index++) {
      features.add(
        _detectRegistrationFeatures(
          detector,
          projected[index].raster.toGrayscale(sources.first.pixelFormat.colorModel),
        ),
      );
      _report(onProgress, PanoramaStage.featureDetection, index + 1, sources.length);
    }
    final int modelMinimum = _geometricModel(resolvedProjection).minimumSampleSize;
    final List<int> insufficient = [
      for (int index = 0; index < features.length; index++)
        if (features[index].features.length < modelMinimum) index,
    ];
    if (insufficient.isNotEmpty) {
      throw PanoramaException(
        code: PanoramaFailureCode.insufficientFeatures,
        message: 'Sources ${insufficient.join(', ')} do not contain enough stable corners for ${_geometricModel(resolvedProjection).name} registration.',
        sourceIndices: insufficient,
      );
    }

    final ({List<_PairEdge> edges, List<PairwiseRegistrationDiagnostic> diagnostics}) registration = _registerPairs(
      features,
      resolvedProjection,
      onProgress,
    );
    final _GlobalRegistration global = _assembleRegistrationGraph(
      sources.length,
      registration.edges,
    );
    _report(onProgress, PanoramaStage.globalRegistration, 1, 1);
    final _CanvasGeometry canvas = _canvasGeometry(
      projected,
      global.transforms,
      sources.first.pixelFormat,
    );
    final List<ProjectiveTransform> canvasTransforms = [
      for (final ProjectiveTransform transform in global.transforms) canvas.translation.compose(transform),
    ];

    final DynamicProgrammingSeamFinder seamFinder = DynamicProgrammingSeamFinder(options: options.seamFinder);
    final MultibandBlender blender = MultibandBlender(options: options.blender);
    final int firstSourceIndex = global.compositionOrder.first;
    FloatRaster composite = _warp(
      projected[firstSourceIndex].raster,
      canvasTransforms[firstSourceIndex],
      canvas.width,
      canvas.height,
    );
    final List<FloatRaster?> retainedSources = options.includeSourceLayers ? List<FloatRaster?>.filled(sources.length, null) : const [];
    final List<Float32List?> retainedMasks = options.includeSourceLayers ? List<Float32List?>.filled(sources.length, null) : const [];
    if (options.includeSourceLayers) {
      retainedSources[firstSourceIndex] = composite;
    }
    _report(onProgress, PanoramaStage.warping, 1, sources.length);
    for (int orderIndex = 1; orderIndex < global.compositionOrder.length; orderIndex++) {
      final int sourceIndex = global.compositionOrder[orderIndex];
      FloatRaster incoming = _warp(
        projected[sourceIndex].raster,
        canvasTransforms[sourceIndex],
        canvas.width,
        canvas.height,
      );
      _report(onProgress, PanoramaStage.warping, orderIndex + 1, sources.length);
      if (options.compensateExposure && sources.first.pixelFormat.colorModel == PanoramaColorModel.rgb) {
        incoming = _compensateExposure(composite, incoming);
      }
      final SeamMask seam = seamFinder.findFloat(
        composite,
        incoming,
        colorModel: sources.first.pixelFormat.colorModel,
      );
      if (options.includeSourceLayers) {
        retainedSources[sourceIndex] = incoming;
        retainedMasks[sourceIndex] = seam.incomingWeights;
      }
      _report(onProgress, PanoramaStage.seamFinding, orderIndex, sources.length - 1);
      composite = blender.blendFloat(composite, incoming, seamMask: seam);
      _report(onProgress, PanoramaStage.blending, orderIndex, sources.length - 1);
    }

    final _CroppedFloatRaster cropped = options.cropTransparentBorders ? _cropToAlpha(composite) : _CroppedFloatRaster(raster: composite, left: 0, top: 0);
    final ProjectiveTransform cropTranslation = ProjectiveTransform.translation(
      x: -cropped.left.toDouble(),
      y: -cropped.top.toDouble(),
    );
    final PanoramaRaster output = PanoramaRaster.fromPremultipliedComponents(
      width: cropped.raster.width,
      height: cropped.raster.height,
      pixelFormat: sources.first.pixelFormat,
      components: cropped.raster.components,
    );
    final List<PanoramaSourceLayer> sourceLayers = options.includeSourceLayers
        ? [
            for (final int sourceIndex in global.compositionOrder)
              _editableSourceLayer(
                sourceIndex: sourceIndex,
                source: retainedSources[sourceIndex],
                mask: retainedMasks[sourceIndex],
                left: cropped.left,
                top: cropped.top,
                width: cropped.raster.width,
                height: cropped.raster.height,
                pixelFormat: sources.first.pixelFormat,
              ),
          ]
        : const [];
    _report(onProgress, PanoramaStage.encoding, 1, 1);
    return PanoramaResult(
      raster: output,
      projection: resolvedProjection,
      sourceTransforms: [
        for (int index = 0; index < projected.length; index++)
          PanoramaSourceTransform(
            projection: projected[index].mapping,
            projectedToPanorama: cropTranslation.compose(canvasTransforms[index]),
          ),
      ],
      sourceLayers: sourceLayers,
      diagnostics: PanoramaDiagnostics(
        featureCounts: [for (final _RegistrationFeatures featureSet in features) featureSet.features.length],
        pairwiseRegistrations: registration.diagnostics,
        referenceSourceIndex: global.referenceIndex,
        compositionOrder: global.compositionOrder,
      ),
    );
  }

  /// Downscales registration evidence while retaining full-image coordinates.
  _RegistrationFeatures _detectRegistrationFeatures(
    OrbFeatureDetector detector,
    GrayImage fullResolution,
  ) {
    final int largestDimension = math.max(fullResolution.width, fullResolution.height);
    if (largestDimension <= options.registrationMaximumDimension) {
      return _RegistrationFeatures(
        features: detector.detectGrayscale(fullResolution),
        coordinateScale: 1,
      );
    }
    final double resizeScale = options.registrationMaximumDimension / largestDimension;
    final int width = math.max(1, (fullResolution.width * resizeScale).round());
    final int height = math.max(1, (fullResolution.height * resizeScale).round());
    final GrayImage reduced = _resizeGrayscaleArea(fullResolution, width, height);
    final ImageFeatures detected = detector.detectGrayscale(reduced);
    final double horizontalScale = width > 1 ? (fullResolution.width - 1) / (width - 1) : 1;
    final double verticalScale = height > 1 ? (fullResolution.height - 1) / (height - 1) : 1;
    final double coordinateScale = math.max(horizontalScale, verticalScale);
    return _RegistrationFeatures(
      features: ImageFeatures(
        keypoints: [
          for (final FeatureKeypoint keypoint in detected.keypoints)
            FeatureKeypoint(
              position: Point2(
                x: keypoint.position.x * horizontalScale,
                y: keypoint.position.y * verticalScale,
              ),
              octave: keypoint.octave,
              scale: keypoint.scale * math.sqrt(horizontalScale * verticalScale),
              orientation: keypoint.orientation,
              response: keypoint.response,
            ),
        ],
        descriptors: detected.descriptors,
      ),
      coordinateScale: coordinateScale,
    );
  }

  /// Area-averages luminance into bounded registration dimensions.
  GrayImage _resizeGrayscaleArea(GrayImage source, int width, int height) {
    final Uint8List output = Uint8List(width * height);
    for (int y = 0; y < height; y++) {
      final int sourceTop = y * source.height ~/ height;
      final int sourceBottom = math.max(sourceTop + 1, (y + 1) * source.height ~/ height);
      for (int x = 0; x < width; x++) {
        final int sourceLeft = x * source.width ~/ width;
        final int sourceRight = math.max(sourceLeft + 1, (x + 1) * source.width ~/ width);
        int sum = 0;
        int count = 0;
        for (int sourceY = sourceTop; sourceY < sourceBottom; sourceY++) {
          for (int sourceX = sourceLeft; sourceX < sourceRight; sourceX++) {
            sum += source.bytes[sourceY * source.width + sourceX];
            count++;
          }
        }
        output[y * width + x] = (sum + count ~/ 2) ~/ count;
      }
    }
    return GrayImage.takeBytes(width: width, height: height, bytes: output);
  }

  /// Runs [stitch] on a worker isolate.
  ///
  /// Progress callbacks are deliberately unavailable across this convenience
  /// boundary; applications needing streamed progress can call [stitch] from
  /// an isolate they own.
  Future<PanoramaResult> stitchAsync(List<PanoramaRaster> sources) => Isolate.run(() => stitch(sources));

  /// Rejects unsafe source collections before expensive floating allocations.
  void _validateSources(List<PanoramaRaster> sources) {
    if (sources.length < 2 || sources.length > 32) {
      throw PanoramaException(
        code: PanoramaFailureCode.invalidInput,
        message: 'A panorama requires between 2 and 32 sources.',
      );
    }
    final PanoramaPixelFormat format = sources.first.pixelFormat;
    int totalPixels = 0;
    int totalBytes = 0;
    for (int index = 0; index < sources.length; index++) {
      final PanoramaRaster source = sources[index];
      if (source.pixelFormat != format) {
        throw PanoramaException(
          code: PanoramaFailureCode.invalidInput,
          message: 'Every source must use the same canonical pixel format.',
          sourceIndices: [0, index],
        );
      }
      totalPixels += source.pixelCount;
      totalBytes += source.bytes.lengthInBytes;
      if (totalPixels > options.maximumInputPixels) {
        throw PanoramaException(
          code: PanoramaFailureCode.invalidInput,
          message: 'Aggregate source area exceeds ${options.maximumInputPixels} pixels.',
        );
      }
      if (totalBytes > options.maximumInputBytes) {
        throw PanoramaException(
          code: PanoramaFailureCode.invalidInput,
          message: 'Aggregate source storage exceeds ${options.maximumInputBytes} canonical bytes.',
        );
      }
    }
  }

  /// Matches configured source pairs and retains reliable RANSAC edges.
  ({List<_PairEdge> edges, List<PairwiseRegistrationDiagnostic> diagnostics}) _registerPairs(
    List<_RegistrationFeatures> features,
    PanoramaProjection projection,
    PanoramaProgressCallback? onProgress,
  ) {
    final BruteForceFeatureMatcher matcher = BruteForceFeatureMatcher(options: options.featureMatcher);
    final GeometricModel model = _geometricModel(projection);
    final List<(int, int)> pairs = [];
    for (int first = 0; first < features.length - 1; first++) {
      for (int second = first + 1; second < features.length; second++) {
        if (features.length <= options.exhaustivePairSearchLimit || second - first <= options.maximumPairDistance) {
          pairs.add((first, second));
        }
      }
    }
    final List<_PairEdge> edges = [];
    final List<PairwiseRegistrationDiagnostic> diagnostics = [];
    for (int pairIndex = 0; pairIndex < pairs.length; pairIndex++) {
      final (int first, int second) = pairs[pairIndex];
      final ImageFeatures firstFeatures = features[first].features;
      final ImageFeatures secondFeatures = features[second].features;
      final List<FeatureMatch> descriptorMatches = matcher.match(firstFeatures, secondFeatures);
      GeometricEstimate? estimate;
      if (descriptorMatches.length >= math.max(options.minimumPairMatches, model.minimumSampleSize)) {
        final List<PointMatch> pointMatches = [
          for (final FeatureMatch match in descriptorMatches)
            PointMatch(
              source: firstFeatures.keypoints[match.queryIndex].position,
              destination: secondFeatures.keypoints[match.trainIndex].position,
              descriptorDistance: match.distance,
            ),
        ];
        final double coordinateScale = math.max(
          features[first].coordinateScale,
          features[second].coordinateScale,
        );
        final RansacOptions ransacOptions = RansacOptions(
          inlierThreshold: options.ransac.inlierThreshold * coordinateScale,
          maximumIterations: options.ransac.maximumIterations,
          confidence: options.ransac.confidence,
          randomSeed: options.ransac.randomSeed,
          symmetricTransferError: options.ransac.symmetricTransferError,
        );
        final RansacGeometricEstimator estimator = RansacGeometricEstimator(
          options: ransacOptions,
        );
        estimate = estimator.estimate(pointMatches, model: model);
      }
      final int inliers = estimate?.inlierIndices.length ?? 0;
      final double ratio = descriptorMatches.isEmpty ? 0 : inliers / descriptorMatches.length;
      final bool accepted = estimate != null && inliers >= math.max(options.minimumPairInliers, model.minimumSampleSize) && ratio >= options.minimumInlierRatio;
      diagnostics.add(
        PairwiseRegistrationDiagnostic(
          firstSourceIndex: first,
          secondSourceIndex: second,
          matchCount: descriptorMatches.length,
          inlierCount: inliers,
          rootMeanSquareError: estimate?.rootMeanSquareError ?? double.infinity,
          accepted: accepted,
        ),
      );
      if (accepted) {
        final GeometricEstimate acceptedEstimate = estimate;
        edges.add(
          _PairEdge(
            first: first,
            second: second,
            firstToSecond: acceptedEstimate.transform,
            quality: inliers / (1 + acceptedEstimate.rootMeanSquareError),
          ),
        );
      }
      _report(onProgress, PanoramaStage.pairwiseRegistration, pairIndex + 1, pairs.length);
    }
    return (edges: edges, diagnostics: diagnostics);
  }

  /// Builds a maximum-quality spanning tree and source-to-reference transforms.
  _GlobalRegistration _assembleRegistrationGraph(int sourceCount, List<_PairEdge> edges) {
    final List<double> qualities = List<double>.filled(sourceCount, 0);
    for (final _PairEdge edge in edges) {
      qualities[edge.first] += edge.quality;
      qualities[edge.second] += edge.quality;
    }
    int reference = 0;
    for (int index = 1; index < sourceCount; index++) {
      if (qualities[index] > qualities[reference] || qualities[index] == qualities[reference] && (index - sourceCount / 2).abs() < (reference - sourceCount / 2).abs()) {
        reference = index;
      }
    }
    final List<ProjectiveTransform?> transforms = List<ProjectiveTransform?>.filled(sourceCount, null);
    transforms[reference] = const ProjectiveTransform.identity();
    final Set<int> connected = {reference};
    final List<int> order = [reference];
    while (connected.length < sourceCount) {
      _PairEdge? best;
      for (final _PairEdge edge in edges) {
        final bool firstConnected = connected.contains(edge.first);
        final bool secondConnected = connected.contains(edge.second);
        if (firstConnected == secondConnected) {
          continue;
        }
        if (best == null || edge.quality > best.quality) {
          best = edge;
        }
      }
      if (best == null) {
        final List<int> missing = [
          for (int index = 0; index < sourceCount; index++)
            if (!connected.contains(index)) index,
        ];
        throw PanoramaException(
          code: PanoramaFailureCode.disconnectedOverlapGraph,
          message: 'No reliable feature path connects sources ${missing.join(', ')} to the panorama.',
          sourceIndices: missing,
        );
      }
      if (connected.contains(best.first)) {
        final ProjectiveTransform? reverse = best.firstToSecond.inverse();
        if (reverse == null) {
          throw PanoramaException(
            code: PanoramaFailureCode.unstableGeometry,
            message: 'The transform between sources ${best.first} and ${best.second} is singular.',
            sourceIndices: [best.first, best.second],
          );
        }
        transforms[best.second] = transforms[best.first]!.compose(reverse);
        connected.add(best.second);
        order.add(best.second);
      } else {
        transforms[best.first] = transforms[best.second]!.compose(best.firstToSecond);
        connected.add(best.first);
        order.add(best.first);
      }
    }
    return _GlobalRegistration(
      transforms: [for (final ProjectiveTransform? transform in transforms) transform!],
      referenceIndex: reference,
      compositionOrder: order,
    );
  }

  /// Computes finite translated output bounds for every registered source.
  _CanvasGeometry _canvasGeometry(
    List<ProjectedFloatImage> images,
    List<ProjectiveTransform> transforms,
    PanoramaPixelFormat pixelFormat,
  ) {
    double minimumX = double.infinity;
    double minimumY = double.infinity;
    double maximumX = double.negativeInfinity;
    double maximumY = double.negativeInfinity;
    for (int index = 0; index < images.length; index++) {
      final FloatRaster image = images[index].raster;
      final List<Point2> corners = [
        const Point2(x: 0, y: 0),
        Point2(x: (image.width - 1).toDouble(), y: 0),
        Point2(x: (image.width - 1).toDouble(), y: (image.height - 1).toDouble()),
        Point2(x: 0, y: (image.height - 1).toDouble()),
      ];
      final List<double> denominators = [
        for (final Point2 corner in corners) transforms[index].m20 * corner.x + transforms[index].m21 * corner.y + transforms[index].m22,
      ];
      final double smallestDenominator = denominators.reduce(math.min);
      final double largestDenominator = denominators.reduce(math.max);
      if (denominators.any((value) => !value.isFinite || value.abs() < 1e-8) || smallestDenominator < 0 && largestDenominator > 0) {
        throw PanoramaException(
          code: PanoramaFailureCode.unstableGeometry,
          message: 'Source $index crosses the fitted homography line at infinity.',
          sourceIndices: [index],
        );
      }
      for (final Point2 corner in corners) {
        final Point2 transformed = transforms[index].transform(corner);
        if (!transformed.isFinite || transformed.x.abs() > 1e9 || transformed.y.abs() > 1e9) {
          throw PanoramaException(
            code: PanoramaFailureCode.unstableGeometry,
            message: 'Source $index maps outside stable panorama coordinates.',
            sourceIndices: [index],
          );
        }
        minimumX = math.min(minimumX, transformed.x);
        minimumY = math.min(minimumY, transformed.y);
        maximumX = math.max(maximumX, transformed.x);
        maximumY = math.max(maximumY, transformed.y);
      }
    }
    final int left = minimumX.floor();
    final int top = minimumY.floor();
    final int width = maximumX.ceil() - left + 1;
    final int height = maximumY.ceil() - top + 1;
    if (width < 1 || height < 1 || width > options.maximumOutputPixels ~/ height) {
      throw PanoramaException(
        code: PanoramaFailureCode.outputTooLarge,
        message: 'Registered geometry requires a ${width}x$height canvas, exceeding ${options.maximumOutputPixels} pixels.',
      );
    }
    final int pixelCount = width * height;
    if (pixelCount > options.maximumOutputBytes ~/ pixelFormat.bytesPerPixel) {
      throw PanoramaException(
        code: PanoramaFailureCode.outputTooLarge,
        message: 'Registered geometry exceeds ${options.maximumOutputBytes} canonical output bytes in $pixelFormat.',
      );
    }
    final int sourceWorkingBytes = images.fold<int>(
      0,
      (total, image) => total + image.raster.width * image.raster.height * pixelFormat.channelCount * Float32List.bytesPerElement,
    );
    final int canvasWorkingBytes = pixelCount * pixelFormat.channelCount * Float32List.bytesPerElement * 12;
    final int editableOutputBytes = options.includeSourceLayers
        ? pixelCount * images.length * (pixelFormat.channelCount * Float32List.bytesPerElement + pixelFormat.bytesPerPixel + Float32List.bytesPerElement + 2)
        : 0;
    if (sourceWorkingBytes + canvasWorkingBytes + editableOutputBytes > options.maximumWorkingBytes) {
      throw PanoramaException(
        code: PanoramaFailureCode.workingSetTooLarge,
        message: 'Estimated floating and editable-output working set exceeds ${options.maximumWorkingBytes} bytes; reduce dimensions, source count, blend levels, or raise the explicit limit.',
      );
    }
    return _CanvasGeometry(
      width: width,
      height: height,
      translation: ProjectiveTransform.translation(x: -left.toDouble(), y: -top.toDouble()),
    );
  }

  /// Inverse-warps one projected image into the common output canvas.
  FloatRaster _warp(
    FloatRaster source,
    ProjectiveTransform sourceToCanvas,
    int width,
    int height,
  ) {
    final ProjectiveTransform? inverse = sourceToCanvas.inverse();
    if (inverse == null) {
      throw PanoramaException(
        code: PanoramaFailureCode.unstableGeometry,
        message: 'A source-to-canvas transform cannot be inverted.',
      );
    }
    final FloatRaster output = FloatRaster.empty(
      width: width,
      height: height,
      channelCount: source.channelCount,
    );
    final List<Point2> corners = [
      const Point2(x: 0, y: 0),
      Point2(x: (source.width - 1).toDouble(), y: 0),
      Point2(x: (source.width - 1).toDouble(), y: (source.height - 1).toDouble()),
      Point2(x: 0, y: (source.height - 1).toDouble()),
    ].map(sourceToCanvas.transform).where((point) => point.isFinite).toList(growable: false);
    final Rectangle2 bounds = Rectangle2.fromPoints(corners);
    final int left = bounds.left.floor().clamp(0, width - 1);
    final int top = bounds.top.floor().clamp(0, height - 1);
    final int right = bounds.right.ceil().clamp(0, width - 1);
    final int bottom = bounds.bottom.ceil().clamp(0, height - 1);
    for (int y = top; y <= bottom; y++) {
      for (int x = left; x <= right; x++) {
        final Point2 sample = inverse.transform(Point2(x: x.toDouble(), y: y.toDouble()));
        _sampleBilinear(source, sample.x, sample.y, output, x, y);
      }
    }
    return output;
  }

  /// Converts one retained aligned source into a cropped editable layer.
  PanoramaSourceLayer _editableSourceLayer({
    required int sourceIndex,
    required FloatRaster? source,
    required Float32List? mask,
    required int left,
    required int top,
    required int width,
    required int height,
    required PanoramaPixelFormat pixelFormat,
  }) {
    if (source == null) {
      throw StateError('Panorama source $sourceIndex was not retained');
    }
    final FloatRaster croppedSource = _cropRaster(
      source,
      left: left,
      top: top,
      width: width,
      height: height,
    );
    return PanoramaSourceLayer(
      sourceIndex: sourceIndex,
      raster: PanoramaRaster.fromPremultipliedComponents(
        width: width,
        height: height,
        pixelFormat: pixelFormat,
        components: croppedSource.components,
      ),
      mask: mask == null
          ? null
          : _cropMask(
              mask,
              sourceWidth: source.width,
              left: left,
              top: top,
              width: width,
              height: height,
            ),
    );
  }

  /// Crops one canvas-aligned floating raster without changing its samples.
  FloatRaster _cropRaster(
    FloatRaster source, {
    required int left,
    required int top,
    required int width,
    required int height,
  }) {
    if (left == 0 && top == 0 && width == source.width && height == source.height) {
      return source;
    }
    final int channelCount = source.channelCount;
    final int rowLength = width * channelCount;
    final Float32List values = Float32List(width * height * channelCount);
    for (int y = 0; y < height; y++) {
      final int sourceOffset = ((top + y) * source.width + left) * channelCount;
      final int destinationOffset = y * rowLength;
      values.setRange(
        destinationOffset,
        destinationOffset + rowLength,
        source.components,
        sourceOffset,
      );
    }
    return FloatRaster(
      width: width,
      height: height,
      channelCount: channelCount,
      components: values,
    );
  }

  /// Quantizes and crops fractional seam ownership into one-byte mask samples.
  Uint8List _cropMask(
    Float32List source, {
    required int sourceWidth,
    required int left,
    required int top,
    required int width,
    required int height,
  }) {
    final Uint8List output = Uint8List(width * height);
    for (int y = 0; y < height; y++) {
      final int sourceRow = (top + y) * sourceWidth + left;
      final int destinationRow = y * width;
      for (int x = 0; x < width; x++) {
        output[destinationRow + x] = (source[sourceRow + x].clamp(0.0, 1.0) * 255).round();
      }
    }
    return output;
  }

  /// Bilinearly samples premultiplied process channels into a destination.
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
    final int outputOffset = (destinationY * destination.width + destinationX) * destination.channelCount;
    final int topLeft = (y0 * source.width + x0) * source.channelCount;
    final int topRight = (y0 * source.width + x1) * source.channelCount;
    final int bottomLeft = (y1 * source.width + x0) * source.channelCount;
    final int bottomRight = (y1 * source.width + x1) * source.channelCount;
    for (int channel = 0; channel < source.channelCount; channel++) {
      final double upper = source.components[topLeft + channel] * (1 - fractionX) + source.components[topRight + channel] * fractionX;
      final double lower = source.components[bottomLeft + channel] * (1 - fractionX) + source.components[bottomRight + channel] * fractionX;
      destination.components[outputOffset + channel] = upper * (1 - fractionY) + lower * fractionY;
    }
  }

  /// Equalizes mean straight RGB values over reliable overlap pixels.
  FloatRaster _compensateExposure(FloatRaster existing, FloatRaster incoming) {
    final List<double> existingSums = [0, 0, 0];
    final List<double> incomingSums = [0, 0, 0];
    int count = 0;
    for (int pixel = 0; pixel < existing.width * existing.height; pixel++) {
      final int offset = pixel * existing.channelCount;
      final double alphaA = existing.components[offset + existing.alphaChannelIndex];
      final double alphaB = incoming.components[offset + incoming.alphaChannelIndex];
      if (alphaA < 0.95 || alphaB < 0.95) {
        continue;
      }
      double luminanceA = 0;
      double luminanceB = 0;
      for (int channel = 0; channel < 3; channel++) {
        luminanceA += existing.components[offset + channel] * _luminanceWeights[channel];
        luminanceB += incoming.components[offset + channel] * _luminanceWeights[channel];
      }
      if (luminanceA < 0.03 || luminanceB < 0.03 || luminanceA > 0.97 || luminanceB > 0.97) {
        continue;
      }
      for (int channel = 0; channel < 3; channel++) {
        existingSums[channel] += existing.components[offset + channel];
        incomingSums[channel] += incoming.components[offset + channel];
      }
      count++;
    }
    if (count < 32) {
      return incoming;
    }
    final List<double> gains = List<double>.generate(3, (channel) {
      if (incomingSums[channel] <= 1e-8) {
        return 1;
      }
      return (existingSums[channel] / incomingSums[channel]).clamp(0.5, 2.0);
    });
    final Float32List adjusted = Float32List.fromList(incoming.components);
    for (int pixel = 0; pixel < incoming.width * incoming.height; pixel++) {
      final int offset = pixel * incoming.channelCount;
      for (int channel = 0; channel < 3; channel++) {
        adjusted[offset + channel] *= gains[channel];
      }
    }
    return FloatRaster(
      width: incoming.width,
      height: incoming.height,
      channelCount: incoming.channelCount,
      components: adjusted,
    );
  }

  /// Crops floating storage and records the removed origin.
  _CroppedFloatRaster _cropToAlpha(FloatRaster source) {
    int left = source.width;
    int top = source.height;
    int right = 0;
    int bottom = 0;
    for (int y = 0; y < source.height; y++) {
      for (int x = 0; x < source.width; x++) {
        if (source.alphaAt(x, y) <= 1e-6) {
          continue;
        }
        left = math.min(left, x);
        top = math.min(top, y);
        right = math.max(right, x + 1);
        bottom = math.max(bottom, y + 1);
      }
    }
    if (right <= left || bottom <= top || left == 0 && top == 0 && right == source.width && bottom == source.height) {
      return _CroppedFloatRaster(raster: source, left: 0, top: 0);
    }
    final int width = right - left;
    final int height = bottom - top;
    final Float32List output = Float32List(width * height * source.channelCount);
    for (int y = 0; y < height; y++) {
      final int sourceStart = ((top + y) * source.width + left) * source.channelCount;
      final int destinationStart = y * width * source.channelCount;
      output.setRange(
        destinationStart,
        destinationStart + width * source.channelCount,
        source.components,
        sourceStart,
      );
    }
    return _CroppedFloatRaster(
      raster: FloatRaster(
        width: width,
        height: height,
        channelCount: source.channelCount,
        components: output,
      ),
      left: left,
      top: top,
    );
  }

  /// Maps a projection choice to its RANSAC transformation family.
  GeometricModel _geometricModel(PanoramaProjection projection) => switch (projection) {
    PanoramaProjection.affine => GeometricModel.affine,
    PanoramaProjection.translation => GeometricModel.translation,
    PanoramaProjection.automatic || PanoramaProjection.perspective || PanoramaProjection.cylindrical || PanoramaProjection.spherical => GeometricModel.homography,
  };

  /// Sends a progress update when a callback was supplied.
  void _report(
    PanoramaProgressCallback? callback,
    PanoramaStage stage,
    int completed,
    int total,
  ) => callback?.call(
    PanoramaProgress(stage: stage, completed: completed, total: total),
  );

  /// Rec. 709 RGB luminance coefficients.
  static const List<double> _luminanceWeights = [0.2126, 0.7152, 0.0722];

  /// Failures that describe a badly chosen camera surface rather than
  /// unusable sources, so automatic mode may retry them on another surface.
  static const Set<PanoramaFailureCode> _recoverableOnAnotherSurface = {
    PanoramaFailureCode.insufficientFeatures,
    PanoramaFailureCode.disconnectedOverlapGraph,
    PanoramaFailureCode.unstableGeometry,
  };
}

/// One accepted directed pairwise overlap transform.
final class _PairEdge {
  /// Lower source index.
  final int first;

  /// Higher source index.
  final int second;

  /// Mapping from [first] projected pixels to [second] projected pixels.
  final ProjectiveTransform firstToSecond;

  /// Inlier support penalized by reprojection error.
  final double quality;

  /// Creates one overlap edge.
  const _PairEdge({
    required this.first,
    required this.second,
    required this.firstToSecond,
    required this.quality,
  });
}

/// Feature evidence expressed in full projected-image coordinates.
final class _RegistrationFeatures {
  /// Keypoints and descriptors used by pair matching.
  final ImageFeatures features;

  /// Full-resolution pixels represented by one detector input pixel.
  final double coordinateScale;

  /// Creates scaled registration evidence.
  const _RegistrationFeatures({
    required this.features,
    required this.coordinateScale,
  });
}

/// Source-to-reference transforms selected by the overlap spanning tree.
final class _GlobalRegistration {
  /// Transform for every projected source.
  final List<ProjectiveTransform> transforms;

  /// Chosen reference source index.
  final int referenceIndex;

  /// Spanning-tree discovery order.
  final List<int> compositionOrder;

  /// Creates global registration storage.
  const _GlobalRegistration({
    required this.transforms,
    required this.referenceIndex,
    required this.compositionOrder,
  });
}

/// Output allocation dimensions and reference-to-canvas translation.
final class _CanvasGeometry {
  /// Output width.
  final int width;

  /// Output height.
  final int height;

  /// Translation making every transformed corner nonnegative.
  final ProjectiveTransform translation;

  /// Creates canvas geometry.
  const _CanvasGeometry({
    required this.width,
    required this.height,
    required this.translation,
  });
}

/// A cropped floating raster and its removed canvas origin.
final class _CroppedFloatRaster {
  /// Cropped raster data.
  final FloatRaster raster;

  /// Removed columns on the left.
  final int left;

  /// Removed rows at the top.
  final int top;

  /// Creates one crop result.
  const _CroppedFloatRaster({
    required this.raster,
    required this.left,
    required this.top,
  });
}
