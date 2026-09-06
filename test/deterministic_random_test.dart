import 'package:panoramerge/src/core/deterministic_random.dart';
import 'package:test/test.dart';

void main() {
  test('generator emits one pinned xorshift32 stream on every target', () {
    // Arrange.
    // These values were captured from both the virtual machine and the same
    // library compiled to JavaScript. Native integers are 64-bit while
    // compiled JavaScript truncates shifts to 32 bits, so dropping the masks
    // inside the generator changes this stream on one target only. Check both
    // with `dart test -p vm,chrome test/deterministic_random_test.dart`; the
    // stitcher suite is virtual-machine only because `stitchAsync` needs
    // isolates.
    final DeterministicRandom random = DeterministicRandom(0x51f15e);

    // Act.
    final List<int> stream = [for (int index = 0; index < 8; index++) random.next()];

    // Assert.
    expect(stream, [
      4047495683,
      2727285564,
      1576983112,
      1346681715,
      578659366,
      1239204890,
      4204952085,
      166763589,
    ]);
  });

  test('every emitted state stays inside the unsigned 32-bit range', () {
    // Arrange.
    final DeterministicRandom random = DeterministicRandom(0x51f15e);

    // Act and assert.
    for (int index = 0; index < 4096; index++) {
      expect(random.next(), inInclusiveRange(0, 0xffffffff));
    }
  });

  test('generator replaces the forbidden zero state', () {
    // Arrange.
    final DeterministicRandom random = DeterministicRandom(0);

    // Act.
    final List<int> stream = [for (int index = 0; index < 4; index++) random.next()];

    // Assert.
    expect(stream, [1359758873, 3761132862, 2075758394, 25405621]);
  });

  test('bounded draws stay in range and reject an empty bound', () {
    // Arrange.
    final DeterministicRandom random = DeterministicRandom(1);

    // Act.
    final List<int> draws = [for (int index = 0; index < 6; index++) random.nextInt(37)];

    // Assert.
    expect(draws, [10, 21, 28, 10, 27, 5]);
    expect(draws, everyElement(inInclusiveRange(0, 36)));
    expect(() => DeterministicRandom(1).nextInt(0), throwsRangeError);
  });
}
