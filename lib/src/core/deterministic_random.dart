/// A small xorshift32 generator producing one stream on every Dart target.
///
/// This library is deliberately absent from the public `panoramerge` exports.
/// It lives on its own so tests can pin the sample stream directly, because
/// the estimators that consume it converge on the same answer from different
/// streams and therefore cannot detect a change in the generator.
final class DeterministicRandom {
  /// Current unsigned 32-bit state.
  int _state;

  /// Creates a generator, replacing the forbidden zero state.
  DeterministicRandom(int seed) : _state = seed & 0xffffffff {
    if (_state == 0) {
      _state = 0x9e3779b9;
    }
  }

  /// Returns the next raw unsigned 32-bit state.
  ///
  /// Each left shift is masked back to 32 bits before the next step. Native
  /// integers are 64-bit while compiled JavaScript truncates shifts to 32
  /// bits, so an unmasked intermediate would feed different bits to the
  /// following right shift and yield a different stream on the web.
  int next() {
    int value = _state;
    value ^= (value << 13) & 0xffffffff;
    value ^= value >>> 17;
    value ^= (value << 5) & 0xffffffff;
    _state = value;
    return value;
  }

  /// Returns a uniformly mapped integer in `[0, upperBound)`.
  int nextInt(int upperBound) {
    if (upperBound <= 0) {
      throw RangeError.range(upperBound, 1, null, 'upperBound');
    }
    return next() % upperBound;
  }
}
