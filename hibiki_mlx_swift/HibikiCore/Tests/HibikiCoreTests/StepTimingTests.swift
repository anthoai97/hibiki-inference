import HibikiCore
import XCTest

/// The step-timing line must stay in the Python `--metrics` layout so the two
/// runners can be compared by eye. Timing is diagnostics: the session only
/// fills it when `measureTiming` is on.
final class StepTimingTests: XCTestCase {
    func testFormattedLineMatchesPythonMetricsLayout() {
        let timing = StepTiming(
            sourceEncodeSeconds: 0.002,
            generationSeconds: 0.003,
            targetDecodeSeconds: 0.004,
            textDecodeSeconds: 0.001)
        XCTAssertEqual(timing.totalSeconds, 0.01, accuracy: 1e-12)
        XCTAssertEqual(
            timing.formatted(textFrameIndex: 10, audioFrameIndex: 8),
            "step=10 text_frame=10 audio_frame=8 phases: encode=2.0ms generate=3.0ms decode=4.0ms text=1.0ms total=10.0ms")
        XCTAssertEqual(
            timing.formatted(textFrameIndex: 0, audioFrameIndex: nil),
            "step=0 text_frame=0 audio_frame=- phases: encode=2.0ms generate=3.0ms decode=4.0ms text=1.0ms total=10.0ms")
    }
}
