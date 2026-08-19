import HibikiCore
import XCTest

/// Live English starts only after a 1 s preroll, and after an underrun waits
/// for a shorter refill. That is how a 160–300 ms generate spike stays in the
/// ring instead of becoming a click.
final class LivePlaybackGateTests: XCTestCase {
    func testWaitsForPrerollThenPlaysThroughAShortSpike() {
        var gate = LivePlaybackGate(sampleRate: 24_000, prerollSeconds: 1.0, rebufferSeconds: 0.4)
        let preroll = 24_000

        XCTAssertFalse(gate.shouldConsume(available: preroll - 1, streamEnded: false),
                       "played before 1 s was buffered")
        XCTAssertTrue(gate.shouldConsume(available: preroll, streamEnded: false))
        XCTAssertTrue(gate.playing)

        // A 200 ms spike drains 4_800 samples; 1 s preroll still has headroom.
        XCTAssertTrue(gate.shouldConsume(available: preroll - 4_800, streamEnded: false),
                      "a 200 ms spike should not stop playback after a 1 s preroll")
    }

    func testUnderrunWaitsForRebufferNotFullPreroll() {
        var gate = LivePlaybackGate(sampleRate: 24_000, prerollSeconds: 1.0, rebufferSeconds: 0.4)
        XCTAssertTrue(gate.shouldConsume(available: 24_000, streamEnded: false))
        XCTAssertFalse(gate.shouldConsume(available: 0, streamEnded: false),
                       "should stop consuming once the ring is empty")
        XCTAssertFalse(gate.shouldConsume(available: 5_000, streamEnded: false),
                       "should not resume after only 200 ms of refill")
        XCTAssertTrue(gate.shouldConsume(available: 9_600, streamEnded: false),
                      "should resume after the 400 ms rebuffer")
    }

    func testEndedPlaysAShortTailWithoutWaiting() {
        var gate = LivePlaybackGate()
        XCTAssertTrue(gate.shouldConsume(available: 100, streamEnded: true),
                      "the last incomplete preroll must still be heard")
        XCTAssertFalse(gate.shouldConsume(available: 0, streamEnded: true))
    }
}
