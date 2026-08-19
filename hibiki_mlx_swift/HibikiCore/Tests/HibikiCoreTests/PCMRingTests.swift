import Foundation
import HibikiCore
import XCTest

/// Feedback loop for the "breaks every frame" symptom: a continuous 24 kHz
/// pull (the Python `OutputStream.write` model) must replay written frames
/// with no inserted zeros at the 80 ms seams. Discrete `scheduleBuffer`
/// playback is the iOS failure mode this catches.
final class PCMRingTests: XCTestCase {
    private let frameSize = 1_920
    private let callback = 256

    /// A slow linear ramp, so a seam is a jump that cannot occur inside a frame.
    private func ramp(start: Float, count: Int) -> [Float] {
        (0 ..< count).map { start + Float($0) * 0.0001 }
    }

    func testPullDoesNotInsertGapsBetweenWrittenFrames() {
        let ring = PCMRing(capacity: frameSize * 16)
        var expected: [Float] = []
        for index in 0 ..< 8 {
            let frame = ramp(start: Float(index) * 0.2, count: frameSize)
            expected.append(contentsOf: frame)
            ring.write(frame)
        }

        var got = [Float](repeating: 0, count: expected.count)
        var offset = 0
        got.withUnsafeMutableBufferPointer { dest in
            while offset < expected.count {
                let n = min(callback, expected.count - offset)
                let copied = ring.read(into: dest.baseAddress! + offset, count: n)
                XCTAssertEqual(copied, n, "ring ran dry while frames were still queued")
                offset += n
            }
        }
        XCTAssertEqual(got, expected, "pull inserted or dropped samples between frames")
        XCTAssertEqual(maxSeamJump(got, every: frameSize), maxSeamJump(expected, every: frameSize),
                       "frame seams in the pulled stream diverged from the written PCM")
    }

    /// The PlayerNode failure mode: a few milliseconds of silence between
    /// 80 ms frames. That is the "breaking each frame" the listener hears.
    func testInsertedSilenceBetweenFramesIsDetectable() {
        var chopped: [Float] = []
        let hole = [Float](repeating: 0, count: 120) // 5 ms at 24 kHz
        for index in 0 ..< 6 {
            chopped.append(contentsOf: ramp(start: 0.3, count: frameSize))
            if index < 5 { chopped.append(contentsOf: hole) }
        }
        let expectedSeam = maxSeamJump(ramp(start: 0.3, count: frameSize * 6), every: frameSize)
        XCTAssertGreaterThan(
            maxSeamJump(chopped, every: frameSize + hole.count),
            expectedSeam + 0.2,
            "the detector must go red on silence stuffed between frames")
    }

    func testUnderrunFillsZerosThenContinuesWithoutDropping() {
        let ring = PCMRing(capacity: frameSize * 4)
        let first = ramp(start: 0.1, count: 100)
        ring.write(first)

        var got = [Float](repeating: -1, count: 180)
        let copied = got.withUnsafeMutableBufferPointer { dest in
            ring.read(into: dest.baseAddress!, count: 180)
        }
        XCTAssertEqual(copied, 100)
        XCTAssertEqual(Array(got.prefix(100)), first)
        XCTAssertTrue(got.suffix(80).allSatisfy { $0 == 0 })

        let second = ramp(start: 0.5, count: 50)
        ring.write(second)
        var tail = [Float](repeating: 0, count: 50)
        tail.withUnsafeMutableBufferPointer { dest in
            XCTAssertEqual(ring.read(into: dest.baseAddress!, count: 50), 50)
        }
        XCTAssertEqual(tail, second)
    }

    func testAbortUnblocksAFullWrite() {
        let ring = PCMRing(capacity: 8)
        ring.write(Array(repeating: 1, count: 8))

        let done = expectation(description: "write returned after abort")
        DispatchQueue.global().async {
            ring.write(Array(repeating: 2, count: 4))
            done.fulfill()
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.05) { ring.abort() }
        wait(for: [done], timeout: 1.0)
    }
}

/// Largest absolute jump between the last sample of one block and the first
/// sample of the next. Continuous speech stays small; a stuffed hole does not.
private func maxSeamJump(_ samples: [Float], every block: Int) -> Float {
    var best: Float = 0
    var index = block
    while index < samples.count {
        best = max(best, abs(samples[index] - samples[index - 1]))
        index += block
    }
    return best
}
