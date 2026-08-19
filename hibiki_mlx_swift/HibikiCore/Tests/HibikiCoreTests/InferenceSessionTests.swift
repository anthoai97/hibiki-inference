import Foundation
import HibikiCore
import MLX
import XCTest

/// End-to-end parity for the streaming session: pushing a fixed PCM tone and
/// finishing must reproduce the Python reference's per-step text tokens and the
/// concatenated output PCM, exercising Mimi encode → delayed scheduling →
/// generation → Mimi decode → the silence-tail finalization. Greedy samplers
/// keep it deterministic.
///
/// This verifies the pipeline wiring and ordering on macOS-CPU. A real-audio
/// transcript from the 43 s `2.wav` (issue #24 AC 1) is impractical here — the
/// CPU path runs far slower than real time — so that is verified on-device in
/// #26; this test drives a short synthetic tone instead.
final class InferenceSessionTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    func testSessionParityQ8() throws {
        let bundleDirectory = MLXTestSupport.bundleURL("artifacts/hibiki-1b-mlx-q8")
        try MLXTestSupport.requireBundle(bundleDirectory)

        let model = try LoadedModel.load(directory: bundleDirectory)
        let fixture = try loadArrays(url: MLXTestSupport.fixtureURL("session_q8.safetensors"))
        let inputPCM = try XCTUnwrap(fixture["input_pcm"]).asType(.float32) // [1, 1, total]
        let expectedText = try XCTUnwrap(fixture["text_tokens"]).asType(.int32) // [steps]
        let expectedPCM = try XCTUnwrap(fixture["output_pcm"]).asType(.float32) // [samples]

        let session = try InferenceSession(
            model: model, condition: "very_good",
            textSampler: Sampler(), audioSampler: Sampler())

        let samples = inputPCM.reshaped([-1]).asArray(Float.self)
        var results = try session.pushPCM(samples)
        results += session.finish()

        let text = MLXArray(results.map { Int32($0.textToken) })
        XCTAssertEqual(text.shape, expectedText.shape, "step count diverged from the reference")
        XCTAssertTrue((text .== expectedText).all().item(Bool.self), "session text tokens diverged")

        let pcm = MLXArray(results.compactMap { $0.pcm }.flatMap { $0 })
        eval(pcm)
        XCTAssertEqual(pcm.shape, expectedPCM.shape, "output PCM length diverged from the reference")
        let maxAbsDiff = MLX.abs(pcm - expectedPCM).max().item(Float.self)
        print("session PCM max abs diff = \(maxAbsDiff)")
        XCTAssertLessThan(maxAbsDiff, 1e-3, "session output PCM diverged from the reference")

        // Ordering (AC 2): text frames are produced in model-time order, and each
        // completed audio frame trails its text frame by the max delay (two).
        XCTAssertEqual(results.map { $0.textFrameIndex }, Array(0 ..< results.count),
                       "text frames were not emitted in model-time order")
        for result in results where result.pcm != nil {
            XCTAssertEqual(result.audioFrameIndex, result.textFrameIndex - 2,
                           "completed audio frame was not delayed by two frames")
        }

        // Failure behavior (AC 5): pushing after finish is rejected, recoverably.
        XCTAssertThrowsError(try session.pushPCM(samples)) { error in
            XCTAssertTrue(error is SessionError)
        }
    }

    func testWarmupLeavesNoState() throws {
        let bundleDirectory = MLXTestSupport.bundleURL("artifacts/hibiki-1b-mlx-q8")
        try MLXTestSupport.requireBundle(bundleDirectory)
        let model = try LoadedModel.load(directory: bundleDirectory)
        let session = try InferenceSession(model: model)

        session.warmup()
        // After warmup the session is fresh: no text has accumulated and a new
        // push produces results without error.
        XCTAssertEqual(session.text, "")
        let results = try session.pushPCM([Float](repeating: 0, count: model.mimi.cfg.frameSize))
        XCTAssertFalse(results.isEmpty)
    }
}
