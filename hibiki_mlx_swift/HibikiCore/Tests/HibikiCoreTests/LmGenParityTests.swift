import Foundation
import HibikiCore
import MLX
import XCTest

/// Parity: the native delayed-stream scheduler (`LmGen`) must reproduce the
/// Python reference's per-step text tokens and completed audio frames on a fixed
/// sequence of source tokens, for both bundles. Greedy sampling keeps it
/// deterministic; the fixture carries the exact source tokens.
final class LmGenParityTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    private func runLmGenParity(bundleSubpath: String, fixtureName: String) throws {
        let bundleDirectory = MLXTestSupport.bundleURL(bundleSubpath)
        try MLXTestSupport.requireBundle(bundleDirectory)

        let model = try LoadedModel.load(directory: bundleDirectory)
        let condition = try model.lm.conditionTensor("description", "very_good")

        let fixture = try loadArrays(url: MLXTestSupport.fixtureURL(fixtureName))
        let source = try XCTUnwrap(fixture["source_tokens"]) // [frames, sourceCodebooks]
        let expectedText = try XCTUnwrap(fixture["text_tokens"]).asType(.int32) // [frames]
        let expectedAudio = try XCTUnwrap(fixture["audio_frames"]).asType(.int32) // [ready, targetCodebooks]

        let generator = LmGen(model: model.lm, textSampler: Sampler(), audioSampler: Sampler())
        var texts: [MLXArray] = []
        var audioFrames: [MLXArray] = []
        for index in 0 ..< source.dim(0) {
            let frame = source[index].reshaped([1, -1])
            let textToken = generator.step(sourceTokens: frame, condition: condition)
            texts.append(textToken.reshaped([1]))
            if let audio = generator.lastAudioTokens() { audioFrames.append(audio) }
        }

        let text = concatenated(texts, axis: 0)
        eval(text)
        XCTAssertTrue((text .== expectedText).all().item(Bool.self),
                      "scheduled text tokens diverged from the reference")

        let audio = concatenated(audioFrames, axis: 0)
        eval(audio)
        XCTAssertEqual(audio.shape, expectedAudio.shape, "ready audio frame count diverged")
        XCTAssertTrue((audio .== expectedAudio).all().item(Bool.self),
                      "scheduled audio frames diverged from the reference")
    }

    func testLmGenBF16() throws {
        try runLmGenParity(bundleSubpath: "artifacts/hibiki-1b-mlx-bf16", fixtureName: "lmgen_bf16.safetensors")
    }

    func testLmGenQ8() throws {
        try runLmGenParity(bundleSubpath: "artifacts/hibiki-1b-mlx-q8", fixtureName: "lmgen_q8.safetensors")
    }
}
