import Foundation
import HibikiCore
import MLX
import XCTest

/// Parity: the native full generation step (`frameState` + `sampleStep`) must
/// reproduce the Python reference's temporal state and greedily-sampled text and
/// audio tokens on a fixed frame, for both the BF16 and Q8 bundles. This
/// exercises the audio embeddings, the conditioner, and the Depth Transformer —
/// the parts the loaded-model warm eval adds on top of the temporal path. The
/// fixtures were produced in float32 on CPU (see `scripts/fixtures`).
final class SampleStepParityTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    private func runSampleStepParity(bundleSubpath: String, fixtureName: String) throws {
        let bundleDirectory = MLXTestSupport.bundleURL(bundleSubpath)
        try MLXTestSupport.requireBundle(bundleDirectory)

        let fixture = try loadArrays(url: MLXTestSupport.fixtureURL(fixtureName))
        let text = try XCTUnwrap(fixture["text_token_ids"]) // [1, 1]
        let audioStacked = try XCTUnwrap(fixture["audio_token_ids"]) // [nQ, 1]
        let audio = (0 ..< audioStacked.dim(0)).map { audioStacked[$0].reshaped(1, 1) }
        let expectedOut = try XCTUnwrap(fixture["transformer_out"]).asType(.float32)
        let expectedTextToken = try XCTUnwrap(fixture["text_token"]).asType(.int32)
        let expectedAudioTokens = try XCTUnwrap(fixture["audio_tokens"]).asType(.int32)

        let model = try LoadedModel.load(directory: bundleDirectory)
        let condition = try model.lm.conditionTensor("description", "very_good")

        // Temporal state (float check).
        let out = model.lm.frameState(
            textTokenIds: text, audioTokenIds: audio,
            cache: model.lm.makeTransformerCache(), condition: condition)
        eval(out)
        let maxAbsDiff = MLX.abs(out.asType(.float32) - expectedOut).max().item(Float.self)
        print("sample_step parity (\(bundleSubpath)) transformer_out max abs diff = \(maxAbsDiff)")
        XCTAssertLessThan(maxAbsDiff, 1e-3, "temporal state diverged from the reference")

        // Sampled tokens (exact check).
        let greedy = Sampler()
        let (textToken, audioTokens) = model.lm.sampleStep(
            textTokenIds: text, audioTokenIds: audio,
            transformerCache: model.lm.makeTransformerCache(),
            depformerCache: model.lm.makeDepformerCache(),
            textSampler: greedy, audioSampler: greedy, condition: condition)
        eval(textToken, audioTokens)
        XCTAssertTrue((textToken .== expectedTextToken).all().item(Bool.self),
                      "sampled text token diverged from the reference")
        XCTAssertTrue((audioTokens .== expectedAudioTokens).all().item(Bool.self),
                      "sampled audio tokens diverged from the reference")
    }

    func testSampleStepBF16() throws {
        try runSampleStepParity(
            bundleSubpath: "artifacts/hibiki-1b-mlx-bf16",
            fixtureName: "sample_step_bf16.safetensors")
    }

    func testSampleStepQ8() throws {
        try runSampleStepParity(
            bundleSubpath: "artifacts/hibiki-1b-mlx-q8",
            fixtureName: "sample_step_q8.safetensors")
    }
}
