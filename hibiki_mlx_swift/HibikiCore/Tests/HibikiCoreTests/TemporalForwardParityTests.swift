import Foundation
import HibikiCore
import MLX
import XCTest

/// Parity: the native temporal `forwardText` must reproduce the Python
/// reference's transformer output and predicted tokens on a fixed input. The
/// fixture (`Fixtures/temporal_forward.safetensors`) was produced by the Python
/// implementation in float32 on CPU; this test runs the Swift port the same way.
final class TemporalForwardParityTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    private var fixtureURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/temporal_forward.safetensors")
    }

    private var bundleDirectory: URL {
        if let override = ProcessInfo.processInfo.environment["HIBIKI_ARTIFACTS"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root.appendingPathComponent("artifacts/hibiki-1b-mlx-bf16", isDirectory: true)
    }

    func testTemporalForwardMatchesReference() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: bundleDirectory.appendingPathComponent("config.json").path),
            "artifact bundle not present; download it first")

        let fixture = try loadArrays(url: fixtureURL)
        let tokens = try XCTUnwrap(fixture["tokens"])
        let expectedOut = try XCTUnwrap(fixture["transformer_out"]).asType(.float32)
        let expectedArgmax = try XCTUnwrap(fixture["text_argmax"]).asType(.int32)

        let bundle = try ArtifactBundle.validate(directory: bundleDirectory)
        let model = try HibikiLM.loadTemporal(from: bundle)
        let cache = model.makeTransformerCache()
        let (out, logits) = model.forwardText(tokens, cache: cache)
        eval(out, logits)

        // Continuous parity: the transformer output must match within a small
        // float32 tolerance (same algorithm, same backend).
        let maxAbsDiff = MLX.abs(out.asType(.float32) - expectedOut).max().item(Float.self)
        print("temporal parity max abs diff = \(maxAbsDiff)")
        XCTAssertLessThan(maxAbsDiff, 1e-3, "transformer output diverged from the reference")

        // Token parity: predicted text tokens must match exactly.
        let argmax = logits.argMax(axis: -1).asType(.int32)
        let tokensEqual = (argmax .== expectedArgmax).all().item(Bool.self)
        XCTAssertTrue(tokensEqual, "predicted text tokens diverged from the reference")
    }
}
