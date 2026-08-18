import Foundation
import HibikiCore
import MLX
import XCTest

/// Parity: the native temporal `forwardText` must reproduce the Python
/// reference's transformer output and predicted tokens on a fixed input, for
/// both the full-precision (BF16) and the quantized (Q8) bundle. The fixtures
/// were produced by the Python implementation in float32 on CPU; this test runs
/// the Swift port the same way.
final class TemporalForwardParityTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    private func repoRoot() -> URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root
    }

    private func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/\(name)")
    }

    private func runTemporalParity(bundleSubpath: String, fixtureName: String) throws {
        let bundleDirectory = repoRoot().appendingPathComponent(bundleSubpath, isDirectory: true)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: bundleDirectory.appendingPathComponent("config.json").path),
            "artifact bundle not present at \(bundleDirectory.path); download it first")

        let fixture = try loadArrays(url: fixtureURL(fixtureName))
        let tokens = try XCTUnwrap(fixture["tokens"])
        let expectedOut = try XCTUnwrap(fixture["transformer_out"]).asType(.float32)
        let expectedArgmax = try XCTUnwrap(fixture["text_argmax"]).asType(.int32)

        let bundle = try ArtifactBundle.validate(directory: bundleDirectory)
        let model = try HibikiLM.loadTemporal(from: bundle)
        let cache = model.makeTransformerCache()
        let (out, logits) = model.forwardText(tokens, cache: cache)
        eval(out, logits)

        let maxAbsDiff = MLX.abs(out.asType(.float32) - expectedOut).max().item(Float.self)
        print("temporal parity (\(bundleSubpath)) max abs diff = \(maxAbsDiff)")
        XCTAssertLessThan(maxAbsDiff, 1e-3, "transformer output diverged from the reference")

        let argmax = logits.argMax(axis: -1).asType(.int32)
        XCTAssertTrue((argmax .== expectedArgmax).all().item(Bool.self),
                      "predicted text tokens diverged from the reference")
    }

    func testTemporalForwardBF16() throws {
        try runTemporalParity(
            bundleSubpath: "artifacts/hibiki-1b-mlx-bf16",
            fixtureName: "temporal_forward.safetensors")
    }

    func testTemporalForwardQ8() throws {
        try runTemporalParity(
            bundleSubpath: "artifacts/hibiki-1b-mlx-q8",
            fixtureName: "temporal_forward_q8.safetensors")
    }
}
