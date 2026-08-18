import Foundation
import HibikiCore
import XCTest

/// Loaded-model seam tests, front half: config decode + artifact-bundle shape
/// validation, run against the real local bundle. The heavy weights are not in
/// the repo (~4 GB, gitignored), so the bundle-backed test skips when they are
/// absent rather than failing.
final class ArtifactBundleTests: XCTestCase {
    /// The real bundle location: `$HIBIKI_ARTIFACTS`, else the repo's
    /// `artifacts/hibiki-1b-mlx-bf16` relative to this test file.
    private var bundleDirectory: URL {
        if let override = ProcessInfo.processInfo.environment["HIBIKI_ARTIFACTS"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        var root = URL(fileURLWithPath: #filePath)
        // .../HibikiCore/Tests/HibikiCoreTests/ArtifactBundleTests.swift -> repo root
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root.appendingPathComponent("artifacts/hibiki-1b-mlx-bf16", isDirectory: true)
    }

    func testValidatesRealBundle() throws {
        let configPath = bundleDirectory.appendingPathComponent("config.json").path
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: configPath),
            "artifact bundle not present at \(bundleDirectory.path); download it first")

        let bundle = try ArtifactBundle.validate(directory: bundleDirectory)
        XCTAssertEqual(bundle.config.dim, 2048)
        XCTAssertEqual(bundle.config.nQ, 16)
        XCTAssertEqual(bundle.config.depQ, 8)
        XCTAssertEqual(bundle.config.textCard, 48000)
        XCTAssertEqual(bundle.config.sourceCodebooks, bundle.config.targetCodebooks)
        XCTAssertEqual(bundle.config.dimFeedforward, 8448)

        let hibiki = try SafetensorsIndex(fileURL: bundle.hibikiWeightsURL)
        let mimi = try SafetensorsIndex(fileURL: bundle.mimiWeightsURL)
        XCTAssertEqual(hibiki.count, 430)
        XCTAssertEqual(mimi.count, 318)
        XCTAssertEqual(hibiki.shape(of: "text_emb.weight"), [48001, 2048])
    }

    func testMissingBundleRejected() {
        let missing = URL(fileURLWithPath: "/tmp/hibiki-nonexistent-\(UUID().uuidString)")
        XCTAssertThrowsError(try ArtifactBundle.validate(directory: missing))
    }

    func testShapeMismatchRejected() throws {
        // A config whose dim disagrees with the real weights must be rejected by
        // the shape check, not silently accepted.
        let realConfig = bundleDirectory.appendingPathComponent("config.json").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: realConfig), "bundle absent")

        let scratch = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hibiki-bad-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        // Tamper: dim 2048 -> 999, keep everything else.
        let original = try String(contentsOf: URL(fileURLWithPath: realConfig), encoding: .utf8)
        let tampered = original.replacingOccurrences(of: "\"dim\": 2048", with: "\"dim\": 999")
        XCTAssertNotEqual(tampered, original, "expected to rewrite the dim field")
        try tampered.write(to: scratch.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        // Symlink the big weight/tokenizer files so only the config differs.
        for name in ["hibiki-mlx-dc2cf5a5@80.safetensors", "mimi-dbaa9758@125.safetensors", "tokenizer_spm_48k_multi6_2.model"] {
            try FileManager.default.createSymbolicLink(
                at: scratch.appendingPathComponent(name),
                withDestinationURL: bundleDirectory.appendingPathComponent(name))
        }

        XCTAssertThrowsError(try ArtifactBundle.validate(directory: scratch)) { error in
            guard case ModelLoadError.shapeMismatch = error else {
                return XCTFail("expected a shape mismatch, got \(error)")
            }
        }
    }
}
