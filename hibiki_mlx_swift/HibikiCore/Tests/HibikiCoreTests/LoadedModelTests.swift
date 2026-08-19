import Foundation
import HibikiCore
import XCTest

/// The loaded-model seam: the whole Hibiki generator plus the tokenizer load
/// from the real bundle, the module shapes conform to the config, and a warm
/// evaluation succeeds. The heavy weights are gitignored, so these skip when the
/// bundle is absent.
final class LoadedModelTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    /// `$HIBIKI_ARTIFACTS`, else the repo's Q8 bundle (the current target).
    private var bundleDirectory: URL {
        MLXTestSupport.defaultBundleURL(fallback: "artifacts/hibiki-1b-mlx-q8")
    }

    func testLoadsAndWarmsUp() throws {
        try MLXTestSupport.requireBundle(bundleDirectory)

        // A strict load (verify: .all) already proves every module parameter
        // conforms to the config-derived shape and nothing in the file is left
        // over; reaching here is that guarantee.
        let model = try LoadedModel.load(directory: bundleDirectory)

        XCTAssertEqual(model.config.dim, 2048)
        XCTAssertEqual(model.config.nQ, 16)
        XCTAssertEqual(model.config.depQ, 8)
        XCTAssertGreaterThan(model.tokenizer.count, 0)

        // The warm / first evaluation succeeds and holds no per-run state: a
        // second warm-up from the same immutable model runs just the same.
        model.warmup()
        model.warmup()
    }

    func testMissingBundleRejected() {
        let missing = URL(fileURLWithPath: "/tmp/hibiki-nonexistent-\(UUID().uuidString)")
        XCTAssertThrowsError(try LoadedModel.load(directory: missing))
    }
}
