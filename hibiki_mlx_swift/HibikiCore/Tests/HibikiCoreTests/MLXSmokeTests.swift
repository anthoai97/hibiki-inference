import MLX
import XCTest

/// Proves MLX actually evaluates in this test process — i.e. `swift test` on
/// macOS can run MLX. This is the capability the whole HibikiCore test harness
/// depends on: it lets the loaded-model and inference-session seams be verified
/// here instead of only on a physical iPhone. Runs on the CPU backend (see
/// `MLXTestSupport`).
final class MLXSmokeTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    func testMLXEvaluatesOnThisMachine() {
        let sum = MLXArray([1, 2, 3]).sum().item(Int.self)
        XCTAssertEqual(sum, 6)
    }
}
