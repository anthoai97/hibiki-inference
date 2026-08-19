import HibikiCore
import MLX
import XCTest

/// The `Sampler` is verified by properties rather than exact values (stochastic
/// draws use MLX's global RNG, which need not match the Python reference): greedy
/// equals argmax, top-1 always yields the argmax whatever the temperature, and a
/// plain temperature draw stays in range.
final class SamplerTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    private let logits = MLXArray([0.1, 3.0, -1.0, 2.0] as [Float]).reshaped([1, 4])

    func testGreedyIsArgmax() {
        XCTAssertEqual(Sampler()(logits).item(Int32.self), 1)
    }

    func testTopOneIsArgmaxRegardlessOfTemperature() {
        // Only the single most likely token survives the mask, so the draw is
        // forced onto the argmax no matter the RNG state.
        let token = Sampler(temperature: 1.0, topK: 1)(logits).item(Int32.self)
        XCTAssertEqual(token, 1)
    }

    func testTemperatureStaysInRange() {
        let token = Sampler(temperature: 0.8)(MLXArray.zeros([1, 10])).item(Int32.self)
        XCTAssertTrue(token >= 0 && token < 10)
    }

    func testTopKExcludesLowLogits() {
        // Only the two clearly-largest logits survive top-2, so the draw can
        // never land on a masked index whatever the RNG state.
        let peaked = MLXArray([10.0, 9.0, -100.0, -100.0] as [Float]).reshaped([1, 4])
        let sampler = Sampler(temperature: 1.0, topK: 2)
        for _ in 0 ..< 8 {
            let token = sampler(peaked).item(Int32.self)
            XCTAssertTrue(token == 0 || token == 1, "top-2 draw escaped the top-2 set: \(token)")
        }
    }
}
