import Foundation
import HibikiCore
import MLX
import XCTest

/// Parity: the native Mimi codec must reproduce the Python reference's streaming
/// encode (PCM → codes) and decode (codes → PCM) on a fixed tone. The fixture
/// carries the exact input samples so both sides stream identical audio; it was
/// produced in float32 on CPU (see `scripts/fixtures`).
final class MimiParityTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.forceCPUDevice()
    }

    func testEncodeDecodeRoundTrip() throws {
        let bundleDirectory = MLXTestSupport.bundleURL("artifacts/hibiki-1b-mlx-q8")
        try MLXTestSupport.requireBundle(bundleDirectory)

        let bundle = try ArtifactBundle.validate(directory: bundleDirectory)
        let mimi = try Mimi.load(from: bundle)

        let fixture = try loadArrays(url: MLXTestSupport.fixtureURL("mimi_roundtrip.safetensors"))
        let inputPCM = try XCTUnwrap(fixture["input_pcm"]).asType(.float32)
        let expectedCodes = try XCTUnwrap(fixture["codes"]).asType(.int32)
        let expectedPCM = try XCTUnwrap(fixture["output_pcm"]).asType(.float32)

        let frameSize = mimi.cfg.frameSize
        let frames = inputPCM.dim(-1) / frameSize

        // Streaming encode, frame by frame.
        mimi.resetState()
        let encoderCache = mimi.makeEncoderCache()
        var codeSteps: [MLXArray] = []
        for index in 0 ..< frames {
            let frame = inputPCM[0..., 0..., (index * frameSize) ..< ((index + 1) * frameSize)]
            let step = mimi.encodeStep(frame, cache: encoderCache)
            if step.dim(-1) > 0 { codeSteps.append(step) }
        }
        let codes = concatenated(codeSteps, axis: -1)
        eval(codes)
        XCTAssertEqual(codes.shape, expectedCodes.shape, "code shape diverged from the reference")
        XCTAssertTrue((codes .== expectedCodes).all().item(Bool.self),
                      "encoded codes diverged from the reference")

        // Streaming decode, one code step at a time.
        let decoderCache = mimi.makeDecoderCache()
        var pcmSteps: [MLXArray] = []
        for index in 0 ..< codes.dim(-1) {
            pcmSteps.append(mimi.decodeStep(codes[0..., 0..., index ..< (index + 1)], cache: decoderCache))
        }
        let pcm = concatenated(pcmSteps, axis: -1)
        eval(pcm)
        XCTAssertEqual(pcm.shape, expectedPCM.shape, "decoded PCM shape diverged from the reference")
        let maxAbsDiff = MLX.abs(pcm - expectedPCM).max().item(Float.self)
        print("mimi round-trip PCM max abs diff = \(maxAbsDiff)")
        XCTAssertLessThan(maxAbsDiff, 1e-3, "decoded PCM diverged from the reference")
    }
}
