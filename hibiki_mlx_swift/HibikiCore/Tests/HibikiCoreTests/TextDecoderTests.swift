import Foundation
import HibikiCore
import XCTest

/// The incremental text decoder must accumulate to the same text a one-shot
/// decode of the same ids produces, and must skip control tokens. Needs only the
/// small tokenizer model.
final class TextDecoderTests: XCTestCase {
    private func tokenizer() throws -> SentencePieceTokenizer? {
        for subpath in ["artifacts/hibiki-1b-mlx-bf16", "artifacts/hibiki-1b-mlx-q8"] {
            let directory = MLXTestSupport.bundleURL(subpath)
            let configURL = directory.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else { continue }
            let config = try HibikiConfig.load(from: configURL)
            return try SentencePieceTokenizer(contentsOf: directory.appendingPathComponent(config.tokenizerName))
        }
        return nil
    }

    func testIncrementalMatchesOneShotDecode() throws {
        let tokenizer = try XCTUnwrap(try tokenizer(), "no bundle present to read the tokenizer from")
        var decoder = TextDecoder(tokenizer: tokenizer, noTextToken: 3)

        let ids = [1000, 2000, 3000, 500, 12345]
        var streamed = ""
        for id in ids {
            if let fragment = decoder.push(id) { streamed += fragment }
        }
        XCTAssertEqual(streamed, decoder.text, "fragments did not reconstruct the accumulated text")
        XCTAssertEqual(decoder.text, tokenizer.decode(ids), "incremental decode diverged from one-shot decode")
    }

    func testControlTokensAreSkipped() throws {
        let tokenizer = try XCTUnwrap(try tokenizer(), "no bundle present to read the tokenizer from")
        var decoder = TextDecoder(tokenizer: tokenizer, noTextToken: 3)
        XCTAssertNil(decoder.push(TextDecoder.endOfPaddingToken)) // 0
        XCTAssertNil(decoder.push(3)) // the no-text id
        XCTAssertEqual(decoder.text, "")
    }
}
