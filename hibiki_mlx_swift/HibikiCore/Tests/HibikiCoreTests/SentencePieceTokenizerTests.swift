import Foundation
import HibikiCore
import XCTest

/// The pure-Swift SentencePiece decoder must reproduce `SentencePieceProcessor.
/// decode` for the released tokenizer. The fixture pairs id sequences with the
/// reference decode output (including a byte-fallback case). Only the small
/// `.model` file is needed, so this runs without the multi-gigabyte weights.
final class SentencePieceTokenizerTests: XCTestCase {
    private struct DecodeFixture: Decodable {
        let vocabSize: Int
        let cases: [Case]
        struct Case: Decodable {
            let ids: [Int]
            let text: String
        }
    }

    private func tokenizerURL() throws -> URL? {
        for subpath in ["artifacts/hibiki-1b-mlx-bf16", "artifacts/hibiki-1b-mlx-q8"] {
            let directory = MLXTestSupport.bundleURL(subpath)
            let configURL = directory.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else { continue }
            let config = try HibikiConfig.load(from: configURL)
            return directory.appendingPathComponent(config.tokenizerName)
        }
        return nil
    }

    private func loadFixture() throws -> DecodeFixture {
        let url = MLXTestSupport.fixtureURL("tokenizer_decode.json")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(DecodeFixture.self, from: Data(contentsOf: url))
    }

    func testDecodeMatchesReference() throws {
        let url = try XCTUnwrap(try tokenizerURL(), "no bundle present to read the tokenizer from")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: url.path),
            "tokenizer model not present at \(url.path)")

        let tokenizer = try SentencePieceTokenizer(contentsOf: url)
        let fixture = try loadFixture()

        XCTAssertEqual(tokenizer.count, fixture.vocabSize, "piece table size diverged from the reference")
        for testCase in fixture.cases {
            XCTAssertEqual(
                tokenizer.decode(testCase.ids), testCase.text,
                "decode diverged for ids \(testCase.ids)")
        }
    }

    func testMissingModelThrows() {
        let missing = URL(fileURLWithPath: "/tmp/hibiki-no-tokenizer-\(UUID().uuidString).model")
        XCTAssertThrowsError(try SentencePieceTokenizer(contentsOf: missing))
    }
}
