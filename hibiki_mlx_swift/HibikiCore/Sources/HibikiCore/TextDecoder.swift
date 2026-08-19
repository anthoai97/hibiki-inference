import Foundation

/// Incremental decoding of the English text stream.
///
/// SentencePiece pieces do not decode independently: a piece's rendering depends
/// on what precedes it, and the word-boundary marker belongs to the sequence
/// rather than to any one piece. Decoding the accumulated ids and emitting the
/// newly added suffix therefore gives the right text. Mirrors the reference
/// `TextDecoder`.
public struct TextDecoder {
    /// The end-of-padding control id. Unlike the no-text id, the bundle does not
    /// name this one, so it is fixed here.
    public static let endOfPaddingToken = 0

    private let tokenizer: SentencePieceTokenizer
    private let controlTokens: Set<Int>
    private var tokens: [Int] = []
    private var decoded = ""

    /// `noTextToken` has no default: every bundle names its own, and guessing
    /// would silently emit a control id as text.
    public init(tokenizer: SentencePieceTokenizer, noTextToken: Int) {
        self.tokenizer = tokenizer
        self.controlTokens = [TextDecoder.endOfPaddingToken, noTextToken]
    }

    /// Everything decoded so far in this run.
    public var text: String { decoded }

    public mutating func reset() {
        tokens = []
        decoded = ""
    }

    /// Add one sampled token, returning the text it completes, if any.
    public mutating func push(_ token: Int) -> String? {
        if controlTokens.contains(token) { return nil }
        tokens.append(token)
        let full = tokenizer.decode(tokens)
        // Slice by Unicode scalar, matching the reference's code-point slicing —
        // a grapheme-based `dropFirst` could split a multi-scalar cluster.
        let fragment = String(full.unicodeScalars.dropFirst(decoded.unicodeScalars.count))
        decoded = full
        return fragment.isEmpty ? nil : fragment
    }
}
