import Foundation

/// The SentencePiece piece table, loaded from the bundle's `.model` file, with
/// just enough to turn sampled text ids back into text.
///
/// Only `decode` is needed: Hibiki generates the English text stream, it never
/// consumes one, so the Unigram/BPE encoder is not ported. The `.model` file is
/// a protobuf whose field 1 is the repeated piece list (each entry a `piece`
/// string and a `type`); the pieces appear in id order. `decode` reproduces
/// `SentencePieceProcessor.decode`: byte pieces (`<0xNN>`) accumulate and decode
/// together as UTF-8 (ill-formed runs become U+FFFD), every other piece renders
/// literally with the word-boundary marker `▁` turned into a space, and a single
/// leading space is removed.
public struct SentencePieceTokenizer {
    /// The piece string for each id, in id order.
    public let pieces: [String]
    /// Whether the piece at each id is a raw byte piece (`<0xNN>`).
    private let isByte: [Bool]

    private static let wordBoundary = "\u{2581}" // ▁
    private static let byteType = 6              // SentencePiece Type.BYTE

    /// Number of pieces in the vocabulary.
    public var count: Int { pieces.count }

    public init(contentsOf url: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ModelLoadError.missingFile(
                "\(url.lastPathComponent) could not be read: \(error.localizedDescription).")
        }

        var reader = ProtobufReader(bytes: [UInt8](data))
        var pieces: [String] = []
        var isByte: [Bool] = []
        while let field = reader.readField() {
            // Field 1 is the repeated SentencePiece list; skip everything else.
            guard field.number == 1, case let .lengthDelimited(sub) = field.value else { continue }
            let (piece, type) = Self.parsePiece(sub)
            pieces.append(piece)
            isByte.append(type == Self.byteType)
        }
        guard !pieces.isEmpty else {
            throw ModelLoadError.shapeMismatch("\(url.lastPathComponent) contains no SentencePiece entries.")
        }
        self.pieces = pieces
        self.isByte = isByte
    }

    /// Turn a sequence of piece ids into text, as `sp.decode` would. Ids outside
    /// the vocabulary are skipped.
    public func decode(_ ids: [Int]) -> String {
        var result = ""
        var byteRun: [UInt8] = []
        func flushBytes() {
            guard !byteRun.isEmpty else { return }
            result += String(decoding: byteRun, as: UTF8.self) // ill-formed -> U+FFFD
            byteRun.removeAll(keepingCapacity: true)
        }
        for id in ids {
            guard id >= 0, id < pieces.count else { continue }
            if isByte[id] {
                if let byte = Self.byteValue(pieces[id]) { byteRun.append(byte) }
            } else {
                flushBytes()
                result += pieces[id].replacingOccurrences(of: Self.wordBoundary, with: " ")
            }
        }
        flushBytes()
        if result.hasPrefix(" ") { result.removeFirst() }
        return result
    }

    /// Parse a `<0xNN>` byte piece into its byte value.
    private static func byteValue(_ piece: String) -> UInt8? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">") else { return nil }
        return UInt8(piece.dropFirst(3).dropLast(), radix: 16)
    }

    /// Read the `piece` (field 1) and `type` (field 3) of one SentencePiece.
    private static func parsePiece(_ bytes: ArraySlice<UInt8>) -> (piece: String, type: Int) {
        var reader = ProtobufReader(bytes: Array(bytes))
        var piece = ""
        var type = 1 // NORMAL
        while let field = reader.readField() {
            switch field.number {
            case 1: if case let .lengthDelimited(sub) = field.value { piece = String(decoding: sub, as: UTF8.self) }
            case 3: if case let .varint(value) = field.value { type = Int(value) }
            default: break
            }
        }
        return (piece, type)
    }
}

/// A minimal reader for the protobuf wire format — enough to walk the pieces of a
/// SentencePiece model without a protobuf runtime.
private struct ProtobufReader {
    enum Value {
        case varint(UInt64)
        case fixed64
        case lengthDelimited(ArraySlice<UInt8>)
        case fixed32
    }

    struct Field {
        let number: Int
        let value: Value
    }

    private let bytes: [UInt8]
    private var index = 0

    init(bytes: [UInt8]) { self.bytes = bytes }

    mutating func readField() -> Field? {
        guard let tag = readVarint() else { return nil }
        let number = Int(tag >> 3)
        switch tag & 0x7 {
        case 0:
            guard let value = readVarint() else { return nil }
            return Field(number: number, value: .varint(value))
        case 1:
            guard advance(by: 8) else { return nil }
            return Field(number: number, value: .fixed64)
        case 2:
            guard let length = readVarint() else { return nil }
            let start = index
            guard advance(by: Int(length)) else { return nil }
            return Field(number: number, value: .lengthDelimited(bytes[start ..< index]))
        case 5:
            guard advance(by: 4) else { return nil }
            return Field(number: number, value: .fixed32)
        default:
            return nil
        }
    }

    private mutating func readVarint() -> UInt64? {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 { return result }
            shift += 7
            if shift >= 64 { return nil }
        }
        return nil
    }

    private mutating func advance(by count: Int) -> Bool {
        guard count >= 0, index + count <= bytes.count else { return false }
        index += count
        return true
    }
}
