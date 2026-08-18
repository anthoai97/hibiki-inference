import Foundation

/// Error raised when the artifact bundle is missing, unreadable, or does not
/// match the configuration contract.
public enum ModelLoadError: LocalizedError {
    case unreadableConfig(String)
    case invalidConfig(String)
    case missingFile(String)
    case shapeMismatch(String)

    public var errorDescription: String? {
        switch self {
        case let .unreadableConfig(message): return message
        case let .invalidConfig(message): return message
        case let .missingFile(message): return message
        case let .shapeMismatch(message): return message
        }
    }
}

/// The bundle's own `config.json`, decoded and checked against the
/// contract this implementation supports. Mirrors `LmConfig.from_config_dict`
/// in the Python reference: the released weights are already in MLX naming, so
/// these values are the load contract every tensor name and shape must match.
public struct HibikiConfig: Decodable {
    // Artifact file names, named by the bundle itself.
    public let mimiName: String
    public let moshiName: String
    public let tokenizerName: String

    // Temporal Transformer.
    public let dim: Int
    public let numHeads: Int
    public let numLayers: Int
    public let hiddenScale: Double
    public let context: Int
    public let maxPeriod: Int
    public let causal: Bool
    public let norm: String
    public let gating: String
    public let positionalEmbedding: String

    // Depth Transformer.
    public let depformerDim: Int
    public let depformerNumHeads: Int
    public let depformerNumLayers: Int
    public let depformerDimFeedforward: Int
    public let depformerContext: Int
    public let depformerMaxPeriod: Int
    public let depformerWeightsPerStep: Bool
    public let depformerPosEmb: String
    public let depformerCausal: Bool

    // Vocabularies and streams.
    public let textCard: Int
    public let existingTextPaddingId: Int
    public let card: Int
    public let nQ: Int
    public let depQ: Int
    public let delays: [Int]

    // MARK: Derived contract values

    /// Text embedding rows: the text vocabulary plus one no-text id.
    public var textInVocabSize: Int { textCard + 1 }
    /// Text logits width.
    public var textOutVocabSize: Int { textCard }
    /// Audio embedding rows: the codebook cardinality plus one padding id.
    public var audioVocabSize: Int { card + 1 }
    /// Total audio streams the model models (source + target).
    public var audioCodebooks: Int { nQ }
    /// Target-audio codebooks the Depth Transformer samples.
    public var targetCodebooks: Int { depQ }
    /// Source-audio codebooks Mimi supplies from the French input.
    public var sourceCodebooks: Int { nQ - depQ }
    /// Feed-forward width of the Temporal Transformer.
    public var dimFeedforward: Int { Int(hiddenScale * Double(dim)) }

    private static let supportedNorms = ["rms_norm", "rms_norm_f32"]

    /// Reject a bundle this implementation cannot load, echoing the reference's
    /// checks so failures are diagnosable rather than a later shape crash.
    public func validate() throws {
        guard Self.supportedNorms.contains(norm) else {
            throw ModelLoadError.invalidConfig("unsupported norm '\(norm)'")
        }
        guard gating == "silu" else {
            throw ModelLoadError.invalidConfig("unsupported gating '\(gating)'")
        }
        guard depformerWeightsPerStep else {
            throw ModelLoadError.invalidConfig("this implementation expects per-step Depth Transformer weights")
        }
        guard delays.count == nQ + 1 else {
            throw ModelLoadError.invalidConfig(
                "expected \(nQ + 1) delays for one text and \(nQ) audio streams, found \(delays.count)")
        }
        guard sourceCodebooks == targetCodebooks else {
            throw ModelLoadError.invalidConfig(
                "the bundle generates \(targetCodebooks) audio codebooks but supplies "
                + "\(sourceCodebooks), so one codec cannot serve both streams")
        }
    }

    public static func load(from url: URL) throws -> HibikiConfig {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ModelLoadError.unreadableConfig("\(url.lastPathComponent) could not be read: \(error.localizedDescription). Download the artifacts first.")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        do {
            return try decoder.decode(HibikiConfig.self, from: data)
        } catch {
            throw ModelLoadError.invalidConfig("\(url.lastPathComponent) is not a valid Hibiki configuration: \(error)")
        }
    }
}
