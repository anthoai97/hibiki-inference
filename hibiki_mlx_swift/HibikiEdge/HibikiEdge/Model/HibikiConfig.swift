import Foundation

/// Error raised when the artifact bundle is missing, unreadable, or does not
/// match the configuration contract.
enum ModelLoadError: LocalizedError {
    case unreadableConfig(String)
    case invalidConfig(String)
    case missingFile(String)
    case shapeMismatch(String)

    var errorDescription: String? {
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
struct HibikiConfig: Decodable {
    // Artifact file names, named by the bundle itself.
    let mimiName: String
    let moshiName: String
    let tokenizerName: String

    // Temporal Transformer.
    let dim: Int
    let numHeads: Int
    let numLayers: Int
    let hiddenScale: Double
    let context: Int
    let maxPeriod: Int
    let causal: Bool
    let norm: String
    let gating: String
    let positionalEmbedding: String

    // Depth Transformer.
    let depformerDim: Int
    let depformerNumHeads: Int
    let depformerNumLayers: Int
    let depformerDimFeedforward: Int
    let depformerContext: Int
    let depformerMaxPeriod: Int
    let depformerWeightsPerStep: Bool
    let depformerPosEmb: String
    let depformerCausal: Bool

    // Vocabularies and streams.
    let textCard: Int
    let existingTextPaddingId: Int
    let card: Int
    let nQ: Int
    let depQ: Int
    let delays: [Int]

    // MARK: Derived contract values

    /// Text embedding rows: the text vocabulary plus one no-text id.
    var textInVocabSize: Int { textCard + 1 }
    /// Text logits width.
    var textOutVocabSize: Int { textCard }
    /// Audio embedding rows: the codebook cardinality plus one padding id.
    var audioVocabSize: Int { card + 1 }
    /// Total audio streams the model models (source + target).
    var audioCodebooks: Int { nQ }
    /// Target-audio codebooks the Depth Transformer samples.
    var targetCodebooks: Int { depQ }
    /// Source-audio codebooks Mimi supplies from the French input.
    var sourceCodebooks: Int { nQ - depQ }
    /// Feed-forward width of the Temporal Transformer.
    var dimFeedforward: Int { Int(hiddenScale * Double(dim)) }

    private static let supportedNorms = ["rms_norm", "rms_norm_f32"]

    /// Reject a bundle this implementation cannot load, echoing the reference's
    /// checks so failures are diagnosable rather than a later shape crash.
    func validate() throws {
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

    static func load(from url: URL) throws -> HibikiConfig {
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
