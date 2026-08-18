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

/// Weight-only quantization of the language-model Linear layers, as declared
/// by the bundle's `hibiki_mlx_quantization` block. Embeddings, norms, and the
/// Mimi codec stay unquantized.
public struct QuantizationSpec: Decodable, Equatable {
    public let format: String
    public let target: String
    public let bits: Int
    public let groupSize: Int
}

/// A lookup-table conditioner declared by the bundle: a bin embedding projected
/// to the model dimension and added at every time step. Hibiki uses one such
/// conditioner ("description") to carry a quality label like `very_good`.
public struct LutConditionerConfig: Decodable {
    public let nBins: Int
    public let dim: Int
    public let tokenizer: String
    public let possibleValues: [String]
}

/// One entry of the config's `conditioners` map. Only the `lut` type is used by
/// the released bundle; `validate()` rejects anything else.
public struct ConditionerConfig: Decodable {
    public let type: String
    public let lut: LutConditionerConfig?
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

    /// Conditioners the model adds to its input, keyed by name. The released
    /// bundle declares one "description" LUT conditioner.
    public let conditioners: [String: ConditionerConfig]?

    /// Present when the LM Linear layers are quantized (e.g. the Q8 bundle).
    public let hibikiMlxQuantization: QuantizationSpec?

    /// Weight-only LM quantization, or nil for a full-precision bundle.
    public var quantization: QuantizationSpec? { hibikiMlxQuantization }

    // MARK: Derived contract values

    /// Text embedding rows: the text vocabulary plus one no-text id.
    public var textInVocabSize: Int { textCard + 1 }
    /// Text logits width.
    public var textOutVocabSize: Int { textCard }
    /// Audio embedding rows: the codebook cardinality plus one padding id.
    public var audioVocabSize: Int { card + 1 }
    /// The audio padding id (the last audio embedding row).
    public var audioPaddingToken: Int { audioVocabSize - 1 }
    /// Total audio streams the model models (source + target).
    public var audioCodebooks: Int { nQ }
    /// Target-audio codebooks the Depth Transformer samples.
    public var targetCodebooks: Int { depQ }
    /// Source-audio codebooks Mimi supplies from the French input.
    public var sourceCodebooks: Int { nQ - depQ }
    /// Feed-forward width of the Temporal Transformer.
    public var dimFeedforward: Int { Int(hiddenScale * Double(dim)) }

    /// The LUT conditioners to build, keyed by name (empty if the bundle
    /// declares none). `validate()` has already rejected any unsupported entry.
    public var lutConditioners: [String: LutConditionerConfig] {
        var result: [String: LutConditionerConfig] = [:]
        for (name, entry) in conditioners ?? [:] {
            if let lut = entry.lut { result[name] = lut }
        }
        return result
    }

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
        for (name, entry) in conditioners ?? [:] {
            guard entry.type == "lut" else {
                throw ModelLoadError.invalidConfig("unsupported conditioner type '\(entry.type)' for '\(name)'")
            }
            guard let lut = entry.lut else {
                throw ModelLoadError.invalidConfig("conditioner '\(name)' is declared 'lut' but carries no lut block")
            }
            guard lut.tokenizer == "noop" else {
                throw ModelLoadError.invalidConfig("unsupported conditioner tokenizer '\(lut.tokenizer)' for '\(name)'")
            }
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
