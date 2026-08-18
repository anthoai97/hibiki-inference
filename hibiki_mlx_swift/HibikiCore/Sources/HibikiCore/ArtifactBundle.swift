import Foundation

/// A validated artifact bundle on disk: the decoded configuration plus the
/// resolved paths of the weight and tokenizer files, checked to be present and
/// shape-consistent with the configuration.
///
/// This is the load contract's front half — the "minimal validation to avoid
/// loading missing or obviously incompatible files". It stops short of
/// constructing MLX modules or evaluating them; building the loaded model
/// (Temporal/Depth Transformers and the Mimi codec) and running a warm
/// evaluation is the native-inference work that follows.
public struct ArtifactBundle {
    public let directory: URL
    public let config: HibikiConfig
    public let hibikiWeightsURL: URL
    public let mimiWeightsURL: URL
    public let tokenizerURL: URL

    /// Read `config.json`, resolve the three named files, and validate that the
    /// Hibiki and Mimi weight files carry the tensors this contract requires at
    /// the shapes the configuration implies. Throws `ModelLoadError` with a
    /// plain, diagnosable message on any mismatch.
    public static func validate(directory: URL) throws -> ArtifactBundle {
        let config = try HibikiConfig.load(from: directory.appendingPathComponent("config.json"))
        try config.validate()

        let hibiki = try requireFile(directory, config.moshiName, role: "Hibiki weights")
        let mimi = try requireFile(directory, config.mimiName, role: "Mimi weights")
        let tokenizer = try requireFile(directory, config.tokenizerName, role: "tokenizer")

        try validateHibikiShapes(SafetensorsIndex(fileURL: hibiki), config: config)
        try validateMimiPresence(SafetensorsIndex(fileURL: mimi), config: config)

        return ArtifactBundle(
            directory: directory,
            config: config,
            hibikiWeightsURL: hibiki,
            mimiWeightsURL: mimi,
            tokenizerURL: tokenizer)
    }

    private static func requireFile(_ directory: URL, _ name: String, role: String) throws -> URL {
        guard !name.isEmpty else {
            throw ModelLoadError.invalidConfig("config.json does not name its \(role) artifact.")
        }
        let url = directory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelLoadError.missingFile("\(name) is missing, but the bundle names it as \(role).")
        }
        return url
    }

    /// Check the load-bearing Hibiki tensors exist at config-derived shapes.
    /// These few checks catch a truncated, mismatched, or wrong-revision file
    /// before any weights are loaded.
    private static func validateHibikiShapes(_ index: SafetensorsIndex, config: HibikiConfig) throws {
        try expect(index, "text_emb.weight", [config.textInVocabSize, config.dim])
        try expect(index, "text_linear.weight", [config.textOutVocabSize, config.dim])
        try expect(index, "out_norm.weight", [config.dim])

        for codebook in 0..<config.audioCodebooks {
            try expect(index, "audio_embs.\(codebook).weight", [config.audioVocabSize, config.dim])
        }

        for slice in 0..<config.targetCodebooks {
            // The first depth slice is conditioned on the sampled text token, the
            // rest on the previous codebook's audio token.
            let rows = slice == 0 ? config.textInVocabSize : config.audioVocabSize
            try expect(index, "depformer.slices.\(slice).emb.weight", [rows, config.depformerDim])
        }
    }

    private static func validateMimiPresence(_ index: SafetensorsIndex, config: HibikiConfig) throws {
        // The codec is a SEANet encoder/decoder plus a residual vector quantizer;
        // its tensor names are internal, so validate that it is a non-trivial
        // codec bundle rather than pinning exact shapes here.
        guard index.count > 0 else {
            throw ModelLoadError.shapeMismatch("the Mimi weights file contains no tensors.")
        }
    }

    private static func expect(_ index: SafetensorsIndex, _ name: String, _ shape: [Int]) throws {
        guard let actual = index.shape(of: name) else {
            throw ModelLoadError.shapeMismatch("the Hibiki weights are missing '\(name)'.")
        }
        guard actual == shape else {
            throw ModelLoadError.shapeMismatch("'\(name)' has shape \(actual), expected \(shape).")
        }
    }
}
