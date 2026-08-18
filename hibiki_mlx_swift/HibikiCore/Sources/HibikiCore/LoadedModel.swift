import Foundation
import MLX

/// The immutable weights of one artifact bundle, ready to translate.
///
/// This is the loaded-model seam an inference session (#24) is started from: it
/// owns the Mimi codec, the Hibiki generator, and the text tokenizer, and no
/// per-run streaming state — a session asks the codec and generator for their
/// caches when it starts, so one loaded model can back many sessions.
public final class LoadedModel {
    public let config: HibikiConfig
    public let mimi: Mimi
    public let lm: HibikiLM
    public let tokenizer: SentencePieceTokenizer

    private init(config: HibikiConfig, mimi: Mimi, lm: HibikiLM, tokenizer: SentencePieceTokenizer) {
        self.config = config
        self.mimi = mimi
        self.lm = lm
        self.tokenizer = tokenizer
    }

    /// Load the Mimi and Hibiki weights and the tokenizer from a validated bundle.
    public static func load(from bundle: ArtifactBundle) throws -> LoadedModel {
        let mimi = try Mimi.load(from: bundle)
        let lm = try HibikiLM.load(from: bundle)
        let tokenizer = try SentencePieceTokenizer(contentsOf: bundle.tokenizerURL)
        return LoadedModel(config: bundle.config, mimi: mimi, lm: lm, tokenizer: tokenizer)
    }

    /// Validate a bundle directory, then load it.
    public static func load(directory: URL) throws -> LoadedModel {
        try load(from: ArtifactBundle.validate(directory: directory))
    }

    /// Run one frame through every fixed-shape path of the generator.
    ///
    /// This is the "warm / first evaluation": it forces cold Metal-kernel
    /// compilation before a caller starts streaming, and proves the loaded
    /// weights actually evaluate. It uses fresh caches and keeps no state.
    public func warmup() {
        let text = MLXArray([Int32(config.existingTextPaddingId)]).reshaped([1, 1])
        let audioPad = Int32(config.audioPaddingToken)
        let audio = (0 ..< config.audioCodebooks).map { _ in MLXArray([audioPad]).reshaped([1, 1]) }
        let condition = lm.conditionProvider.flatMap { $0.anyConditionTensor() }
        let greedy = Sampler()
        let (textToken, audioTokens) = lm.sampleStep(
            textTokenIds: text,
            audioTokenIds: audio,
            transformerCache: lm.makeTransformerCache(),
            depformerCache: lm.makeDepformerCache(),
            textSampler: greedy,
            audioSampler: greedy,
            condition: condition)
        eval(textToken, audioTokens)
    }
}
