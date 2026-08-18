import Foundation
import MLX
import MLXNN

/// The Hibiki language model's **temporal** path: text embedding → Temporal
/// Transformer → output norm → text logits. This is the part of the loaded
/// model needed to prove weights load and a forward pass runs (a "warm eval").
/// The Depth Transformer, audio embeddings, and conditioners are added by the
/// generation-step work.
public final class HibikiLM: Module {
    @ModuleInfo(key: "text_emb") var textEmb: ScaledEmbedding
    @ModuleInfo var transformer: Transformer
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm
    @ModuleInfo(key: "text_linear") var textLinear: Linear

    public let config: HibikiConfig

    public init(config: HibikiConfig) {
        self.config = config
        self._textEmb.wrappedValue = ScaledEmbedding(config.textInVocabSize, config.dim)
        self._transformer.wrappedValue = Transformer(HibikiLM.temporalConfig(config))
        self._outNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: 1e-8)
        self._textLinear.wrappedValue = Linear(config.dim, config.textOutVocabSize, bias: false)
        super.init()
    }

    static func temporalConfig(_ c: HibikiConfig) -> TransformerConfig {
        TransformerConfig(
            dModel: c.dim,
            numHeads: c.numHeads,
            numLayers: c.numLayers,
            dimFeedforward: c.dimFeedforward,
            causal: c.causal,
            positionalEmbedding: c.positionalEmbedding,
            context: c.context,
            maxPeriod: c.maxPeriod)
    }

    public func makeTransformerCache() -> [KVCache] { transformer.makeCache() }

    /// One temporal step (or prefill): returns the transformer output and the
    /// text logits for each input position.
    public func forwardText(_ tokens: MLXArray, cache: [KVCache]) -> (out: MLXArray, logits: MLXArray) {
        let out = outNorm(transformer(textEmb(tokens), caches: cache))
        return (out, textLinear(out))
    }

    /// Load only the temporal-path weights (`text_emb`, `transformer`,
    /// `out_norm`, `text_linear`) from the bundle's Hibiki safetensors. When the
    /// bundle declares weight-only quantization, the Linear layers are quantized
    /// first (matching the reference's `quantize_linear_layers`) so the packed
    /// weight/scales/biases load into them. Float parameters are widened to
    /// float32 for CPU parity; packed integer weights are kept as-is. (The
    /// depth/audio/conditioner weights in the file are ignored here; `update`
    /// runs with no verification so the extra keys are harmless.)
    public static func loadTemporal(from bundle: ArtifactBundle) throws -> HibikiLM {
        let model = HibikiLM(config: bundle.config)
        if let quant = bundle.config.quantization {
            quantize(model: model, groupSize: quant.groupSize, bits: quant.bits) { _, module in
                module is Linear
            }
        }
        let all = try loadArrays(url: bundle.hibikiWeightsURL)
        let prefixes = ["text_emb.", "out_norm.", "text_linear.", "transformer."]
        var temporal: [String: MLXArray] = [:]
        for (key, value) in all where prefixes.contains(where: key.hasPrefix) {
            temporal[key] = value.dtype == .uint32 ? value : value.asType(.float32)
        }
        model.update(parameters: ModuleParameters.unflattened(temporal))
        eval(model)
        return model
    }
}
