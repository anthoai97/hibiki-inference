import Foundation
import MLX
import MLXNN

/// The Hibiki language model: the **temporal** path (text embedding → Temporal
/// Transformer → output norm → text logits) plus the **depth** path (audio
/// embeddings, the per-step Depth Transformer, and the optional conditioner)
/// used to sample the audio codebooks for a frame.
///
/// The parameter tree is the load contract: every `@ModuleInfo` name matches a
/// tensor name in the released Hibiki safetensors exactly. The model is
/// immutable and owns no streaming state — attention caches are handed out by
/// `makeTransformerCache()` / `makeDepformerCache()` and belong to a session.
public final class HibikiLM: Module {
    @ModuleInfo(key: "text_emb") var textEmb: ScaledEmbedding
    @ModuleInfo var transformer: Transformer
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm
    @ModuleInfo(key: "text_linear") var textLinear: Linear
    @ModuleInfo(key: "audio_embs") var audioEmbs: [ScaledEmbedding]
    @ModuleInfo var depformer: DepFormer
    @ModuleInfo(key: "condition_provider") var conditionProvider: ConditionProvider?

    public let config: HibikiConfig

    public init(config: HibikiConfig) {
        self.config = config
        self._textEmb.wrappedValue = ScaledEmbedding(config.textInVocabSize, config.dim)
        self._transformer.wrappedValue = Transformer(HibikiLM.temporalConfig(config))
        self._outNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: 1e-8)
        self._textLinear.wrappedValue = Linear(config.dim, config.textOutVocabSize, bias: false)
        self._audioEmbs.wrappedValue = (0 ..< config.audioCodebooks).map { _ in
            ScaledEmbedding(config.audioVocabSize, config.dim)
        }
        self._depformer.wrappedValue = DepFormer(
            config: config, transformerConfig: HibikiLM.depformerConfig(config))
        let luts = config.lutConditioners
        self._conditionProvider.wrappedValue = luts.isEmpty
            ? nil
            : ConditionProvider(outputDim: config.dim, configs: luts)
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

    static func depformerConfig(_ c: HibikiConfig) -> TransformerConfig {
        TransformerConfig(
            dModel: c.depformerDim,
            numHeads: c.depformerNumHeads,
            numLayers: c.depformerNumLayers,
            dimFeedforward: c.depformerDimFeedforward,
            causal: c.depformerCausal,
            positionalEmbedding: c.depformerPosEmb,
            context: c.depformerContext,
            maxPeriod: c.depformerMaxPeriod)
    }

    public func makeTransformerCache() -> [KVCache] { transformer.makeCache() }
    public func makeDepformerCache() -> [KVCache] { depformer.makeCache() }

    /// The `[1, dim]` condition tensor for `value` under conditioner `name`, or
    /// nil when the bundle declares no conditioner. Added at every time step.
    public func conditionTensor(_ name: String, _ value: String) throws -> MLXArray? {
        try conditionProvider?.conditionTensor(name, value)
    }

    /// One temporal step (or prefill): returns the transformer output and the
    /// text logits for each input position.
    public func forwardText(_ tokens: MLXArray, cache: [KVCache]) -> (out: MLXArray, logits: MLXArray) {
        let out = outNorm(transformer(textEmb(tokens), caches: cache))
        return (out, textLinear(out))
    }

    /// The temporal state for one frame: the summed text, audio, and condition
    /// embeddings run through the Temporal Transformer and output norm.
    /// `condition` is the `[1, dim]` tensor from the conditioner, or nil.
    public func frameState(
        textTokenIds: MLXArray,
        audioTokenIds: [MLXArray],
        cache: [KVCache],
        condition: MLXArray?
    ) -> MLXArray {
        var xs = textEmb(textTokenIds)
        for (tokenIds, embedding) in zip(audioTokenIds, audioEmbs) {
            xs = xs + embedding(tokenIds)
        }
        if let condition {
            xs = xs + condition.expandedDimensions(axis: 1)
        }
        return outNorm(transformer(xs, caches: cache))
    }

    /// Advance one frame: sample the text token, then the audio codebooks.
    ///
    /// `textTokenIds` is `[B, 1]` and `audioTokenIds` holds one `[B, 1]` column
    /// per audio stream, already placed at its delayed position by the caller
    /// (the delayed-stream scheduler is #23). The returned audio tokens are
    /// `[B, depQ, 1]`.
    public func sampleStep(
        textTokenIds: MLXArray,
        audioTokenIds: [MLXArray],
        transformerCache: [KVCache],
        depformerCache: [KVCache],
        textSampler: Sampler,
        audioSampler: Sampler,
        condition: MLXArray?
    ) -> (textToken: MLXArray, audioTokens: MLXArray) {
        let out = frameState(
            textTokenIds: textTokenIds, audioTokenIds: audioTokenIds,
            cache: transformerCache, condition: condition)
        let textToken = textSampler(textLinear(out))
        let audioTokens = depformer.sample(out, textToken: textToken, cache: depformerCache, sampler: audioSampler)
        return (textToken, audioTokens)
    }

    /// Load the whole Hibiki model from the bundle's safetensors.
    ///
    /// When the bundle declares weight-only quantization, the compatible Linear
    /// layers are quantized first (matching the reference's
    /// `quantize_linear_layers`) so the packed weight/scales/biases load into
    /// them. Float parameters are widened to float32 for CPU parity; packed
    /// integer weights are kept as-is. The load is strict: it fails with a
    /// diagnosable error if the file and the model contract disagree.
    public static func load(from bundle: ArtifactBundle) throws -> HibikiLM {
        let model = HibikiLM(config: bundle.config)
        if let quant = bundle.config.quantization {
            quantizeLMLinears(model, quant)
        }
        try loadWeights(into: model, from: bundle.hibikiWeightsURL, keeping: nil, strict: true)
        eval(model)
        return model
    }

    /// Load only the temporal-path weights (`text_emb`, `transformer`,
    /// `out_norm`, `text_linear`) from the bundle's Hibiki safetensors. Used by
    /// the temporal parity test; `load` is the production path.
    public static func loadTemporal(from bundle: ArtifactBundle) throws -> HibikiLM {
        let model = HibikiLM(config: bundle.config)
        if let quant = bundle.config.quantization {
            quantizeLMLinears(model, quant)
        }
        try loadWeights(
            into: model, from: bundle.hibikiWeightsURL,
            keeping: ["text_emb.", "out_norm.", "text_linear.", "transformer."],
            strict: false)
        eval(model)
        return model
    }

    /// Quantize the LM Linear layers that use the packed MLX layout, replicating
    /// the reference `compatible_linear`: a 2-D Linear weight whose dimensions are
    /// all multiples of 32 and whose input width is a multiple of the group size.
    /// This deliberately skips the conditioner's narrow `output_proj` and every
    /// embedding, which stay full precision in the released weights.
    static func quantizeLMLinears(_ model: Module, _ quant: QuantizationSpec) {
        quantize(model: model, groupSize: quant.groupSize, bits: quant.bits) { _, module in
            guard let linear = module as? Linear else { return false }
            let shape = linear.weight.shape
            return shape.count == 2
                && shape.allSatisfy { $0 % 32 == 0 }
                && shape.last! % quant.groupSize == 0
        }
    }

    /// Read the safetensors, optionally filter to `keeping` prefixes, widen float
    /// tensors to float32 (keeping packed `uint32` weights), and update the model.
    private static func loadWeights(
        into model: HibikiLM, from url: URL, keeping prefixes: [String]?, strict: Bool
    ) throws {
        let raw = try loadArrays(url: url)
        var params: [String: MLXArray] = [:]
        for (key, value) in raw {
            if let prefixes, !prefixes.contains(where: key.hasPrefix) { continue }
            params[key] = value.dtype == .uint32 ? value : value.asType(.float32)
        }
        guard strict else {
            model.update(parameters: ModuleParameters.unflattened(params))
            return
        }
        do {
            try model.update(parameters: ModuleParameters.unflattened(params), verify: .all)
        } catch {
            throw ModelLoadError.shapeMismatch(
                "the Hibiki weights in \(url.lastPathComponent) do not match the model contract: \(error)")
        }
    }
}
