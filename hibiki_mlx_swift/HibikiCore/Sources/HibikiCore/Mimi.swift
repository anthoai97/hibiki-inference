import Foundation
import MLX
import MLXNN

/// The Mimi codec: SEANet, streaming Transformers, and a split RVQ.
///
/// The bundle's `config.json` says nothing about the codec, so the architecture
/// here is this implementation's explicit contract for the released
/// `mimi-*.safetensors`. Unlike the Hibiki weights, that file is in PyTorch
/// naming and layout, so `remapReleasedMimiWeights` converts it before the
/// strict load.
public struct MimiConfig {
    public let channels: Int
    public let sampleRate: Double
    public let frameRate: Double
    public let seanet: SeanetConfig
    public let transformer: TransformerConfig
    public let quantizerNq: Int
    public let quantizerBins: Int
    public let quantizerDim: Int

    /// The 1,920 samples of 24 kHz PCM that make up one 80 ms frame.
    public var frameSize: Int { Int(sampleRate / frameRate) }
}

/// The released causal 24 kHz Mimi at 12.5 Hz.
public func mimi202407(numCodebooks: Int) -> MimiConfig {
    let seanet = SeanetConfig(
        dimension: 512, channels: 1, causal: true, nfilters: 64, nresidualLayers: 1,
        ratios: [8, 6, 5, 4], ksize: 7, residualKsize: 3, lastKsize: 3, dilationBase: 2,
        padMode: .constant, trueSkip: true, compress: 2)
    let transformer = TransformerConfig(
        dModel: seanet.dimension, numHeads: 8, numLayers: 8, dimFeedforward: 2048,
        causal: true, positionalEmbedding: "rope", context: 250, maxPeriod: 10000,
        layerScale: 0.01, gating: false, norm: "layer_norm", convLayout: true)
    return MimiConfig(
        channels: 1, sampleRate: 24000, frameRate: 12.5, seanet: seanet, transformer: transformer,
        quantizerNq: numCodebooks, quantizerBins: 2048, quantizerDim: 256)
}

/// The codec's weights, plus the convolution state one session streams through.
///
/// Attention caches are handed out by `makeEncoderCache`/`makeDecoderCache`
/// rather than owned here. The streaming convolution state still lives inside
/// the layers, so one loaded codec drives one session until that state is reset.
public final class Mimi: Module {
    @ModuleInfo var encoder: SeanetEncoder
    @ModuleInfo var decoder: SeanetDecoder
    @ModuleInfo var quantizer: SplitResidualVectorQuantizer
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "decoder_transformer") var decoderTransformer: ProjectedTransformer
    @ModuleInfo var downsample: ConvDownsample1d
    @ModuleInfo var upsample: ConvTrUpsample1d

    public let cfg: MimiConfig

    public init(_ cfg: MimiConfig) {
        self.cfg = cfg
        let dim = cfg.seanet.dimension
        let encoderFrameRate = cfg.sampleRate / Double(cfg.seanet.ratios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / cfg.frameRate)
        self._encoder.wrappedValue = SeanetEncoder(cfg.seanet)
        self._decoder.wrappedValue = SeanetDecoder(cfg.seanet)
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: cfg.quantizerDim, inputDim: dim, outputDim: dim, nq: cfg.quantizerNq, bins: cfg.quantizerBins)
        self._encoderTransformer.wrappedValue = ProjectedTransformer(cfg.transformer, inputDim: dim, outputDims: [dim])
        self._decoderTransformer.wrappedValue = ProjectedTransformer(cfg.transformer, inputDim: dim, outputDims: [dim])
        self._downsample.wrappedValue = ConvDownsample1d(stride: downsampleStride, dim: dim, causal: true)
        self._upsample.wrappedValue = ConvTrUpsample1d(stride: downsampleStride, dim: dim, causal: true)
        super.init()
    }

    public func makeEncoderCache() -> [KVCache] { encoderTransformer.makeCache() }
    public func makeDecoderCache() -> [KVCache] { decoderTransformer.makeCache() }

    /// Clear the streaming convolution state held inside the layers.
    public func resetState() {
        encoder.reset()
        decoder.reset()
        downsample.reset()
        upsample.reset()
    }

    /// Turn one PCM step `[B, 1, T]` into audio codes `[B, nq, T']`.
    public func encodeStep(_ xs: MLXArray, cache: [KVCache]) -> MLXArray {
        var ys = encoder.step(xs)
        ys = encoderTransformer(ys, caches: cache)[0]
        return quantizer.encode(downsample.step(ys))
    }

    /// Turn one code step `[B, nq, T']` back into PCM `[B, 1, T]`.
    public func decodeStep(_ xs: MLXArray, cache: [KVCache]) -> MLXArray {
        var ys = upsample.step(quantizer.decode(xs))
        ys = decoderTransformer(ys, caches: cache)[0]
        return decoder.step(ys)
    }

    /// Rebuild state derived from the loaded weights: the codebook centroids and
    /// the expanded depthwise transposed convolutions. MLX's `update` writes
    /// parameters without calling any layer hook, so this runs after each load.
    public func refreshDerivedState() {
        visit(modules: { _, module in
            (module as? EuclideanCodebook)?.refresh()
            (module as? ConvTranspose1d)?.refresh()
        })
    }

    /// Load the codec from the bundle's Mimi safetensors, remapping the released
    /// PyTorch names/layout onto this module's parameter tree first.
    public static func load(from bundle: ArtifactBundle) throws -> Mimi {
        let codebooks = bundle.config.targetCodebooks
        let model = Mimi(mimi202407(numCodebooks: codebooks))
        let raw = try loadArrays(url: bundle.mimiWeightsURL)
        let params = try remapReleasedMimiWeights(raw, codebooks: codebooks)
            .mapValues { $0.dtype == .uint32 ? $0 : $0.asType(.float32) }
        do {
            try model.update(parameters: ModuleParameters.unflattened(params), verify: .all)
        } catch {
            throw ModelLoadError.shapeMismatch(
                "the Mimi weights in \(bundle.mimiWeightsURL.lastPathComponent) do not match the codec contract: \(error)")
        }
        model.refreshDerivedState()
        eval(model)
        return model
    }
}
