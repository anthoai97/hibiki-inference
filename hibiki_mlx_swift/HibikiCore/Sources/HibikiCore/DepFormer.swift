import Foundation
import MLX
import MLXNN

/// One depth position: its own embedding, projections, and Transformer.
///
/// Mirrors the reference `DepFormerSlice`. The slices carry per-step weights, so
/// each depth position runs a distinct Transformer while sharing one scratch KV
/// cache for the frame (see `DepFormer.sample`).
final class DepFormerSlice: Module {
    @ModuleInfo var emb: ScaledEmbedding
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo var transformer: Transformer

    init(
        inVocabSize: Int,
        outVocabSize: Int,
        mainTransformerDim: Int,
        transformerConfig: TransformerConfig
    ) {
        let dim = transformerConfig.dModel
        self._emb.wrappedValue = ScaledEmbedding(inVocabSize, dim)
        self._linearIn.wrappedValue = Linear(mainTransformerDim, dim, bias: false)
        self._linearOut.wrappedValue = Linear(dim, outVocabSize, bias: false)
        self._transformer.wrappedValue = Transformer(transformerConfig)
        super.init()
    }
}

/// Samples the target audio codebooks for one frame, in depth order.
public final class DepFormer: Module {
    @ModuleInfo var slices: [DepFormerSlice]

    init(config: HibikiConfig, transformerConfig: TransformerConfig) {
        self._slices.wrappedValue = (0 ..< config.targetCodebooks).map { index in
            DepFormerSlice(
                // The first slice is conditioned on the sampled text token, the
                // rest on the previous codebook's audio token.
                inVocabSize: index == 0 ? config.textInVocabSize : config.audioVocabSize,
                outVocabSize: config.audioVocabSize - 1,
                mainTransformerDim: config.dim,
                transformerConfig: transformerConfig)
        }
        super.init()
    }

    /// One scratch cache, shared by the slices and reset every frame.
    func makeCache() -> [KVCache] { slices[0].transformer.makeCache() }

    /// Sample the target codebooks for one frame, in depth order.
    ///
    /// Each slice is conditioned on the temporal state and on the token the
    /// previous slice produced, so the cache is scratch state for this frame
    /// alone and is cleared before the walk starts. Returns `[B, depQ, 1]`.
    func sample(_ transformerOut: MLXArray, textToken: MLXArray, cache: [KVCache], sampler: Sampler) -> MLXArray {
        for layerCache in cache { layerCache.reset() }
        var tokens: [MLXArray] = []
        var lastToken = textToken
        for slice in slices {
            let xs = slice.linearIn(transformerOut) + slice.emb(lastToken)
            let out = slice.transformer(xs, caches: cache)
            lastToken = sampler(slice.linearOut(out))
            tokens.append(lastToken)
        }
        return stacked(tokens, axis: 1)
    }
}
