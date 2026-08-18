import Foundation
import MLX
import MLXNN

/// Additive causal mask: 0 where a query may attend to a key, a large negative
/// value where it may not. Queries are the last `queryLen` of `keyLen` keys.
private func causalMask(queryLen: Int, keyLen: Int, dtype: DType) -> MLXArray {
    let queryStart = keyLen - queryLen
    let queries = MLXArray((0 ..< queryLen).map { Int32(queryStart + $0) }).reshaped(queryLen, 1)
    let keys = MLXArray((0 ..< keyLen).map { Int32($0) }).reshaped(1, keyLen)
    let allowed = (keys .<= queries).asType(dtype) // [queryLen, keyLen]: 1 allowed, 0 forbidden
    return (allowed - 1) * Float(1e9) // 0 allowed, -1e9 forbidden
}

/// Self-attention over a bounded context with a fused QKV projection.
final class Attention: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let rope: RoPE?

    private let cfg: TransformerConfig
    private let scale: Float

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.scale = pow(Float(cfg.headDim), -0.5)
        self._inProj.wrappedValue = Linear(cfg.dModel, 3 * cfg.dModel, bias: false)
        self._outProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: false)
        if cfg.positionalEmbedding == "rope" {
            self.rope = RoPE(dimensions: cfg.headDim, traditional: true, base: Float(cfg.maxPeriod))
        } else {
            self.rope = nil
        }
        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: KVCache) -> MLXArray {
        let batch = xs.dim(0)
        let steps = xs.dim(1)
        let dim = xs.dim(2)

        // [b, s, 3*d] -> [3, b, heads, s, headDim]
        let qkv = inProj(xs)
            .reshaped(batch, steps, 3, cfg.numHeads, cfg.headDim)
            .transposed(2, 0, 3, 1, 4)
        var queries = qkv[0]
        var keys = qkv[1]
        let values = qkv[2]

        if let rope {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        }

        var (allKeys, allValues) = cache.update(keys: keys, values: values)
        var keyLen = allKeys.dim(2)
        let targetLen = steps + min(cfg.context, keyLen - steps)
        if targetLen < keyLen {
            allKeys = allKeys[0..., 0..., (keyLen - targetLen) ..< keyLen]
            allValues = allValues[0..., 0..., (keyLen - targetLen) ..< keyLen]
            keyLen = targetLen
        }

        // A single streaming step needs no mask: every cached key precedes it.
        let mask: MLXArray? = (cfg.causal && steps > 1)
            ? causalMask(queryLen: steps, keyLen: keyLen, dtype: xs.dtype)
            : nil

        var ys = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues, scale: scale, mask: mask)
        ys = ys.transposed(0, 2, 1, 3).reshaped(batch, steps, dim)
        return outProj(ys)
    }
}

/// Gated SiLU feed-forward whose two branches share one input projection.
final class MlpGating: Module, UnaryLayer {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(_ cfg: TransformerConfig) {
        let hidden = 2 * cfg.dimFeedforward / 3
        self._linearIn.wrappedValue = Linear(cfg.dModel, 2 * hidden, bias: false)
        self._linearOut.wrappedValue = Linear(hidden, cfg.dModel, bias: false)
        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let projected = linearIn(xs)
        let batch = projected.dim(0)
        let steps = projected.dim(1)
        // [b, s, 2*hidden] -> [2, b, s, hidden]
        let halves = projected.reshaped(batch, steps, 2, -1).transposed(2, 0, 1, 3)
        return linearOut(silu(halves[0]) * halves[1])
    }
}

/// The plain GELU feed-forward used by the Mimi Transformers.
final class MlpNoGating: Module, UnaryLayer {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(_ cfg: TransformerConfig) {
        self._linear1.wrappedValue = Linear(cfg.dModel, cfg.dimFeedforward, bias: false)
        self._linear2.wrappedValue = Linear(cfg.dimFeedforward, cfg.dModel, bias: false)
        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        linear2(geluApproximate(linear1(xs)))
    }
}

/// A learnable per-channel scale applied to a residual branch (Mimi only).
final class LayerScale: Module, UnaryLayer {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ dModel: Int, initValue: Float) {
        self._scale.wrappedValue = MLXArray.ones([dModel]) * initValue
        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray { xs * scale }
}

/// The norm the config selects: RMS norm for Hibiki, layer norm for Mimi.
private func makeNorm(_ cfg: TransformerConfig) -> UnaryLayer {
    switch cfg.norm {
    case "layer_norm": return LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
    default: return RMSNorm(dimensions: cfg.dModel, eps: 1e-8) // rms_norm / rms_norm_f32
    }
}

final class TransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Attention
    @ModuleInfo var gating: UnaryLayer
    @ModuleInfo var norm1: UnaryLayer
    @ModuleInfo var norm2: UnaryLayer
    @ModuleInfo(key: "layer_scale_1") var layerScale1: LayerScale?
    @ModuleInfo(key: "layer_scale_2") var layerScale2: LayerScale?

    init(_ cfg: TransformerConfig) {
        self._selfAttn.wrappedValue = Attention(cfg)
        self._gating.wrappedValue = cfg.gating ? MlpGating(cfg) : MlpNoGating(cfg)
        self._norm1.wrappedValue = makeNorm(cfg)
        self._norm2.wrappedValue = makeNorm(cfg)
        if let scale = cfg.layerScale {
            self._layerScale1.wrappedValue = LayerScale(cfg.dModel, initValue: scale)
            self._layerScale2.wrappedValue = LayerScale(cfg.dModel, initValue: scale)
        } else {
            self._layerScale1.wrappedValue = nil
            self._layerScale2.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: KVCache) -> MLXArray {
        var attn = selfAttn(norm1(xs), cache: cache)
        if let layerScale1 { attn = layerScale1(attn) }
        let xs = xs + attn
        var ff = gating(norm2(xs))
        if let layerScale2 { ff = layerScale2(ff) }
        return xs + ff
    }
}

public final class Transformer: Module {
    @ModuleInfo var layers: [TransformerLayer]
    private let cfg: TransformerConfig

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self._layers.wrappedValue = (0 ..< cfg.numLayers).map { _ in TransformerLayer(cfg) }
        super.init()
    }

    func callAsFunction(_ xs: MLXArray, caches: [KVCache]) -> MLXArray {
        var xs = xs
        for (layer, cache) in zip(layers, caches) {
            xs = layer(xs, cache: cache)
        }
        return xs
    }

    func makeCache() -> [KVCache] { layers.map { _ in KVCache() } }
}

/// A Transformer with optional input/output projections, used by the Mimi codec.
///
/// The Mimi Transformers run at the SEANet dimension, so both projections are
/// absent from the released weights; `convLayout` swaps the codec's NCL tensors
/// into the Transformer's NLC layout for the duration of the stack.
public final class ProjectedTransformer: Module {
    private let convLayout: Bool
    @ModuleInfo var transformer: Transformer
    @ModuleInfo(key: "input_proj") var inputProj: Linear?
    @ModuleInfo(key: "output_projs") var outputProjs: [Linear?]

    init(_ cfg: TransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout
        self._transformer.wrappedValue = Transformer(cfg)
        self._inputProj.wrappedValue = inputDim == cfg.dModel ? nil : Linear(inputDim, cfg.dModel, bias: false)
        self._outputProjs.wrappedValue = outputDims.map { dim in
            dim == cfg.dModel ? nil : Linear(cfg.dModel, dim, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ xs: MLXArray, caches: [KVCache]) -> [MLXArray] {
        var xs = convLayout ? xs.swappedAxes(1, 2) : xs
        if let inputProj { xs = inputProj(xs) }
        xs = transformer(xs, caches: caches)
        return outputProjs.map { proj in
            let out = proj?(xs) ?? xs
            return convLayout ? out.swappedAxes(1, 2) : out
        }
    }

    func makeCache() -> [KVCache] { transformer.makeCache() }
}
