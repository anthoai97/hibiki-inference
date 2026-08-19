import Foundation

/// Configuration for one Transformer stack. Covers both variants the released
/// bundle uses: the Hibiki temporal/depth stacks (gated SiLU feed-forward, RMS
/// norm, no layer scale) and the Mimi codec stacks (plain GELU feed-forward,
/// layer norm, a layer scale, and a conv NCL⇄NLC layout swap). The defaults are
/// the Hibiki values, so its call sites are unaffected.
public struct TransformerConfig {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let dimFeedforward: Int
    public let causal: Bool
    /// "rope" or "none".
    public let positionalEmbedding: String
    public let context: Int
    public let maxPeriod: Int

    /// A learnable per-channel scale applied after each residual branch, or nil
    /// for none. The Mimi stacks use 0.01; the Hibiki stacks use none.
    public let layerScale: Float?
    /// true: gated SiLU feed-forward (Hibiki). false: plain GELU (Mimi).
    public let gating: Bool
    /// "rms_norm" (Hibiki) or "layer_norm" (Mimi).
    public let norm: String
    /// Whether the stack runs inside a codec's NCL layout and must swap to NLC.
    public let convLayout: Bool

    public init(
        dModel: Int,
        numHeads: Int,
        numLayers: Int,
        dimFeedforward: Int,
        causal: Bool,
        positionalEmbedding: String,
        context: Int,
        maxPeriod: Int,
        layerScale: Float? = nil,
        gating: Bool = true,
        norm: String = "rms_norm",
        convLayout: Bool = false
    ) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.dimFeedforward = dimFeedforward
        self.causal = causal
        self.positionalEmbedding = positionalEmbedding
        self.context = context
        self.maxPeriod = maxPeriod
        self.layerScale = layerScale
        self.gating = gating
        self.norm = norm
        self.convLayout = convLayout
    }

    public var headDim: Int { dModel / numHeads }
}
