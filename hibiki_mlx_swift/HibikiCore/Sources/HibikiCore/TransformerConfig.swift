import Foundation

/// Configuration for one Transformer stack (temporal or depth). Only the
/// variants the released Hibiki bundle uses are modelled: self-attention with
/// optional RoPE, a gated SiLU feed-forward, RMS norm, no layer scale, no bias.
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

    public var headDim: Int { dModel / numHeads }
}
