import MLX

/// Turns one head's logits into one token per batch entry.
///
/// The loaded-model warm eval and the parity fixtures are deterministic, so only
/// greedy (argmax) sampling is implemented here. Temperature / top-k / top-p
/// sampling belongs to the streaming generation loop and arrives with #23; the
/// type carries `temperature` now so that work can extend it without changing
/// call sites.
public struct Sampler {
    public let temperature: Float

    public init(temperature: Float = 0) {
        self.temperature = temperature
    }

    public func callAsFunction(_ logits: MLXArray) -> MLXArray {
        precondition(
            temperature == 0,
            "only greedy sampling is implemented; temperature sampling arrives with the generation loop (#23)")
        return logits.argMax(axis: -1).asType(.int32)
    }
}
