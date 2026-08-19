import MLX
import MLXRandom

/// Turns one head's logits into one token per batch entry.
///
/// Mirrors the reference `Sampler`: greedy when `temperature == 0` (used by the
/// parity fixtures, which must be deterministic), otherwise temperature scaling
/// with an optional top-k restriction and a categorical draw. Sampling draws on
/// MLX's global random state, so a run is reproducible only for a fixed seed and
/// a fixed order of calls. The reference's top-p path is not ported — the session
/// uses top-k — so only `temperature` and `topK` are exposed.
public struct Sampler {
    public let temperature: Float
    public let topK: Int?

    public init(temperature: Float = 0, topK: Int? = nil) {
        self.temperature = temperature
        self.topK = topK
    }

    public func callAsFunction(_ logits: MLXArray) -> MLXArray {
        if temperature == 0 {
            return logits.argMax(axis: -1).asType(.int32)
        }
        var scaled = logits * (1.0 / temperature)
        if let topK, topK > 0, topK < logits.dim(-1) {
            // Mask everything outside the top-k most likely tokens to -inf.
            let outside = argPartition(-scaled, kth: topK - 1, axis: -1)[.ellipsis, topK...]
            scaled = putAlong(scaled, outside, values: MLXArray(-Float.infinity), axis: -1)
        }
        return MLXRandom.categorical(scaled, axis: -1).asType(.int32)
    }
}
