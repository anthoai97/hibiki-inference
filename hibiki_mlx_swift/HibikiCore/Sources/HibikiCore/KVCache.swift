import MLX

/// One attention layer's streaming key/value state for a session.
///
/// This is session state, not a model parameter. It grows by concatenation and
/// keeps `offset` (the absolute number of positions seen) so RoPE stays correct.
/// A single prefill from an empty cache returns exactly what the reference does;
/// the bounded/rotating variant used for very long sessions is added with the
/// streaming-parity work.
public final class KVCache {
    public private(set) var offset = 0
    private var keys: MLXArray?
    private var values: MLXArray?

    public init() {}

    /// Append `keys`/`values` (`[batch, heads, steps, headDim]`) and return the
    /// full cached keys and values.
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            keys = concatenated([existingKeys, newKeys], axis: 2)
            values = concatenated([existingValues, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }
        offset += newKeys.dim(2)
        return (keys!, values!)
    }

    public func reset() {
        offset = 0
        keys = nil
        values = nil
    }
}
