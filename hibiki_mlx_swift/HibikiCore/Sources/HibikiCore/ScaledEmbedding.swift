import MLX
import MLXNN

/// An embedding that maps a negative "no input" id to an all-zero row, matching
/// the reference `ScaledEmbedding`. The zeroing is expressed as a multiply so it
/// needs no conditional-select op.
final class ScaledEmbedding: Embedding {
    private let zeroIdx: Int32

    init(_ embeddingCount: Int, _ dimensions: Int, zeroIdx: Int32 = -1) {
        self.zeroIdx = zeroIdx
        super.init(embeddingCount: embeddingCount, dimensions: dimensions)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rows = super.callAsFunction(maximum(x, 0))
        let isZero = (x .== zeroIdx).asType(rows.dtype).expandedDimensions(axis: -1)
        return rows * (1 - isZero)
    }
}
