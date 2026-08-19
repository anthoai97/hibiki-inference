import Foundation
import MLX
import MLXNN

/// The split residual vector quantizer at the centre of the Mimi codec.
///
/// The released codebooks are stored as an unnormalised `embedding_sum` and a
/// `cluster_usage` count rather than as centroids, so the usable embedding is
/// derived after the weights are loaded (via `Mimi.refreshDerivedState`).

/// One codebook, addressed by nearest centroid.
final class EuclideanCodebook: Module {
    @ParameterInfo(key: "initialized") var initialized: MLXArray
    @ParameterInfo(key: "embedding_sum") var embeddingSum: MLXArray
    @ParameterInfo(key: "cluster_usage") var clusterUsage: MLXArray

    private let dim: Int
    private let epsilon: Float = 1e-5
    private let embeddingBox: DerivedArray
    private let c2Box: DerivedArray

    private var embedding: MLXArray { embeddingBox.value }
    private var c2: MLXArray { c2Box.value }

    init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self._initialized.wrappedValue = MLXArray.zeros([1])
        let embeddingSum = MLXArray.zeros([codebookSize, dim])
        let clusterUsage = MLXArray.zeros([codebookSize])
        self._embeddingSum.wrappedValue = embeddingSum
        self._clusterUsage.wrappedValue = clusterUsage
        let (embedding, c2) = Self.derive(embeddingSum, clusterUsage, epsilon: 1e-5)
        self.embeddingBox = DerivedArray(embedding)
        self.c2Box = DerivedArray(c2)
        super.init()
    }

    /// Recompute the centroids and their squared norms from the loaded weights.
    func refresh() {
        (embeddingBox.value, c2Box.value) = Self.derive(embeddingSum, clusterUsage, epsilon: epsilon)
    }

    private static func derive(_ embeddingSum: MLXArray, _ clusterUsage: MLXArray, epsilon: Float)
        -> (MLXArray, MLXArray) {
        let usage = maximum(clusterUsage, epsilon).expandedDimensions(axis: -1)
        let embedding = embeddingSum / usage
        return (embedding, embedding.square().sum(axis: -1) / 2)
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        let prefix = Array(xs.shape.dropLast())
        let flat = xs.reshaped([-1, dim])
        let dot = matmul(flat, embedding.swappedAxes(-1, -2))
        return (c2 - dot).argMin(axis: -1).reshaped(prefix)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        let targetShape = xs.shape + [dim]
        return take(embedding, xs.reshaped([-1]), axis: 0).reshaped(targetShape)
    }
}

final class VectorQuantization: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear?
    @ModuleInfo(key: "project_out") var projectOut: Linear?
    @ModuleInfo var codebook: EuclideanCodebook

    init(dim: Int, codebookSize: Int, codebookDim: Int?) {
        let cbDim = codebookDim ?? dim
        if dim == cbDim {
            self._projectIn.wrappedValue = nil
            self._projectOut.wrappedValue = nil
        } else {
            self._projectIn.wrappedValue = Linear(dim, cbDim)
            self._projectOut.wrappedValue = Linear(cbDim, dim)
        }
        self._codebook.wrappedValue = EuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
        super.init()
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var ys = xs.swappedAxes(-1, -2)
        if let projectIn { ys = projectIn(ys) }
        return codebook.encode(ys)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        var ys = codebook.decode(xs)
        if let projectOut { ys = projectOut(ys) }
        return ys.swappedAxes(-1, -2)
    }
}

/// Codebooks applied in sequence, each quantizing the previous residual.
final class ResidualVectorQuantization: Module {
    @ModuleInfo var layers: [VectorQuantization]

    init(nq: Int, dim: Int, codebookSize: Int, codebookDim: Int?) {
        self._layers.wrappedValue = (0 ..< nq).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
        super.init()
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = xs
        for layer in layers {
            let indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            codes.append(indices)
        }
        return stacked(codes, axis: 0)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        var quantized = layers[0].decode(xs[0])
        for index in 1 ..< xs.dim(0) {
            quantized = quantized + layers[index].decode(xs[index])
        }
        return quantized
    }
}

final class ResidualVectorQuantizer: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Conv1d?
    @ModuleInfo(key: "output_proj") var outputProj: Conv1d?
    @ModuleInfo var vq: ResidualVectorQuantization

    init(dim: Int, inputDim: Int?, outputDim: Int?, nq: Int, bins: Int, forceProjection: Bool) {
        let inD = inputDim ?? dim
        let outD = outputDim ?? dim
        self._inputProj.wrappedValue = (inD != dim || forceProjection) ? Conv1d(inD, dim, 1, bias: false) : nil
        self._outputProj.wrappedValue = (outD != dim || forceProjection) ? Conv1d(dim, outD, 1, bias: false) : nil
        self._vq.wrappedValue = ResidualVectorQuantization(nq: nq, dim: dim, codebookSize: bins, codebookDim: nil)
        super.init()
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var ys = xs
        if let inputProj { ys = inputProj(ys) }
        return vq.encode(ys).swappedAxes(0, 1)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        var quantized = vq.decode(xs.swappedAxes(0, 1))
        if let outputProj { quantized = outputProj(quantized) }
        return quantized
    }
}

/// The semantic first codebook, split from the acoustic remainder.
public final class SplitResidualVectorQuantizer: Module {
    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    private let nq: Int

    init(dim: Int, inputDim: Int?, outputDim: Int?, nq: Int, bins: Int) {
        self.nq = nq
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dim: dim, inputDim: inputDim, outputDim: outputDim, nq: 1, bins: bins, forceProjection: true)
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            dim: dim, inputDim: inputDim, outputDim: outputDim, nq: nq - 1, bins: bins, forceProjection: true)
        super.init()
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var codes = rvqFirst.encode(xs)
        if nq > 1 { codes = concatenated([codes, rvqRest.encode(xs)], axis: 1) }
        return codes
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        var quantized = rvqFirst.decode(xs[0..., 0 ..< 1])
        if nq > 1 { quantized = quantized + rvqRest.decode(xs[0..., 1...]) }
        return quantized
    }
}
