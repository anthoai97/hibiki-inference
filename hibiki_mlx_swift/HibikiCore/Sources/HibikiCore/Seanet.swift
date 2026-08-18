import Foundation
import MLX
import MLXNN

/// The SEANet encoder and decoder that surround the Mimi Transformers.
/// Ported from the Python reference; the released config is hard-coded in
/// `mimi202407`.
public struct SeanetConfig {
    public let dimension: Int
    public let channels: Int
    public let causal: Bool
    public let nfilters: Int
    public let nresidualLayers: Int
    public let ratios: [Int]
    public let ksize: Int
    public let residualKsize: Int
    public let lastKsize: Int
    public let dilationBase: Int
    public let padMode: PadMode
    public let trueSkip: Bool
    public let compress: Int
}

/// Adds two streams whose steps may produce different numbers of samples.
/// It carries no weights, only streaming state.
final class StreamingAdd {
    private var lhsTail: MLXArray?
    private var rhsTail: MLXArray?

    func reset() {
        lhsTail = nil
        rhsTail = nil
    }

    func step(_ lhs: MLXArray, _ rhs: MLXArray) -> MLXArray {
        var lhs = lhs
        var rhs = rhs
        if let lhsTail { lhs = concatenated([lhsTail, lhs], axis: -1); self.lhsTail = nil }
        if let rhsTail { rhs = concatenated([rhsTail, rhs], axis: -1); self.rhsTail = nil }
        let lhsLength = lhs.dim(-1)
        let rhsLength = rhs.dim(-1)
        if lhsLength == rhsLength { return lhs + rhs }
        if lhsLength < rhsLength {
            rhsTail = rhs[.ellipsis, lhsLength...]
            return lhs + rhs[.ellipsis, 0 ..< lhsLength]
        }
        lhsTail = lhs[.ellipsis, rhsLength...]
        return lhs[.ellipsis, 0 ..< rhsLength] + rhs
    }
}

final class SeanetResnetBlock: Module {
    @ModuleInfo var block: [StreamableConv1d]
    @ModuleInfo var shortcut: StreamableConv1d?
    private let streamingAdd = StreamingAdd()

    init(_ cfg: SeanetConfig, dim: Int, ksizesAndDilations: [(Int, Int)]) {
        let hidden = dim / cfg.compress
        self._block.wrappedValue = ksizesAndDilations.enumerated().map { index, kd in
            StreamableConv1d(
                index == 0 ? dim : hidden,
                index == ksizesAndDilations.count - 1 ? dim : hidden,
                kd.0, stride: 1, dilation: kd.1, groups: 1, bias: true,
                causal: cfg.causal, padMode: cfg.padMode)
        }
        self._shortcut.wrappedValue = cfg.trueSkip
            ? nil
            : StreamableConv1d(dim, dim, 1, stride: 1, dilation: 1, groups: 1, bias: true,
                               causal: cfg.causal, padMode: cfg.padMode)
        super.init()
    }

    func reset() {
        shortcut?.reset()
        streamingAdd.reset()
        block.forEach { $0.reset() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var ys = xs
        for conv in block { ys = conv(elu(ys, alpha: 1.0)) }
        return ys + (shortcut?(xs) ?? xs)
    }

    func step(_ xs: MLXArray) -> MLXArray {
        var ys = xs
        for conv in block { ys = conv.step(elu(ys, alpha: 1.0)) }
        return streamingAdd.step(ys, shortcut?.step(xs) ?? xs)
    }
}

final class EncoderLayer: Module {
    @ModuleInfo var residuals: [SeanetResnetBlock]
    @ModuleInfo var downsample: StreamableConv1d

    init(_ cfg: SeanetConfig, ratio: Int, mult: Int) {
        var dilation = 1
        var residuals: [SeanetResnetBlock] = []
        for _ in 0 ..< cfg.nresidualLayers {
            residuals.append(SeanetResnetBlock(
                cfg, dim: mult * cfg.nfilters, ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)]))
            dilation *= cfg.dilationBase
        }
        self._residuals.wrappedValue = residuals
        self._downsample.wrappedValue = StreamableConv1d(
            mult * cfg.nfilters, mult * cfg.nfilters * 2, ratio * 2,
            stride: ratio, dilation: 1, groups: 1, bias: true, causal: true, padMode: cfg.padMode)
        super.init()
    }

    func reset() {
        downsample.reset()
        residuals.forEach { $0.reset() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var ys = xs
        for residual in residuals { ys = residual(ys) }
        return downsample(elu(ys, alpha: 1.0))
    }

    func step(_ xs: MLXArray) -> MLXArray {
        var ys = xs
        for residual in residuals { ys = residual.step(ys) }
        return downsample.step(elu(ys, alpha: 1.0))
    }
}

public final class SeanetEncoder: Module {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo var layers: [EncoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
        var mult = 1
        self._initConv1d.wrappedValue = StreamableConv1d(
            cfg.channels, mult * cfg.nfilters, cfg.ksize,
            stride: 1, dilation: 1, groups: 1, bias: true, causal: cfg.causal, padMode: cfg.padMode)
        var layers: [EncoderLayer] = []
        for ratio in cfg.ratios.reversed() {
            layers.append(EncoderLayer(cfg, ratio: ratio, mult: mult))
            mult *= 2
        }
        self._layers.wrappedValue = layers
        self._finalConv1d.wrappedValue = StreamableConv1d(
            mult * cfg.nfilters, cfg.dimension, cfg.lastKsize,
            stride: 1, dilation: 1, groups: 1, bias: true, causal: cfg.causal, padMode: cfg.padMode)
        super.init()
    }

    func reset() {
        initConv1d.reset()
        finalConv1d.reset()
        layers.forEach { $0.reset() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var ys = initConv1d(xs)
        for layer in layers { ys = layer(ys) }
        return finalConv1d(elu(ys, alpha: 1.0))
    }

    func step(_ xs: MLXArray) -> MLXArray {
        var ys = initConv1d.step(xs)
        for layer in layers { ys = layer.step(ys) }
        return finalConv1d.step(elu(ys, alpha: 1.0))
    }
}

final class DecoderLayer: Module {
    @ModuleInfo var upsample: StreamableConvTranspose1d
    @ModuleInfo var residuals: [SeanetResnetBlock]

    init(_ cfg: SeanetConfig, ratio: Int, mult: Int) {
        self._upsample.wrappedValue = StreamableConvTranspose1d(
            mult * cfg.nfilters, mult * cfg.nfilters / 2, ratio * 2,
            stride: ratio, groups: 1, bias: true, causal: cfg.causal)
        var dilation = 1
        var residuals: [SeanetResnetBlock] = []
        for _ in 0 ..< cfg.nresidualLayers {
            residuals.append(SeanetResnetBlock(
                cfg, dim: mult * cfg.nfilters / 2, ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)]))
            dilation *= cfg.dilationBase
        }
        self._residuals.wrappedValue = residuals
        super.init()
    }

    func reset() {
        upsample.reset()
        residuals.forEach { $0.reset() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var ys = upsample(elu(xs, alpha: 1.0))
        for residual in residuals { ys = residual(ys) }
        return ys
    }

    func step(_ xs: MLXArray) -> MLXArray {
        var ys = upsample.step(elu(xs, alpha: 1.0))
        for residual in residuals { ys = residual.step(ys) }
        return ys
    }
}

public final class SeanetDecoder: Module {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo var layers: [DecoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
        var mult = 1 << cfg.ratios.count
        self._initConv1d.wrappedValue = StreamableConv1d(
            cfg.dimension, mult * cfg.nfilters, cfg.ksize,
            stride: 1, dilation: 1, groups: 1, bias: true, causal: cfg.causal, padMode: cfg.padMode)
        var layers: [DecoderLayer] = []
        for ratio in cfg.ratios {
            layers.append(DecoderLayer(cfg, ratio: ratio, mult: mult))
            mult /= 2
        }
        self._layers.wrappedValue = layers
        self._finalConv1d.wrappedValue = StreamableConv1d(
            cfg.nfilters, cfg.channels, cfg.lastKsize,
            stride: 1, dilation: 1, groups: 1, bias: true, causal: cfg.causal, padMode: cfg.padMode)
        super.init()
    }

    func reset() {
        initConv1d.reset()
        finalConv1d.reset()
        layers.forEach { $0.reset() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var ys = initConv1d(xs)
        for layer in layers { ys = layer(ys) }
        return finalConv1d(elu(ys, alpha: 1.0))
    }

    func step(_ xs: MLXArray) -> MLXArray {
        var ys = initConv1d.step(xs)
        for layer in layers { ys = layer.step(ys) }
        return finalConv1d.step(elu(ys, alpha: 1.0))
    }
}
