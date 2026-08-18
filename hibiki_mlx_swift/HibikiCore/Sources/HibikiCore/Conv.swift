import Foundation
import MLX
import MLXNN

/// Streaming convolutions used by the Mimi codec.
///
/// MLX convolutions are NLC while the released weights and the codec's own
/// tensor layout are NCL, so every layer transposes on the way in and out.
/// Ported from the Python reference; streaming state is a plain `MLXArray?`.

/// A reference holder for an array *derived* from the loaded weights (a codebook
/// centroid, an expanded transposed-conv kernel). MLX reflects every
/// `MLXArray`-typed stored property as a parameter, so keeping derived arrays
/// behind this box keeps them out of the module's parameter tree.
final class DerivedArray {
    var value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

/// A 1D convolution over NCL input, with MLX's `(out, ksize, in)` weight.
final class Conv1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    let stride: Int
    let padding: Int
    let groups: Int
    let dilation: Int

    init(_ inChannels: Int, _ outChannels: Int, _ ksize: Int,
         stride: Int = 1, padding: Int = 0, groups: Int = 1, dilation: Int = 1, bias: Bool = true) {
        self._weight.wrappedValue = MLXArray.zeros([outChannels, ksize, inChannels / groups])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var ys = conv1d(
            xs.swappedAxes(-1, -2), weight,
            stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias { ys = ys + bias }
        return ys.swappedAxes(-1, -2)
    }
}

/// A transposed 1D convolution over NCL input. A depthwise transposed
/// convolution is expanded into a dense one because MLX has no grouped
/// transposed convolution; the expansion depends on the loaded weight, so
/// `refresh()` must run again after each load (via `Mimi.refreshDerivedState`).
final class ConvTranspose1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    let stride: Int
    let padding: Int
    private let groups: Int
    private let ksize: Int
    private let inChannels: Int
    private let outChannels: Int
    private let expanded: DerivedArray
    private var expandedGroups: Int

    init(_ inChannels: Int, _ outChannels: Int, _ ksize: Int,
         stride: Int = 1, padding: Int = 0, groups: Int = 1, bias: Bool = true) {
        let weight = MLXArray.zeros([outChannels / groups, ksize, inChannels])
        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.ksize = ksize
        self.inChannels = inChannels
        self.outChannels = outChannels
        let (weights, expandedGroups) = Self.expand(
            weight, groups: groups, inChannels: inChannels, outChannels: outChannels, ksize: ksize)
        self.expanded = DerivedArray(weights)
        self.expandedGroups = expandedGroups
        super.init()
    }

    /// Rebuild the dense expansion of a depthwise transposed convolution from
    /// the current weight. MLX's `update` writes parameters straight into the
    /// tree without any layer hook, so this runs after every load.
    func refresh() {
        (expanded.value, expandedGroups) = Self.expand(
            weight, groups: groups, inChannels: inChannels, outChannels: outChannels, ksize: ksize)
    }

    private static func expand(_ weight: MLXArray, groups: Int, inChannels: Int, outChannels: Int, ksize: Int)
        -> (MLXArray, Int) {
        if groups == inChannels, groups == outChannels {
            var identity = eye(outChannels).asType(weight.dtype).reshaped([outChannels, 1, outChannels])
            identity = repeated(identity, count: ksize, axis: 1)
            return (repeated(weight, count: groups, axis: 0) * identity, 1)
        } else if groups > 1 {
            fatalError("only depthwise or dense transposed convolutions are supported")
        }
        return (weight, groups)
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var ys = convTransposed1d(
            xs.swappedAxes(-1, -2), expanded.value, stride: stride, padding: padding, groups: expandedGroups)
        if let bias { ys = ys + bias }
        return ys.swappedAxes(-1, -2)
    }
}

/// The released weights keep an unnormalised convolution under `conv`.
final class NormConv1d: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(_ inChannels: Int, _ outChannels: Int, _ ksize: Int,
         stride: Int = 1, padding: Int = 0, groups: Int = 1, dilation: Int = 1, bias: Bool = true) {
        self._conv.wrappedValue = Conv1d(
            inChannels, outChannels, ksize,
            stride: stride, padding: padding, groups: groups, dilation: dilation, bias: bias)
        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
}

/// The transposed counterpart of `NormConv1d`.
final class NormConvTranspose1d: Module {
    @ModuleInfo(key: "convtr") var convtr: ConvTranspose1d

    init(_ inChannels: Int, _ outChannels: Int, _ ksize: Int,
         stride: Int = 1, padding: Int = 0, groups: Int = 1, bias: Bool = true) {
        self._convtr.wrappedValue = ConvTranspose1d(
            inChannels, outChannels, ksize, stride: stride, padding: padding, groups: groups, bias: bias)
        super.init()
    }

    /// The convolution's bias, so streaming callers need not reach for it.
    var bias: MLXArray? { convtr.bias }

    func callAsFunction(_ xs: MLXArray) -> MLXArray { convtr(xs) }
}

private func extraPaddingForConv1d(_ xs: MLXArray, ksize: Int, stride: Int, paddingTotal: Int) -> Int {
    let length = xs.dim(-1)
    let frames = Double(max(length + paddingTotal - ksize, 0)) / Double(stride) + 1.0
    let idealLength = (Int(frames.rounded(.up)) - 1) * stride + ksize - paddingTotal
    return max(0, idealLength - length)
}

private func unpad1d(_ xs: MLXArray, unpadLeft: Int, unpadRight: Int) -> MLXArray {
    xs[.ellipsis, unpadLeft ..< (xs.dim(-1) - unpadRight)]
}

/// A causal convolution that carries its left context between steps.
final class StreamableConv1d: Module {
    @ModuleInfo(key: "conv") var conv: NormConv1d

    private let causal: Bool
    private let padMode: PadMode
    private let ksize: Int
    private let stride: Int
    private let dilation: Int
    private let outChannels: Int
    private var prevXs: MLXArray?
    private var leftPadApplied = false

    init(_ inChannels: Int, _ outChannels: Int, _ ksize: Int,
         stride: Int, dilation: Int, groups: Int, bias: Bool, causal: Bool, padMode: PadMode) {
        self._conv.wrappedValue = NormConv1d(
            inChannels, outChannels, ksize, stride: stride, groups: groups, dilation: dilation, bias: bias)
        self.causal = causal
        self.padMode = padMode
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation
        self.outChannels = outChannels
        super.init()
    }

    func reset() {
        prevXs = nil
        leftPadApplied = false
    }

    /// The non-streaming forward, over a whole sequence at once.
    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let ksize = (self.ksize - 1) * dilation + 1
        let paddingTotal = ksize - stride
        let extra = extraPaddingForConv1d(xs, ksize: ksize, stride: stride, paddingTotal: paddingTotal)
        let none = IntOrPair((0, 0))
        let widths: [IntOrPair]
        if causal {
            widths = [none, none, IntOrPair((paddingTotal, extra))]
        } else {
            let right = paddingTotal / 2
            widths = [none, none, IntOrPair((paddingTotal - right, right + extra))]
        }
        return conv(padded(xs, widths: widths, mode: padMode))
    }

    /// One streaming step: consume `xs` and emit only the frames now complete.
    func step(_ xs: MLXArray) -> MLXArray {
        let batch = xs.dim(0)
        if xs.dim(-1) == 0 { return MLXArray.zeros([batch, outChannels, 0]) }
        let ksize = (self.ksize - 1) * dilation + 1
        var xs = xs
        if !leftPadApplied {
            leftPadApplied = true
            let none = IntOrPair((0, 0))
            xs = padded(xs, widths: [none, none, IntOrPair((ksize - stride, 0))], mode: padMode)
        }
        if let prevXs { xs = concatenated([prevXs, xs], axis: -1) }
        let length = xs.dim(-1)
        let frames = max(length + stride - ksize, 0) / stride
        if frames == 0 {
            prevXs = xs
            return MLXArray.zeros([batch, outChannels, 0])
        }
        prevXs = xs[.ellipsis, (frames * stride)...]
        return conv(xs[.ellipsis, 0 ..< ((frames - 1) * stride + ksize)])
    }
}

/// A causal transposed convolution that carries its output tail.
final class StreamableConvTranspose1d: Module {
    @ModuleInfo(key: "convtr") var convtr: NormConvTranspose1d

    private let causal: Bool
    private let ksize: Int
    private let stride: Int
    private let outChannels: Int
    private var prevYs: MLXArray?

    init(_ inChannels: Int, _ outChannels: Int, _ ksize: Int,
         stride: Int, groups: Int, bias: Bool, causal: Bool) {
        self._convtr.wrappedValue = NormConvTranspose1d(
            inChannels, outChannels, ksize, stride: stride, groups: groups, bias: bias)
        self.causal = causal
        self.ksize = ksize
        self.stride = stride
        self.outChannels = outChannels
        super.init()
    }

    func reset() { prevYs = nil }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let paddingTotal = max(ksize - stride, 0)
        let ys = convtr(xs)
        if causal { return unpad1d(ys, unpadLeft: 0, unpadRight: paddingTotal) }
        let right = paddingTotal / 2
        return unpad1d(ys, unpadLeft: paddingTotal - right, unpadRight: right)
    }

    func step(_ xs: MLXArray) -> MLXArray {
        let batch = xs.dim(0)
        if xs.dim(-1) == 0 { return MLXArray.zeros([batch, outChannels, 0]) }
        var ys = convtr(xs)
        let produced = ys.dim(-1)
        if let prevYs {
            let overlap = prevYs.dim(-1)
            // The bias was already added to the overlapping tail, so remove it
            // before the two contributions are summed.
            var prev = prevYs
            if let bias = convtr.bias { prev = prev - bias[.newAxis, 0..., .newAxis] }
            ys = concatenated([ys[.ellipsis, 0 ..< overlap] + prev, ys[.ellipsis, overlap...]], axis: -1)
        }
        let invalidSteps = ksize - stride
        prevYs = ys[.ellipsis, (produced - invalidSteps)...]
        return ys[.ellipsis, 0 ..< (produced - invalidSteps)]
    }
}

/// The 25 Hz to 12.5 Hz downsample in front of the quantizer.
final class ConvDownsample1d: Module {
    @ModuleInfo(key: "conv") var conv: StreamableConv1d

    init(stride: Int, dim: Int, causal: Bool) {
        self._conv.wrappedValue = StreamableConv1d(
            dim, dim, 2 * stride, stride: stride, dilation: 1, groups: 1, bias: false,
            causal: causal, padMode: .edge)
        super.init()
    }

    func reset() { conv.reset() }
    func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
    func step(_ xs: MLXArray) -> MLXArray { conv.step(xs) }
}

/// The 12.5 Hz to 25 Hz upsample behind the quantizer.
final class ConvTrUpsample1d: Module {
    @ModuleInfo(key: "convtr") var convtr: StreamableConvTranspose1d

    init(stride: Int, dim: Int, causal: Bool) {
        self._convtr.wrappedValue = StreamableConvTranspose1d(
            dim, dim, 2 * stride, stride: stride, groups: dim, bias: false, causal: causal)
        super.init()
    }

    func reset() { convtr.reset() }
    func callAsFunction(_ xs: MLXArray) -> MLXArray { convtr(xs) }
    func step(_ xs: MLXArray) -> MLXArray { convtr.step(xs) }
}
