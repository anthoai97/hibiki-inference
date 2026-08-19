import Foundation
import MLX

/// Convert released Mimi codec weights into this implementation's parameters.
///
/// The released file is in PyTorch naming and layout. Names are remapped onto
/// this module tree (SEANet `model.N` sequence positions → named layers, the
/// dropped residual activations, the PyTorch attention/MLP names) and conv
/// weights are transposed from PyTorch's NCL layout to MLX's. The released file
/// carries every codebook Mimi was trained with; Hibiki uses only the first
/// `codebooks`, so the surplus `rvq_rest` codebooks are dropped.

/// Map a released SEANet sequence index onto this implementation's layer name.
/// Both stacks are one initial convolution, four resampling stages, and one
/// final convolution; the released files number them by their position in a
/// PyTorch `Sequential` that also holds the activations.
private func seanetLayout(upsampling: Bool) -> [Int: String] {
    var layout: [Int: String] = [0: "init_conv1d", 14: "final_conv1d"]
    for index in 0 ..< 4 {
        if upsampling {
            layout[2 + 3 * index] = "layers.\(index).upsample"
            layout[3 + 3 * index] = "layers.\(index).residuals.0"
        } else {
            layout[1 + 3 * index] = "layers.\(index).residuals.0"
            layout[3 + 3 * index] = "layers.\(index).downsample"
        }
    }
    return layout
}

private let encoderLayout = seanetLayout(upsampling: false)
private let decoderLayout = seanetLayout(upsampling: true)
// The residual block's activations are dropped, so its two convolutions move
// from released positions 1 and 3 down to 0 and 1.
private let seanetBlockIndex = ["1": "0", "3": "1"]

private func remapName(_ raw: String) throws -> String {
    // The released names mark private submodules with a leading underscore,
    // which MLX would read as "not a parameter".
    var name = raw.components(separatedBy: ".")
        .map { $0.hasPrefix("_") ? String($0.dropFirst()) : $0 }
        .joined(separator: ".")

    for (side, layout) in [("encoder", encoderLayout), ("decoder", decoderLayout)] {
        let prefix = "\(side).model."
        guard name.hasPrefix(prefix) else { continue }
        let body = String(name.dropFirst(prefix.count))
        let dot = body.firstIndex(of: ".") ?? body.endIndex
        guard let index = Int(body[..<dot]), let target = layout[index] else {
            throw ModelLoadError.shapeMismatch("Mimi weight '\(raw)' is at an unexpected \(side) position.")
        }
        var parts = String(body[dot...].dropFirst()).components(separatedBy: ".")
        if parts.first == "block" {
            guard let blockIndex = seanetBlockIndex[parts[1]] else {
                throw ModelLoadError.shapeMismatch("Mimi weight '\(raw)' is at an unexpected residual position.")
            }
            parts[1] = blockIndex
        }
        name = "\(side).\(target)." + parts.joined(separator: ".")
        break
    }

    if name.hasSuffix(".in_proj_weight") {
        name = String(name.dropLast(".in_proj_weight".count)) + ".in_proj.weight"
    }
    for linear in ["linear1", "linear2"] where name.hasSuffix(".\(linear).weight") {
        name = String(name.dropLast(".\(linear).weight".count)) + ".gating.\(linear).weight"
    }
    return name
}

private func remapValue(_ name: String, _ value: MLXArray) -> MLXArray {
    // PyTorch convolutions are (out, in, ksize) and MLX's are (out, ksize, in);
    // transposed convolutions are (in, out, ksize) against MLX's (out, ksize, in).
    let isQuantizerProj = name.hasPrefix("quantizer.")
        && (name.hasSuffix("input_proj.weight") || name.hasSuffix("output_proj.weight"))
    if name.hasSuffix(".conv.weight") || isQuantizerProj {
        return value.swappedAxes(-1, -2)
    }
    if name.hasSuffix(".convtr.weight") {
        return value.transposed(1, 2, 0)
    }
    return value
}

/// Remap released codec weights, dropping the surplus `rvq_rest` codebooks.
func remapReleasedMimiWeights(_ raw: [String: MLXArray], codebooks: Int) throws -> [String: MLXArray] {
    let keptRest = codebooks - 1
    let restPrefix = "quantizer.rvq_rest.vq.layers."
    var weights: [String: MLXArray] = [:]
    for (rawName, value) in raw {
        let name = try remapName(rawName)
        if name.hasPrefix(restPrefix) {
            let indexText = name.dropFirst(restPrefix.count).prefix { $0 != "." }
            if let index = Int(indexText), index >= keptRest { continue }
        }
        weights[name] = remapValue(name, value)
    }
    return weights
}
