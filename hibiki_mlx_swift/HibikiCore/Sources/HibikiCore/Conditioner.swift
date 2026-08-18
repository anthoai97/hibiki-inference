import Foundation
import MLX
import MLXNN

/// The lookup-table conditioner that carries Hibiki's quality label.
///
/// The released bundle declares one `description` conditioner whose value is a
/// label such as `very_good`. Its projected embedding is added at every time
/// step, so the same tensor is reused for a whole session. Mirrors the reference
/// `LutConditioner`; `learntPadding` is present in the weights and kept so the
/// parameter tree matches, though this happy-path never reads it.
final class LutConditioner: Module {
    @ModuleInfo var embed: Embedding
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    @ParameterInfo(key: "learnt_padding") var learntPadding: MLXArray

    private let possibleValues: [String: Int]
    /// The first declared value, used to exercise this path during warm-up.
    let anyValue: String?

    init(outputDim: Int, config: LutConditionerConfig) {
        self._embed.wrappedValue = Embedding(embeddingCount: config.nBins + 1, dimensions: config.dim)
        self._outputProj.wrappedValue = Linear(config.dim, outputDim, bias: false)
        self._learntPadding.wrappedValue = MLXArray.zeros([1, 1, outputDim])
        self.possibleValues = Dictionary(
            uniqueKeysWithValues: config.possibleValues.enumerated().map { ($1, $0) })
        self.anyValue = config.possibleValues.first
        super.init()
    }

    /// The `[1, outputDim]` embedding for one label value.
    func condition(_ value: String) throws -> MLXArray {
        guard let index = possibleValues[value] else {
            throw ModelLoadError.invalidConfig(
                "unknown condition '\(value)'; expected one of \(possibleValues.keys.sorted())")
        }
        return outputProj(embed(MLXArray([Int32(index)])))
    }
}

/// Holds the model's conditioners and hands out their condition tensors.
final class ConditionProvider: Module {
    @ModuleInfo var conditioners: [String: LutConditioner]

    init(outputDim: Int, configs: [String: LutConditionerConfig]) {
        self._conditioners.wrappedValue = configs.mapValues {
            LutConditioner(outputDim: outputDim, config: $0)
        }
        super.init()
    }

    /// The `[1, outputDim]` tensor for `value` under conditioner `name`.
    func conditionTensor(_ name: String, _ value: String) throws -> MLXArray {
        guard let conditioner = conditioners[name] else {
            throw ModelLoadError.invalidConfig("unknown conditioner '\(name)'")
        }
        return try conditioner.condition(value)
    }

    /// A valid condition tensor from any conditioner, for warm-up. Nil if no
    /// conditioner declares any value. Non-throwing: `anyValue` is always a
    /// declared value, so `condition` cannot reject it.
    func anyConditionTensor() -> MLXArray? {
        for conditioner in conditioners.values {
            if let value = conditioner.anyValue {
                return try? conditioner.condition(value)
            }
        }
        return nil
    }
}
