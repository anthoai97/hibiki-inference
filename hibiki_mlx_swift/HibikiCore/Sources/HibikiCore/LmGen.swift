import Foundation
import MLX

/// The delayed-stream schedule that drives one generation step at a time.
///
/// Seven of the eight codebooks in each audio stream are delayed by two frames,
/// so the tokens fed to the model at step `t` come from three different
/// positions on the model timeline, and the audio that becomes complete at step
/// `t` belongs to frame `t - 2`. Only the positions still in play are kept, so a
/// run's length is not bounded by the scheduler.
///
/// This owns the per-run generation state (the rolling token window and the
/// attention caches); the loaded model stays immutable. Classifier-free guidance
/// is deliberately absent (see `docs/inference-architecture.md`).
public final class LmGen {
    /// Marks a position written before it was read. Negative so it can never
    /// collide with a real token.
    static let ungeneratedToken: Int32 = -2

    private let model: HibikiLM
    private let textSampler: Sampler
    private let audioSampler: Sampler
    private let batchSize: Int

    private let audioDelays: [Int]
    private let maxDelay: Int
    private let targetCodebooks: Int
    private let audioCodebooks: Int
    private let audioPaddingToken: Int32
    private let textStartToken: Int32
    /// A position is written at `t` and again at `t + maxDelay`, and last read at
    /// `t + maxDelay + 1`, so this many columns are live.
    private let window: Int

    private let transformerCache: [KVCache]
    private let depformerCache: [KVCache]

    private var stepIdx = 0
    private var genSequence: MLXArray

    public init(model: HibikiLM, textSampler: Sampler, audioSampler: Sampler, batchSize: Int = 1) {
        self.model = model
        self.textSampler = textSampler
        self.audioSampler = audioSampler
        self.batchSize = batchSize

        let cfg = model.config
        self.audioDelays = Array(cfg.delays.dropFirst()) // the first delay is the text stream
        self.maxDelay = audioDelays.max() ?? 0
        self.targetCodebooks = cfg.targetCodebooks
        self.audioCodebooks = cfg.audioCodebooks
        self.audioPaddingToken = Int32(cfg.audioPaddingToken)
        self.textStartToken = Int32(cfg.textOutVocabSize)
        self.window = maxDelay + 2
        self.transformerCache = model.makeTransformerCache()
        self.depformerCache = model.makeDepformerCache()
        // Fresh caches (offset 0) and stepIdx 0 already are the reset state.
        self.genSequence = LmGen.emptySequence(batchSize: batchSize, rows: 1 + audioCodebooks, window: window)
    }

    /// The frame the next step will produce text for.
    public var textFrameIndex: Int { stepIdx }
    /// The frame `lastAudioTokens()` refers to after a step.
    public var audioFrameIndex: Int { stepIdx - 1 - maxDelay }

    /// Return the schedule to the state it had before any audio arrived.
    public func reset() {
        stepIdx = 0
        genSequence = LmGen.emptySequence(batchSize: batchSize, rows: 1 + audioCodebooks, window: window)
        transformerCache.forEach { $0.reset() }
        depformerCache.forEach { $0.reset() }
    }

    /// Advance one frame from the source tokens Mimi just produced.
    ///
    /// `sourceTokens` is `[B, sourceCodebooks]` for the current frame. Returns
    /// the sampled text token as `[B, 1]`; the audio for frame `t - 2` is then
    /// read with `lastAudioTokens()`.
    public func step(sourceTokens: MLXArray, condition: MLXArray?) -> MLXArray {
        let column = stepIdx % window
        // This column last held position `stepIdx - window`, already read by all.
        genSequence[0..., 0..., column] = MLXArray(LmGen.ungeneratedToken)
        genSequence[0..., (1 + targetCodebooks)..., column] = sourceTokens

        let textTokens = stepIdx == 0
            ? LmGen.filled([batchSize, 1], textStartToken)
            : read(stream: 0, position: stepIdx - 1, what: "text")
        let audioTokens = audioDelays.enumerated().map { codebook, delay in
            read(stream: codebook + 1, position: stepIdx - 1 - delay, what: "audio codebook \(codebook)")
        }

        let (textToken, generated) = model.sampleStep(
            textTokenIds: textTokens,
            audioTokenIds: audioTokens,
            transformerCache: transformerCache,
            depformerCache: depformerCache,
            textSampler: textSampler,
            audioSampler: audioSampler,
            condition: condition)

        genSequence[0..., 0, column] = textToken.reshaped([batchSize])
        for (codebook, delay) in audioDelays.prefix(targetCodebooks).enumerated() {
            let position = stepIdx - delay
            if position >= 0 {
                genSequence[0..., codebook + 1, position % window] = generated[0..., codebook, 0]
            }
        }
        stepIdx += 1
        return textToken
    }

    /// The newest target frame whose codebooks are all present — frame `t - 2`,
    /// so the first two steps of a session return nil.
    public func lastAudioTokens() -> MLXArray? {
        let position = stepIdx - 1 - maxDelay
        if position < 0 { return nil }
        let tokens = genSequence[0..., 1 ..< (1 + targetCodebooks), position % window]
        // A padding token means this frame is not real audio.
        if (tokens .== audioPaddingToken).any().item(Bool.self) { return nil }
        // Ungenerated here would mean a scheduling bug; fail loudly rather than
        // silently drop a real target frame (mirrors the reference's ScheduleError).
        precondition(!(tokens .== LmGen.ungeneratedToken).any().item(Bool.self),
                     "target audio frame \(position) was never generated")
        return tokens
    }

    /// Read one stream at one position as a `[B, 1]` model input.
    private func read(stream: Int, position: Int, what: String) -> MLXArray {
        if position < 0 {
            return LmGen.filled([batchSize, 1], audioPaddingToken)
        }
        let tokens = genSequence[0..., stream, position % window].expandedDimensions(axis: -1)
        precondition(!(tokens .== LmGen.ungeneratedToken).any().item(Bool.self),
                     "\(what) at frame \(position) was read at step \(stepIdx) before anything wrote it")
        return tokens
    }

    private static func emptySequence(batchSize: Int, rows: Int, window: Int) -> MLXArray {
        filled([batchSize, rows, window], ungeneratedToken)
    }

    private static func filled(_ shape: [Int], _ value: Int32) -> MLXArray {
        MLXArray.full(shape, values: MLXArray(value), type: Int32.self)
    }
}
