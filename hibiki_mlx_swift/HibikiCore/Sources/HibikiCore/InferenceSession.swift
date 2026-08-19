import Foundation
import MLX

/// Raised when a session is used in a way its lifecycle forbids.
public enum SessionError: LocalizedError {
    case finished

    public var errorDescription: String? {
        switch self {
        case .finished: return "this session has been finished; call reset() to reuse it"
        }
    }
}

/// Wall-clock time spent in each part of one generation step.
///
/// Same phases as the Python `StepTiming`: source encode, Hibiki generate,
/// target decode, and text decode. `totalSeconds` is their sum. Only filled
/// when the session was created with `measureTiming: true`; `eval` is called
/// before each stopwatch so GPU work is included.
public struct StepTiming: Equatable {
    public let sourceEncodeSeconds: Double
    public let generationSeconds: Double
    public let targetDecodeSeconds: Double
    public let textDecodeSeconds: Double

    public var totalSeconds: Double {
        sourceEncodeSeconds + generationSeconds + targetDecodeSeconds + textDecodeSeconds
    }

    public init(
        sourceEncodeSeconds: Double,
        generationSeconds: Double,
        targetDecodeSeconds: Double,
        textDecodeSeconds: Double
    ) {
        self.sourceEncodeSeconds = sourceEncodeSeconds
        self.generationSeconds = generationSeconds
        self.targetDecodeSeconds = targetDecodeSeconds
        self.textDecodeSeconds = textDecodeSeconds
    }

    /// One monitor line in the Python `--metrics` layout, without the memory
    /// suffix (MLX Swift does not expose the same counters).
    public func formatted(textFrameIndex: Int, audioFrameIndex: Int?) -> String {
        let audio = audioFrameIndex.map(String.init) ?? "-"
        return "step=\(textFrameIndex) text_frame=\(textFrameIndex) audio_frame=\(audio) phases: "
            + "encode=\(StepTiming.milliseconds(sourceEncodeSeconds)) "
            + "generate=\(StepTiming.milliseconds(generationSeconds)) "
            + "decode=\(StepTiming.milliseconds(targetDecodeSeconds)) "
            + "text=\(StepTiming.milliseconds(textDecodeSeconds)) "
            + "total=\(StepTiming.milliseconds(totalSeconds))"
    }

    public static func milliseconds(_ seconds: Double) -> String {
        String(format: "%.1fms", seconds * 1000)
    }
}

/// What one generation step produced.
///
/// Text and audio leave a step on different positions of the model timeline: the
/// text belongs to frame `t`; the audio that becomes complete during the same
/// call belongs to frame `t - 2`, because seven of the eight codebooks are
/// delayed. Both indices are kept rather than pretending the two are synchronous.
public struct StepResult {
    public let textFrameIndex: Int
    public let textToken: Int
    public let text: String?
    public let audioFrameIndex: Int?
    public let pcm: [Float]?
    public let secondsPerFrame: Double
    public let timing: StepTiming?

    /// The text's model time, in seconds.
    public var textTime: Double { Double(textFrameIndex) * secondsPerFrame }
    /// The target audio frame's model time, in seconds.
    public var audioTime: Double? { audioFrameIndex.map { Double($0) * secondsPerFrame } }
}

/// One streaming translation run over a loaded model: push French PCM in, get
/// English text and English PCM out.
///
/// The session owns everything that changes while French speech is translated:
/// the codec's caches and convolution state, the delayed-stream schedule, the
/// Transformer caches, the text decoder, and the frame buffer. The loaded model
/// stays immutable and can start another session after `reset()`.
public final class InferenceSession {
    /// Frames of silence pushed after the input ends, so the delayed codebooks
    /// of the last real frames can still be completed. An explicit fallback, not
    /// learned end-of-stream behaviour.
    public static let silenceTailFrames = 6

    private let mimi: Mimi
    private let generator: LmGen
    private var textDecoder: TextDecoder
    private let condition: MLXArray?
    private let frameSize: Int
    private let secondsPerFrame: Double
    private let encoderCache: [KVCache]
    private let decoderCache: [KVCache]
    private var pending: [Float] = []
    private var finished = false
    private let measureTiming: Bool

    public init(
        model: LoadedModel,
        condition: String? = "very_good",
        textSampler: Sampler = Sampler(temperature: 0.8, topK: 25),
        audioSampler: Sampler = Sampler(temperature: 0.8, topK: 250),
        measureTiming: Bool = false
    ) throws {
        self.mimi = model.mimi
        self.frameSize = model.mimi.cfg.frameSize
        self.secondsPerFrame = 1.0 / model.mimi.cfg.frameRate
        self.generator = LmGen(model: model.lm, textSampler: textSampler, audioSampler: audioSampler)
        self.textDecoder = TextDecoder(tokenizer: model.tokenizer, noTextToken: model.config.existingTextPaddingId)
        if let condition {
            // Both the tokenizer and the no-text id come from the same bundle, so
            // text ids cannot be paired with weights from another revision.
            guard let tensor = try model.lm.conditionTensor("description", condition) else {
                throw ModelLoadError.invalidConfig("the bundle has no conditioner to set to '\(condition)'")
            }
            self.condition = tensor
        } else {
            self.condition = nil
        }
        self.encoderCache = mimi.makeEncoderCache()
        self.decoderCache = mimi.makeDecoderCache()
        self.measureTiming = measureTiming
        reset()
    }

    /// Everything translated so far in this session.
    public var text: String { textDecoder.text }

    /// Drop all streaming state so the loaded model can be reused.
    public func reset() {
        generator.reset()
        textDecoder.reset()
        mimi.resetState()
        encoderCache.forEach { $0.reset() }
        decoderCache.forEach { $0.reset() }
        pending = []
        finished = false
    }

    /// Run one frame through every fixed-shape path, then reset. Cold
    /// compilation costs far more than the 80 ms frame budget, so it must happen
    /// before a caller starts measuring or streaming.
    public func warmup() {
        // A fresh session is never finished, so this cannot throw.
        _ = try? pushPCM([Float](repeating: 0, count: frameSize))
        reset()
    }

    /// Translate as much of the buffered audio as makes whole frames. Chunks may
    /// be any length; whatever does not fill a frame is kept for the next call.
    /// Throws `SessionError.finished` if the session has been finished.
    public func pushPCM(_ pcm: [Float]) throws -> [StepResult] {
        guard !finished else { throw SessionError.finished }
        pending.append(contentsOf: pcm)
        var results: [StepResult] = []
        while pending.count >= frameSize {
            let frame = Array(pending.prefix(frameSize))
            pending.removeFirst(frameSize)
            results.append(contentsOf: stepFrame(frame))
        }
        return results
    }

    /// Pad the leftover chunk and drain the delayed audio with silence. The tail
    /// also lets the translation catch up, since Hibiki lags the French it is
    /// translating.
    public func finish() -> [StepResult] {
        if finished { return [] }
        var results: [StepResult] = []
        if !pending.isEmpty {
            var tail = [Float](repeating: 0, count: frameSize)
            tail.replaceSubrange(0 ..< pending.count, with: pending)
            pending = []
            results.append(contentsOf: stepFrame(tail))
        }
        let silence = [Float](repeating: 0, count: frameSize)
        for _ in 0 ..< InferenceSession.silenceTailFrames {
            results.append(contentsOf: stepFrame(silence))
        }
        finished = true
        return results
    }

    /// Encode one source frame and run every generation step it yields.
    private func stepFrame(_ frame: [Float]) -> [StepResult] {
        let encodeStarted = measureTiming ? DispatchTime.now() : nil
        let pcm = MLXArray(frame).reshaped([1, 1, frameSize])
        let codes = mimi.encodeStep(pcm, cache: encoderCache)
        var sourceEncodeSeconds: Double?
        if let encodeStarted {
            eval(codes)
            let steps = max(codes.dim(-1), 1)
            sourceEncodeSeconds = InferenceSession.elapsed(encodeStarted) / Double(steps)
        }
        var results: [StepResult] = []
        for index in 0 ..< codes.dim(-1) {
            results.append(generate(sourceTokens: codes[0..., 0..., index], sourceEncodeSeconds: sourceEncodeSeconds))
        }
        return results
    }

    private func generate(sourceTokens: MLXArray, sourceEncodeSeconds: Double?) -> StepResult {
        let textFrameIndex = generator.textFrameIndex
        let generateStarted = sourceEncodeSeconds == nil ? nil : DispatchTime.now()
        let textToken = generator.step(sourceTokens: sourceTokens, condition: condition)
        let audioTokens = generator.lastAudioTokens()
        if let generateStarted {
            if let audioTokens { eval(textToken, audioTokens) } else { eval(textToken) }
        }
        let generationSeconds = generateStarted.map(InferenceSession.elapsed)

        var pcm: [Float]?
        var audioFrameIndex: Int?
        let decodeStarted = sourceEncodeSeconds == nil ? nil : DispatchTime.now()
        if let audioTokens {
            audioFrameIndex = generator.audioFrameIndex
            let decoded = mimi.decodeStep(audioTokens.expandedDimensions(axis: -1), cache: decoderCache)
            if decodeStarted != nil { eval(decoded) }
            pcm = decoded[0, 0].asArray(Float.self)
        }
        let targetDecodeSeconds = decodeStarted.map(InferenceSession.elapsed)

        let textStarted = sourceEncodeSeconds == nil ? nil : DispatchTime.now()
        let token = textToken.item(Int.self)
        let text = textDecoder.push(token)
        let textDecodeSeconds = textStarted.map(InferenceSession.elapsed)

        var timing: StepTiming?
        if let sourceEncodeSeconds, let generationSeconds, let targetDecodeSeconds, let textDecodeSeconds {
            timing = StepTiming(
                sourceEncodeSeconds: sourceEncodeSeconds,
                generationSeconds: generationSeconds,
                targetDecodeSeconds: targetDecodeSeconds,
                textDecodeSeconds: textDecodeSeconds)
        }
        return StepResult(
            textFrameIndex: textFrameIndex, textToken: token, text: text,
            audioFrameIndex: audioFrameIndex, pcm: pcm, secondsPerFrame: secondsPerFrame,
            timing: timing)
    }

    private static func elapsed(_ started: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000_000
    }
}
