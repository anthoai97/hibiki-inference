import AVFoundation
import Foundation
import HibikiCore

/// Runs one translation off the main thread and streams its text and audio back
/// to the UI. The loaded model is cached so a second run skips the ~2 GB load.
///
/// The simplest concurrency that keeps the screen responsive: the heavy work
/// runs on a background queue, target PCM is handed to the player there, and
/// published text hops back to the main queue once per push.
final class Translator: ObservableObject {
    enum Status: Equatable {
        case idle
        case working(String)
        case done
        case failed(String)
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var transcript = ""
    /// The accumulated English target samples, kept so the result can be replayed.
    @Published private(set) var targetSamples: [Float] = []

    private var model: LoadedModel?

    var isWorking: Bool {
        if case .working = status { return true }
        return false
    }

    var canReplay: Bool {
        if case .done = status { return !targetSamples.isEmpty }
        return false
    }

    /// Clear the previous result so a new run — or a new source selection —
    /// starts fresh. Ignored mid-run.
    func clear() {
        guard !isWorking else { return }
        status = .idle
        transcript = ""
        targetSamples = []
    }

    /// Translate `sourceURL` with the downloaded bundle, streaming English text
    /// and audio to the UI and to `playback`. Returns immediately; the work runs
    /// on a background queue.
    func translate(sourceURL: URL, bundleDirectory: URL, playback: AudioPlayback) {
        guard !isWorking else { return }
        // MLX inference runs on the Metal GPU, which the Simulator does not
        // provide — attempting it there aborts the process rather than throwing.
        // Fail with a plain message instead; real translation runs on device.
        if Translator.isSimulator {
            status = .failed("Translation needs a physical device — the Simulator can't run on-device GPU inference.")
            return
        }
        transcript = ""
        targetSamples = []
        status = .working("Loading model…")
        playback.stop()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let model = try self.loadModel(bundleDirectory)
                let pcm = try Translator.readPCM(url: sourceURL)
                let session = try InferenceSession(model: model, measureTiming: true)

                // Pay the cold Metal-compile cost before streaming, so the first
                // real frame is not slow.
                DispatchQueue.main.async { self.status = .working("Warming up…") }
                session.warmup()

                let started = DispatchGroup()
                started.enter()
                DispatchQueue.main.async {
                    self.status = .working("Translating…")
                    playback.beginTargetStream()
                    started.leave()
                }
                started.wait()
                print("[hibiki] \(Translator.runtimeHost())")

                // One source frame at a time, matching the Python runner:
                // each completed target frame is written into the playback
                // ring as soon as it exists.
                let chunk = model.mimi.cfg.frameSize
                var offset = 0
                var timed: [StepTiming] = []
                while offset < pcm.count {
                    let end = min(offset + chunk, pcm.count)
                    let results = try session.pushPCM(Array(pcm[offset ..< end]))
                    offset = end
                    self.emit(results, playback: playback, timed: &timed)
                }
                self.emit(session.finish(), playback: playback, timed: &timed)
                Translator.logTotals(steps: timed, sourceSeconds: Double(pcm.count) / model.mimi.cfg.sampleRate)

                DispatchQueue.main.async {
                    playback.endTargetStream()
                    self.status = .done
                }
            } catch {
                DispatchQueue.main.async {
                    playback.stop()
                    self.status = .failed(Translator.message(for: error))
                }
            }
        }
    }

    private func loadModel(_ bundleDirectory: URL) throws -> LoadedModel {
        // Reached only on the background queue, and serialized by the isWorking
        // guard in translate(), so the cache needs no extra synchronization.
        if let model { return model }
        let loaded = try LoadedModel.load(directory: bundleDirectory)
        model = loaded
        return loaded
    }

    /// Write PCM into the playback ring from this thread (the Python
    /// `PlaybackStream.play` call) and hop text to the main queue.
    private func emit(_ results: [StepResult], playback: AudioPlayback, timed: inout [StepTiming]) {
        var text = ""
        var pcm: [Float] = []
        for result in results {
            if let fragment = result.text { text += fragment }
            if let samples = result.pcm { pcm.append(contentsOf: samples) }
            if let timing = result.timing {
                timed.append(timing)
                // Printing every 80 ms frame from the inference thread overloads
                // the console and Core Audio (HAL "skipping cycle due to overload"),
                // which then stalls the next Metal encode for seconds. Keep a
                // heartbeat and log only real spikes.
                let step = result.textFrameIndex
                if step % 25 == 0 || timing.totalSeconds >= 0.16 {
                    print("[hibiki] \(timing.formatted(textFrameIndex: step, audioFrameIndex: result.audioFrameIndex))")
                }
            }
        }
        if !pcm.isEmpty { playback.appendTarget(pcm) }
        if text.isEmpty && pcm.isEmpty { return }
        DispatchQueue.main.async {
            if !text.isEmpty { self.transcript += text }
            if !pcm.isEmpty { self.targetSamples.append(contentsOf: pcm) }
        }
    }

    /// Same summary line the Python runner prints after `--metrics`.
    private static func logTotals(steps: [StepTiming], sourceSeconds: Double) {
        let measured = steps.reduce(0.0) { $0 + $1.totalSeconds }
        let encode = steps.reduce(0.0) { $0 + $1.sourceEncodeSeconds }
        let generate = steps.reduce(0.0) { $0 + $1.generationSeconds }
        let decode = steps.reduce(0.0) { $0 + $1.targetDecodeSeconds }
        let text = steps.reduce(0.0) { $0 + $1.textDecodeSeconds }
        let mean = steps.isEmpty ? 0 : measured / Double(steps.count)
        let realtime = measured > 0 ? sourceSeconds / measured : 0
        let sorted = steps.map(\.totalSeconds).sorted()
        let p50 = percentile(sorted, 0.50)
        let p95 = percentile(sorted, 0.95)
        let maxStep = sorted.last ?? 0
        print(
            "[hibiki] metrics totals: \(Translator.runtimeHost()) steps=\(steps.count) "
                + "measured_steps=\(StepTiming.milliseconds(measured)) "
                + "phases: encode=\(StepTiming.milliseconds(encode)) "
                + "generate=\(StepTiming.milliseconds(generate)) "
                + "decode=\(StepTiming.milliseconds(decode)) "
                + "text=\(StepTiming.milliseconds(text)) "
                + "mean=\(StepTiming.milliseconds(mean)) "
                + "p50=\(StepTiming.milliseconds(p50)) "
                + "p95=\(StepTiming.milliseconds(p95)) "
                + "max=\(StepTiming.milliseconds(maxStep)) "
                + String(format: "realtime=%.2fx (budget=80.0ms)", realtime))
    }

    private static func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
        return sorted[index]
    }

    /// Short label for the screen: this iOS app is not on an iPhone when
    /// Xcode's destination is “My Mac (Designed for iPhone)”.
    static var hostCaption: String? {
        #if os(macOS)
        #if DEBUG
        return "Native macOS · Debug — pick the HibikiEdge-macOS scheme in Release for the real-time number"
        #else
        return nil
        #endif
        #elseif targetEnvironment(simulator)
        return "Simulator — translation is disabled"
        #else
        if ProcessInfo.processInfo.isiOSAppOnMac {
            #if DEBUG
            return "Running on this Mac (Designed for iPhone) · Debug — use the native macOS target instead"
            #else
            return "Running on this Mac (Designed for iPhone) — use the native macOS target instead"
            #endif
        }
        #if DEBUG
        return "Debug build on device — Release is the real-time number"
        #else
        return nil
        #endif
        #endif
    }

    /// Where this process is actually running. The iOS app can be an iPhone,
    /// the Simulator, or “My Mac (Designed for iPhone)” — those are not the
    /// same Metal GPU, and Debug is much slower than Release.
    private static func runtimeHost() -> String {
        var info = utsname()
        uname(&info)
        let machine = withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
                String(cString: $0)
            }
        }
        #if DEBUG
        let build = "Debug"
        #else
        let build = "Release"
        #endif
        #if os(macOS)
        let host = "macos"
        #elseif targetEnvironment(simulator)
        let host = "simulator"
        #elseif targetEnvironment(macCatalyst)
        let host = "mac-catalyst"
        #else
        let host = ProcessInfo.processInfo.isiOSAppOnMac ? "mac-designed-for-iphone" : "iphone"
        #endif
        return "host=\(host) machine=\(machine) build=\(build)"
    }

    /// Read a WAV as 24 kHz mono float samples. `processingFormat` is float32, and
    /// the bundled recordings are already 24 kHz mono.
    static func readPCM(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw NSError(domain: "HibikiEdge", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not read the source audio."])
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData else {
            throw NSError(domain: "HibikiEdge", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "The source audio is not float PCM."])
        }
        return Array(UnsafeBufferPointer(start: channel[0], count: Int(buffer.frameLength)))
    }

    private static func message(for error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }

    private static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        true
        #else
        false
        #endif
    }
}
