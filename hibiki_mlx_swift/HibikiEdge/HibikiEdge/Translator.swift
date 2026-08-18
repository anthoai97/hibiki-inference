import AVFoundation
import Foundation
import HibikiCore

/// Runs one translation off the main thread and streams its text and audio back
/// to the UI. The loaded model is cached so a second run skips the ~2 GB load.
///
/// The simplest concurrency that keeps the screen responsive: the heavy work
/// runs on a background queue; every published-state and playback update hops
/// back to the main queue, in step order.
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
                let session = try InferenceSession(model: model)

                // Pay the cold Metal-compile cost before streaming, so the first
                // real frame is not slow.
                DispatchQueue.main.async { self.status = .working("Warming up…") }
                session.warmup()

                DispatchQueue.main.async {
                    self.status = .working("Translating…")
                    playback.beginTargetStream()
                }

                // Push in multi-frame chunks so the UI updates as it goes.
                let chunk = model.mimi.cfg.frameSize * 8
                var offset = 0
                while offset < pcm.count {
                    let end = min(offset + chunk, pcm.count)
                    let results = try session.pushPCM(Array(pcm[offset ..< end]))
                    offset = end
                    self.emit(results, playback: playback)
                }
                self.emit(session.finish(), playback: playback)

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

    /// Marshal each step's text and audio to the main queue, in order.
    private func emit(_ results: [StepResult], playback: AudioPlayback) {
        for result in results {
            let fragment = result.text
            let pcm = result.pcm
            DispatchQueue.main.async {
                if let fragment { self.transcript += fragment }
                if let pcm {
                    self.targetSamples.append(contentsOf: pcm)
                    playback.appendTarget(pcm)
                }
            }
        }
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
