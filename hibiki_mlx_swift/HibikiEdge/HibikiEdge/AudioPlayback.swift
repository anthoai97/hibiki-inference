import AVFoundation
import Foundation

/// Plays audio one stream at a time: a bundled source file (French) via
/// `AVAudioPlayer`, or streamed English target PCM blocks via an
/// `AVAudioEngine`. Everything flows through one instance and one `stop()`, so
/// source and target audio can never play at the same time.
final class AudioPlayback: NSObject, ObservableObject {
    @Published private(set) var isPlaying = false

    private var filePlayer: AVAudioPlayer?

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let targetFormat: AVAudioFormat
    private var streaming = false
    private var streamEnded = false
    private var pendingBuffers = 0

    override init() {
        // The codec emits 24 kHz mono float samples.
        targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!
        super.init()
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: targetFormat)
    }

    // MARK: Source file playback

    /// Play the file at `url`, stopping any current playback first.
    /// Returns false if the file could not be opened.
    @discardableResult
    func play(url: URL) -> Bool {
        stop()
        do {
            try activateSession()
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            filePlayer = player
            player.play()
            isPlaying = true
            return true
        } catch {
            filePlayer = nil
            isPlaying = false
            return false
        }
    }

    // MARK: Streamed target playback

    /// Begin a fresh streamed-target playback, stopping anything else first.
    func beginTargetStream() {
        stop()
        do {
            try activateSession()
            if !engine.isRunning { try engine.start() }
            node.play()
            streaming = true
            streamEnded = false
            pendingBuffers = 0
            isPlaying = true
        } catch {
            streaming = false
            isPlaying = false
        }
    }

    /// Schedule one block of 24 kHz mono float samples for playback.
    func appendTarget(_ samples: [Float]) {
        guard streaming, let buffer = makeBuffer(samples) else { return }
        pendingBuffers += 1
        node.scheduleBuffer(buffer) { [weak self] in
            DispatchQueue.main.async { self?.bufferFinished() }
        }
    }

    /// Signal that no more blocks will arrive; playback ends when the queue drains.
    func endTargetStream() {
        streamEnded = true
        if pendingBuffers == 0 { isPlaying = false }
    }

    /// Replay a completed English translation from its accumulated samples.
    func replayTarget(_ samples: [Float]) {
        beginTargetStream()
        appendTarget(samples)
        endTargetStream()
    }

    func stop() {
        filePlayer?.stop()
        filePlayer = nil
        if streaming || node.isPlaying { node.stop() }
        streaming = false
        streamEnded = false
        pendingBuffers = 0
        isPlaying = false
    }

    private func bufferFinished() {
        pendingBuffers = max(0, pendingBuffers - 1)
        if streamEnded, pendingBuffers == 0 { isPlaying = false }
    }

    private func makeBuffer(_ samples: [Float]) -> AVAudioPCMBuffer? {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            buffer.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
        }
        return buffer
    }

    private func activateSession() throws {
        try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try AVAudioSession.sharedInstance().setActive(true)
    }
}

extension AudioPlayback: AVAudioPlayerDelegate {
    // AVAudioPlayer calls its delegate on the thread that started playback (the
    // main thread here), so updating the published flag directly is safe.
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
    }
}
