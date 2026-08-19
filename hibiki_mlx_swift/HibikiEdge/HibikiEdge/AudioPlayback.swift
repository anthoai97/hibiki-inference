import AVFoundation
import Foundation
import HibikiCore

/// Plays audio one stream at a time: a bundled source file (French) via
/// `AVAudioPlayer`, or streamed English target PCM via an `AVAudioSourceNode`.
///
/// Target playback matches the Python `PlaybackStream`: decoded frames are
/// written into a ring and the audio unit pulls a continuous 24 kHz stream.
/// Discrete `AVAudioPlayerNode` buffers are not used — they click at every
/// 80 ms seam after the graph resamples each buffer on its own.
final class AudioPlayback: NSObject, ObservableObject {
    @Published private(set) var isPlaying = false

    private var filePlayer: AVAudioPlayer?

    private let engine = AVAudioEngine()
    private let targetFormat: AVAudioFormat
    private let ring = PCMRing(capacity: 24_000 * 8)
    private var sourceNode: AVAudioSourceNode!
    private let stateLock = NSLock()
    private var streaming = false
    private var streamEnded = false
    private var epoch = 0

    override init() {
        targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!
        super.init()
        let node = AVAudioSourceNode(format: targetFormat) { [weak self] _, _, frameCount, abl in
            self?.render(frames: Int(frameCount), abl: abl)
            return noErr
        }
        sourceNode = node
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

    /// Open the continuous target stream. The node outputs silence until the
    /// first decoded samples arrive, the same way Python opens `OutputStream`
    /// before the first `write`.
    func beginTargetStream() {
        stop()
        do {
            try activateSession()
        } catch {
            return
        }
        stateLock.lock()
        ring.reset()
        streaming = true
        streamEnded = false
        stateLock.unlock()
        do {
            if !engine.isRunning { try engine.start() }
            setPlaying(true)
        } catch {
            stateLock.lock()
            streaming = false
            stateLock.unlock()
            setPlaying(false)
        }
    }

    /// Write one block of 24 kHz mono float samples into the ring. Safe to
    /// call from the inference thread; blocks if the listener is more than a
    /// few seconds behind, like Python's bounded queue.
    func appendTarget(_ samples: [Float]) {
        stateLock.lock()
        let live = streaming
        stateLock.unlock()
        guard live, !samples.isEmpty else { return }
        ring.write(samples)
    }

    /// Signal that no more blocks will arrive; playback ends when the ring drains.
    func endTargetStream() {
        stateLock.lock()
        streamEnded = true
        stateLock.unlock()
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            while self.ring.available > 0 {
                self.stateLock.lock()
                let still = self.streaming
                self.stateLock.unlock()
                if !still { return }
                Thread.sleep(forTimeInterval: 0.05)
            }
            self.setPlaying(false)
        }
    }

    /// Replay a completed English translation through the same continuous stream.
    func replayTarget(_ samples: [Float]) {
        beginTargetStream()
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.appendTarget(samples)
            self?.endTargetStream()
        }
    }

    func stop() {
        filePlayer?.stop()
        filePlayer = nil
        ring.abort()
        stateLock.lock()
        epoch += 1
        streaming = false
        streamEnded = false
        stateLock.unlock()
        if engine.isRunning { engine.stop() }
        if Thread.isMainThread {
            isPlaying = false
        } else {
            setPlaying(false)
        }
    }

    private func render(frames: Int, abl: UnsafeMutablePointer<AudioBufferList>) {
        let buffers = UnsafeMutableAudioBufferListPointer(abl)
        guard let data = buffers[0].mData?.assumingMemoryBound(to: Float.self) else { return }

        stateLock.lock()
        let live = streaming
        stateLock.unlock()
        if !live {
            data.update(repeating: 0, count: frames)
            return
        }
        _ = ring.read(into: data, count: frames)
    }

    private func setPlaying(_ value: Bool) {
        stateLock.lock()
        let captured = epoch
        stateLock.unlock()
        DispatchQueue.main.async {
            self.stateLock.lock()
            let current = self.epoch
            self.stateLock.unlock()
            guard captured == current || !value else { return }
            self.isPlaying = value
        }
    }

    private func activateSession() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setPreferredSampleRate(24_000)
        try session.setActive(true)
        #endif
    }
}

extension AudioPlayback: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
    }
}
