import Foundation

/// When live target audio may start (or resume) so a short generate spike
/// does not empty the playback ring.
///
/// Initial start waits for `prerollSamples` (~1 s). After an underrun, play
/// stays silent until `rebufferSamples` have arrived again. Once the stream
/// has ended, any remainder plays immediately.
public struct LivePlaybackGate: Equatable {
    public let prerollSamples: Int
    public let rebufferSamples: Int
    public private(set) var playing = false
    private var everPlayed = false

    public init(sampleRate: Double = 24_000, prerollSeconds: Double = 1.0, rebufferSeconds: Double = 0.4) {
        self.prerollSamples = max(1, Int((prerollSeconds * sampleRate).rounded()))
        self.rebufferSamples = max(1, Int((rebufferSeconds * sampleRate).rounded()))
    }

    /// Whether the render callback should pull from the ring this quantum.
    public mutating func shouldConsume(available: Int, streamEnded: Bool) -> Bool {
        if streamEnded {
            playing = available > 0
            return available > 0
        }
        if playing {
            if available == 0 { playing = false }
            return playing
        }
        let need = everPlayed ? rebufferSamples : prerollSamples
        if available >= need {
            playing = true
            everPlayed = true
            return true
        }
        return false
    }
}
