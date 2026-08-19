import Foundation

/// Per-frame loudness for the dual timeline. Fixed dBFS mapping — never a
/// running peak — so a live or unfinished run cannot rescale itself.
enum AudioLevel {
    static let frameSize = 1920
    static let sampleRate = 24_000.0
    static let secondsPerFrame = 0.08

    static func level(rms: Float) -> Float {
        guard rms > 0 else { return 0 }
        let dbfs = 20 * log10(rms)
        return min(max((dbfs + 50) / 50, 0), 1)
    }

    static func level(from samples: ArraySlice<Float>) -> Float {
        level(rms: rms(samples))
    }

    static func level(from samples: [Float]) -> Float {
        level(from: samples[samples.startIndex ..< samples.endIndex])
    }

    static func levels(from pcm: [Float], frameSize: Int = frameSize) -> [Float] {
        guard frameSize > 0, !pcm.isEmpty else { return [] }
        var out: [Float] = []
        out.reserveCapacity((pcm.count + frameSize - 1) / frameSize)
        var offset = 0
        while offset < pcm.count {
            let end = min(offset + frameSize, pcm.count)
            out.append(level(from: pcm[offset ..< end]))
            offset = end
        }
        return out
    }

    static func firstVoicedIndex(_ levels: [Float], threshold: Float = 0.08) -> Int? {
        levels.firstIndex { $0 >= threshold }
    }

    private static func rms(_ samples: ArraySlice<Float>) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for sample in samples { sum += sample * sample }
        return sqrt(sum / Float(samples.count))
    }
}
