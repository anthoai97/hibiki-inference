import AVFoundation
import Foundation

/// A bundled French source-audio recording the demonstrator can play and
/// translate. The five long-form recordings ship inside the app under the
/// `SourceAudio` folder and are already 24 kHz mono.
struct SourceRecording: Identifiable, Hashable {
    /// Stable id, also the base file name without extension (e.g. "2").
    let id: String

    var displayName: String { "\(id).wav" }

    /// URL of the bundled WAV. Resources may be copied flat into the bundle
    /// root or kept under `SourceAudio/`, so try both.
    var url: URL? {
        Bundle.main.url(forResource: id, withExtension: "wav", subdirectory: "SourceAudio")
            ?? Bundle.main.url(forResource: id, withExtension: "wav")
    }

    /// Duration and frame count from the file itself. Nil if the resource is missing.
    var info: SourceInfo? {
        guard let url else { return nil }
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let rate = file.fileFormat.sampleRate
        guard rate > 0 else { return nil }
        let seconds = Double(file.length) / rate
        let frames = Int(ceil(seconds / AudioLevel.secondsPerFrame))
        return SourceInfo(seconds: seconds, frames: frames, sampleRate: rate)
    }

    /// The five bundled long-form recordings, in order.
    static let all: [SourceRecording] = (1...5).map { SourceRecording(id: "\($0)") }

    /// The recording selected on first launch — the known `2.wav` example.
    static let defaultSelection = all.first { $0.id == "2" } ?? all[0]
}

struct SourceInfo {
    let seconds: TimeInterval
    let frames: Int
    let sampleRate: Double
}
