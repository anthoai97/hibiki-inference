import Foundation

/// A bundled French source-audio recording the demonstrator can play and,
/// in later tickets, translate. The five long-form recordings ship inside the
/// app under the `SourceAudio` folder and are already 24 kHz mono.
struct SourceRecording: Identifiable, Hashable {
    /// Stable id, also the base file name without extension (e.g. "2").
    let id: String

    var displayName: String { id }

    /// URL of the bundled WAV. Resources may be copied flat into the bundle
    /// root or kept under `SourceAudio/`, so try both.
    var url: URL? {
        Bundle.main.url(forResource: id, withExtension: "wav", subdirectory: "SourceAudio")
            ?? Bundle.main.url(forResource: id, withExtension: "wav")
    }

    /// The five bundled long-form recordings, in order.
    static let all: [SourceRecording] = (1...5).map { SourceRecording(id: "\($0)") }

    /// The recording selected on first launch — the known `2.wav` example.
    static let defaultSelection = all.first { $0.id == "2" } ?? all[0]
}
