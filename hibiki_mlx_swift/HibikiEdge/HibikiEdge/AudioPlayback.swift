import AVFoundation
import Foundation

/// Plays one audio file at a time through native iOS playback.
///
/// All playback in the app flows through a single instance because source
/// audio and target audio must never play simultaneously. Starting new
/// playback stops whatever is currently playing.
final class AudioPlayback: NSObject, ObservableObject {
    @Published private(set) var isPlaying = false

    private var player: AVAudioPlayer?

    /// Play the file at `url`, stopping any current playback first.
    /// Returns false if the file could not be opened.
    @discardableResult
    func play(url: URL) -> Bool {
        stop()
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            let newPlayer = try AVAudioPlayer(contentsOf: url)
            newPlayer.delegate = self
            player = newPlayer
            newPlayer.play()
            isPlaying = true
            return true
        } catch {
            player = nil
            isPlaying = false
            return false
        }
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
    }
}

extension AudioPlayback: AVAudioPlayerDelegate {
    // AVAudioPlayer calls its delegate on the thread that started playback
    // (the main thread here), so updating the published flag directly is safe.
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
    }
}
