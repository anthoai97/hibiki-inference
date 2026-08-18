import SwiftUI

/// The one Hibiki Edge screen.
///
/// Ticket #19 adds source-audio selection and Play French: pick one of the
/// five bundled French recordings (default `2.wav`) and play it through native
/// iOS playback. Model download, Translate, transcript, and Play English are
/// added by later tickets. MLX Swift is a linked dependency but is not
/// exercised here.
struct ContentView: View {
    @StateObject private var playback = AudioPlayback()
    @State private var selection: SourceRecording = .defaultSelection

    var body: some View {
        VStack(spacing: 28) {
            header

            VStack(spacing: 16) {
                Picker("Source recording", selection: $selection) {
                    ForEach(SourceRecording.all) { recording in
                        Text(recording.displayName).tag(recording)
                    }
                }
                .pickerStyle(.segmented)

                Button {
                    if let url = selection.url { playback.play(url: url) }
                } label: {
                    Label(playback.isPlaying ? "Playing…" : "Play French",
                          systemImage: playback.isPlaying ? "speaker.wave.2.fill" : "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(selection.url == nil)
            }

            Spacer()

            Text("Model: Hibiki 1B · CC-BY 4.0 (Kyutai)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding()
        // Selecting another source stops the current playback so the two never overlap.
        .onChange(of: selection) { _, _ in playback.stop() }
    }

    private var header: some View {
        VStack(spacing: 8) {
            Text("Hibiki Edge")
                .font(.largeTitle.bold())
            Text("On-device French → English speech translation")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.top, 40)
    }
}

#Preview {
    ContentView()
}
