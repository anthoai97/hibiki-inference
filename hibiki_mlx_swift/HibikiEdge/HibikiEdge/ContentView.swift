import SwiftUI

/// The one Hibiki Edge screen.
///
/// So far it has model download (ticket #20) and source-audio selection with
/// Play French (ticket #19). Translate, the English transcript, and Play
/// English are added by later tickets. MLX Swift is a linked dependency but is
/// not exercised here.
struct ContentView: View {
    @StateObject private var bundle = ArtifactBundleStore()
    @StateObject private var playback = AudioPlayback()
    @State private var selection: SourceRecording = .defaultSelection

    var body: some View {
        VStack(spacing: 28) {
            header
            modelStatus
            sourceControls
            Spacer()
            attribution
        }
        .padding()
        .onAppear { bundle.refresh() }
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
        .padding(.top, 32)
    }

    @ViewBuilder
    private var modelStatus: some View {
        switch bundle.phase {
        case .idle:
            Button {
                bundle.start()
            } label: {
                Label("Download model", systemImage: "arrow.down.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)

        case let .downloading(label, fraction):
            VStack(spacing: 6) {
                ProgressView(value: fraction) {
                    Text(label).font(.caption)
                }
                Text("\(Int(fraction * 100))%")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

        case .ready:
            Label("Model ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.subheadline)

        case let .failed(message):
            VStack(spacing: 8) {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                Button("Retry download") { bundle.start() }
                    .buttonStyle(.bordered)
            }
        }
    }

    private var sourceControls: some View {
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
    }

    private var attribution: some View {
        Text("Model: Hibiki 1B · CC-BY 4.0 (Kyutai)")
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }
}

#Preview {
    ContentView()
}
