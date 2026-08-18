import SwiftUI

/// The one Hibiki Edge screen: download the model, pick a French source, and
/// translate it to English text and speech. Real inference needs the device's
/// Metal GPU; the simulator runs the UI but not the translation itself.
struct ContentView: View {
    @StateObject private var bundle = ArtifactBundleStore()
    @StateObject private var playback = AudioPlayback()
    @StateObject private var translator = Translator()
    @State private var selection: SourceRecording = .defaultSelection

    var body: some View {
        VStack(spacing: 24) {
            header
            modelStatus
            sourceControls
            translationSection
            Spacer()
            attribution
        }
        .padding()
        .onAppear { bundle.refresh() }
        // Selecting another source stops playback and clears the last result, so
        // the two audio streams never overlap and stale text never lingers.
        .onChange(of: selection) { _, _ in
            playback.stop()
            translator.clear()
        }
    }

    private var header: some View {
        Text("Hibiki Edge")
            .font(.largeTitle.bold())
            .padding(.top, 24)
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
            .disabled(translator.isWorking)

            Button {
                if let url = selection.url { playback.play(url: url) }
            } label: {
                Label(playback.isPlaying ? "Playing…" : "Play French",
                      systemImage: playback.isPlaying ? "speaker.wave.2.fill" : "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(selection.url == nil || translator.isWorking)
        }
    }

    @ViewBuilder
    private var translationSection: some View {
        VStack(spacing: 12) {
            switch translator.status {
            case .idle, .done:
                translateButton
            case let .working(message):
                VStack(spacing: 6) {
                    ProgressView()
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            case let .failed(message):
                VStack(spacing: 8) {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .font(.caption)
                        .multilineTextAlignment(.center)
                    translateButton
                }
            }

            if !translator.transcript.isEmpty {
                ScrollView {
                    Text(translator.transcript)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: 160)
            }

            if translator.canReplay {
                Button {
                    playback.replayTarget(translator.targetSamples)
                } label: {
                    Label("Play English", systemImage: "play.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(playback.isPlaying)
            }
        }
    }

    private var translateButton: some View {
        Button {
            if let url = selection.url {
                translator.translate(sourceURL: url, bundleDirectory: bundle.bundleDirectory, playback: playback)
            }
        } label: {
            Label("Translate", systemImage: "waveform")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(bundle.phase != .ready || selection.url == nil || translator.isWorking)
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
