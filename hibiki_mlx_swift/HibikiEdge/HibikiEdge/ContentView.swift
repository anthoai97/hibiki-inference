import HibikiCore
import SwiftUI

/// The one Hibiki Edge screen: download the model, pick a French source, and
/// translate it to English text and speech. Layout and tokens follow the
/// dual-timeline prototype; live mode is omitted because capture is not wired.
/// Real inference needs the device's Metal GPU; the simulator runs the UI
/// but not the translation itself.
struct ContentView: View {
    @StateObject private var bundle = ArtifactBundleStore()
    @StateObject private var playback = AudioPlayback()
    @StateObject private var translator = Translator()
    @State private var selection: SourceRecording = .defaultSelection
    @State private var previewLevels: [Float] = []
    @State private var previewSeconds: TimeInterval = 0
    @State private var scrubStep: Int?
    @State private var showingPicker = false

    var body: some View {
        VStack(spacing: 0) {
            header
            if let caption = Translator.hostCaption {
                hostBanner(caption)
            }
            metrics
            DualTimeline(
                sourceLevels: sourceLevels,
                targetLevels: translator.targetLevels,
                window: window,
                playhead: playhead,
                completeAudioFrames: translator.completeAudioFrames,
                writtenFrames: writtenFrames)
            .padding(.horizontal, 20)
            if let gap = onsetGap {
                gapCallout(gap)
            }
            CaptionRibbon(
                textFrames: translator.textFrames,
                playhead: playhead,
                placeholder: ribbonPlaceholder,
                failure: failureMessage)
            .padding(.horizontal, 20)
            .padding(.top, 12)
            bottomControls
        }
        .background(Palette.surface.ignoresSafeArea())
        .onAppear { bundle.refresh() }
        .task(id: selection.id) { await loadPreview() }
        .onChange(of: selection) { _, _ in
            playback.stop()
            translator.clear()
            scrubStep = nil
        }
        .sheet(isPresented: $showingPicker) {
            SourcePickerSheet(selection: $selection)
                #if os(iOS)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
                #endif
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Translation run")
                    .font(Typeface.eyebrow)
                    .tracking(1.2)
                    .textCase(.uppercase)
                    .foregroundStyle(Palette.inkFaint)
                Button {
                    if !translator.isWorking { showingPicker = true }
                } label: {
                    HStack(spacing: 4) {
                        Text(selection.displayName)
                            .font(Typeface.title)
                            .foregroundStyle(Palette.ink)
                        Text("▾")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(Palette.inkFaint)
                    }
                }
                .buttonStyle(.plain)
                .disabled(translator.isWorking)
            }
            Spacer()
            Button {
                if playback.stream == .source {
                    playback.stop()
                } else if let url = selection.url {
                    playback.play(url: url)
                }
            } label: {
                Text(playback.stream == .source ? "Stop" : "Play French")
                    .font(Typeface.eyebrow)
                    .tracking(0.6)
                    .textCase(.uppercase)
                    .foregroundStyle(translator.isWorking || selection.url == nil ? Palette.pending : Palette.playhead)
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(Palette.hairline, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .disabled(selection.url == nil || translator.isWorking)
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        .padding(.bottom, 10)
    }

    private func hostBanner(_ caption: String) -> some View {
        Text(caption)
            .font(Typeface.note)
            .foregroundStyle(Palette.playhead)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .background(Palette.wash)
            .padding(.bottom, 4)
    }

    private var metrics: some View {
        HStack(spacing: 0) {
            MetricCell(label: "Audio", value: formatSeconds(audioSeconds), unit: "s")
            Rectangle().fill(Palette.hairline).frame(width: 1)
            MetricCell(label: "Wall clock", value: wallClockValue, unit: wallClockUnit)
                .padding(.leading, 12)
            Rectangle().fill(Palette.hairline).frame(width: 1)
            MetricCell(
                label: "Real time",
                value: realtimeValue,
                unit: translator.realtimeFactor > 0 ? "×" : "",
                emphasis: translator.realtimeFactor >= 1)
            .padding(.leading, 12)
        }
        .fixedSize(horizontal: false, vertical: true)
        .padding(.horizontal, 20)
        .padding(.vertical, 9)
        .overlay(alignment: .top) { Palette.hairline.frame(height: 1) }
        .overlay(alignment: .bottom) { Palette.hairline.frame(height: 1) }
        .padding(.bottom, 12)
    }

    private func gapCallout(_ gap: (source: TimeInterval, target: TimeInterval)) -> some View {
        let banked = max(0, gap.target - gap.source)
        return Text(gapText(source: gap.source, target: gap.target, banked: banked))
            .font(Typeface.note)
            .foregroundStyle(Color(light: 0x5C554A, dark: 0xC4BBA8))
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Palette.wash)
            .overlay(alignment: .leading) { Palette.playhead.frame(width: 3) }
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .padding(.horizontal, 20)
            .padding(.top, 10)
    }

    private func gapText(source: TimeInterval, target: TimeInterval, banked: TimeInterval) -> AttributedString {
        var text = AttributedString("First French speech at ")
        text.append(bold("\(formatSeconds(source))s"))
        text.append(AttributedString(", first English at "))
        text.append(bold("\(formatSeconds(target))s"))
        text.append(AttributedString(" — Hibiki banks "))
        text.append(bold("\(formatSeconds(banked))s"))
        text.append(AttributedString(" of context, then never falls behind again."))
        return text
    }

    private func bold(_ string: String) -> AttributedString {
        var piece = AttributedString(string)
        piece.font = Typeface.note.weight(.bold)
        piece.foregroundColor = Palette.playhead
        return piece
    }

    private var bottomControls: some View {
        VStack(spacing: 10) {
            ProgressTrack(
                progress: trackProgress,
                isScrubbingEnabled: canScrub,
                onScrub: { fraction in
                    let total = max(window.count, 1)
                    scrubStep = Int((fraction * Double(total)).rounded())
                })
            HStack {
                Text(stepLabel)
                Spacer()
                Text(timeLabel)
            }
            .font(Typeface.instrument)
            .foregroundStyle(Palette.inkFaint)

            if let action = primaryAction {
                Button(action: action.run) {
                    Text(action.title)
                }
                .buttonStyle(InkButtonStyle(fill: action.fill, foreground: action.foreground))
                .disabled(action.disabled)
                .opacity(action.disabled ? 0.45 : 1)
            }

            if translator.canReplay, case .done = translator.status {
                Button {
                    if let url = selection.url {
                        scrubStep = nil
                        translator.translate(
                            sourceURL: url,
                            bundleDirectory: bundle.bundleDirectory,
                            playback: playback)
                    }
                } label: {
                    Text("Translate again")
                        .font(Typeface.note.weight(.semibold))
                        .foregroundStyle(Palette.inkMuted)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.plain)
                .disabled(bundle.phase != .ready)
            }

            Text("Model: Hibiki 1B · CC-BY 4.0 (Kyutai)")
                .font(.system(size: 10))
                .foregroundStyle(Palette.inkFaint)
        }
        .padding(.horizontal, 20)
        .padding(.top, 12)
        .padding(.bottom, 20)
    }

    // MARK: Derived state

    private var sourceLevels: [Float] {
        translator.sourceLevels.isEmpty ? previewLevels : translator.sourceLevels
    }

    private var audioSeconds: TimeInterval {
        translator.sourceSeconds > 0 ? translator.sourceSeconds : previewSeconds
    }

    private var window: Range<Int> {
        let total = translator.totalGenerationSteps
            ?? max(sourceLevels.count + InferenceSession.silenceTailFrames, 1)
        return 0 ..< total
    }

    private var playhead: Int {
        if let scrubStep { return min(max(scrubStep, 0), window.count) }
        return translator.generationStep
    }

    private var writtenFrames: Int? {
        switch translator.status {
        case .done:
            return max(translator.targetLevels.count, translator.completeAudioFrames)
        default:
            if let total = translator.totalGenerationSteps {
                return max(0, total - 2)
            }
            return nil
        }
    }

    private var canScrub: Bool {
        if case .done = translator.status { return true }
        return false
    }

    private var trackProgress: Double {
        switch bundle.phase {
        case let .downloading(_, fraction) where !translator.isWorking:
            return fraction
        default:
            let total = Double(max(window.count, 1))
            return Double(playhead) / total
        }
    }

    private var stepLabel: String {
        if case let .downloading(label, _) = bundle.phase, !translator.isWorking {
            return label
        }
        return "step \(playhead) / \(window.count)"
    }

    private var timeLabel: String {
        if case let .downloading(_, fraction) = bundle.phase, !translator.isWorking {
            return "\(Int(fraction * 100))%"
        }
        if translator.computeTime > 0 || translator.isWorking {
            return "\(formatSeconds(translator.modelTime))s model · \(formatSeconds(translator.computeTime))s compute"
        }
        return "\(formatSeconds(audioSeconds))s source"
    }

    private var wallClockValue: String {
        if translator.computeTime > 0 { return formatSeconds(translator.computeTime) }
        return "—"
    }

    private var wallClockUnit: String {
        translator.computeTime > 0 ? "s" : ""
    }

    private var realtimeValue: String {
        translator.realtimeFactor > 0 ? formatSeconds(translator.realtimeFactor) : "—"
    }

    private var ribbonPlaceholder: String {
        switch translator.status {
        case .working("Warming up…"), .working("Loading model…"):
            return "Preparing the artifact bundle on this device…"
        case .working:
            return "Listening… Hibiki holds a couple of seconds of context before it starts speaking."
        default:
            return "Speech in, speech out — no French transcript is produced."
        }
    }

    private var failureMessage: String? {
        if case let .failed(message) = translator.status { return message }
        if case let .failed(message) = bundle.phase { return message }
        return nil
    }

    private var onsetGap: (source: TimeInterval, target: TimeInterval)? {
        guard !translator.targetLevels.isEmpty,
              let source = AudioLevel.firstVoicedIndex(sourceLevels),
              let target = AudioLevel.firstVoicedIndex(translator.targetLevels)
        else { return nil }
        return (
            Double(source) * AudioLevel.secondsPerFrame,
            Double(target) * AudioLevel.secondsPerFrame)
    }

    private struct PrimaryAction {
        let title: String
        let fill: Color
        let foreground: Color
        let disabled: Bool
        let run: () -> Void
    }

    private var primaryAction: PrimaryAction? {
        switch translator.status {
        case let .working(message):
            return PrimaryAction(
                title: message, fill: Palette.ink, foreground: Palette.surface,
                disabled: true, run: {})
        case .failed:
            if bundle.phase != .ready {
                return downloadAction
            }
            return translateAction(title: "Retry translation")
        case .done:
            let stopping = playback.stream == .target
            return PrimaryAction(
                title: stopping ? "Stop" : "Play English",
                fill: stopping ? Palette.playhead : Palette.target,
                foreground: Palette.surface,
                disabled: translator.targetSamples.isEmpty,
                run: {
                    if playback.stream == .target {
                        playback.stop()
                    } else {
                        playback.replayTarget(translator.targetSamples)
                    }
                })
        case .idle:
            switch bundle.phase {
            case .idle, .failed:
                return downloadAction
            case let .downloading(_, fraction):
                return PrimaryAction(
                    title: "Downloading \(Int(fraction * 100))%",
                    fill: Palette.ink, foreground: Palette.surface,
                    disabled: true, run: {})
            case .ready:
                return translateAction(title: "Translate")
            }
        }
    }

    private var downloadAction: PrimaryAction {
        let retry: Bool
        if case .failed = bundle.phase { retry = true } else { retry = false }
        return PrimaryAction(
            title: retry ? "Retry download" : "Download model",
            fill: Palette.ink, foreground: Palette.surface,
            disabled: false, run: { bundle.start() })
    }

    private func translateAction(title: String) -> PrimaryAction {
        PrimaryAction(
            title: title, fill: Palette.ink, foreground: Palette.surface,
            disabled: bundle.phase != .ready || selection.url == nil,
            run: {
                if let url = selection.url {
                    scrubStep = nil
                    translator.translate(
                        sourceURL: url,
                        bundleDirectory: bundle.bundleDirectory,
                        playback: playback)
                }
            })
    }

    private func loadPreview() async {
        previewLevels = []
        previewSeconds = 0
        guard let url = selection.url else { return }
        let loaded: ([Float], TimeInterval)? = await Task.detached(priority: .userInitiated) {
            guard let pcm = try? Translator.readPCM(url: url) else { return nil }
            let levels = AudioLevel.levels(from: pcm)
            return (levels, Double(pcm.count) / AudioLevel.sampleRate)
        }.value
        guard let loaded else { return }
        previewLevels = loaded.0
        previewSeconds = loaded.1
    }

    private func formatSeconds(_ value: TimeInterval) -> String {
        String(format: "%.1f", value)
    }
}

#Preview {
    ContentView()
}
