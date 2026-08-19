import SwiftUI

/// Shared x-mapping for both lanes, the ruler, and the playhead. Both lanes
/// map against generation steps, never against their own array lengths.
enum TimelineGeometry {
    static func x(forFrame frame: Int, in width: CGFloat, window: Range<Int>) -> CGFloat {
        guard window.count > 0 else { return 0 }
        return CGFloat(frame - window.lowerBound) / CGFloat(window.count) * width
    }
}

struct DualTimeline: View {
    let sourceLevels: [Float]
    let targetLevels: [Float]
    let window: Range<Int>
    let playhead: Int
    let completeAudioFrames: Int
    let writtenFrames: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            laneLabel("French in", trailing: "level only — never transcribed")
            LevelLane(
                levels: sourceLevels, window: window, playhead: playhead, fillThrough: playhead,
                fill: Palette.ink, pending: Palette.pending, playheadColor: Palette.playhead)
            .frame(height: 74)
            TimeRuler(window: window, playhead: playhead)
                .frame(height: 26)
            laneLabel("English out", trailing: targetMeta)
            LevelLane(
                levels: targetLevels, window: window, playhead: playhead,
                fillThrough: max(0, playhead - 2),
                fill: Palette.target, pending: Palette.pending, playheadColor: Palette.playhead)
            .frame(height: 74)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Translation timeline")
        .accessibilityValue(accessibilityValue)
    }

    private var targetMeta: String {
        if let writtenFrames {
            return "\(completeAudioFrames) / \(writtenFrames) frames decoded"
        }
        if completeAudioFrames == 0 { return "no frames decoded yet" }
        return "\(completeAudioFrames) frames decoded"
    }

    private var accessibilityValue: String {
        let seconds = String(format: "%.1f", Double(playhead) * AudioLevel.secondsPerFrame)
        return "\(seconds) seconds of model time, \(completeAudioFrames) English frames"
    }

    private func laneLabel(_ title: String, trailing: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .textCase(.uppercase)
                .tracking(0.6)
            Spacer()
            Text(trailing)
        }
        .font(Typeface.lane)
        .foregroundStyle(Palette.inkFaint)
        .padding(.bottom, 4)
    }
}

struct LevelLane: View {
    let levels: [Float]
    let window: Range<Int>
    let playhead: Int
    let fillThrough: Int
    let fill: Color
    let pending: Color
    let playheadColor: Color

    var body: some View {
        Canvas { context, size in
            let span = window.count
            guard span > 0, size.width > 0 else { return }
            var midline = Path()
            midline.move(to: CGPoint(x: 0, y: size.height / 2))
            midline.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            context.stroke(midline, with: .color(Palette.hairline), lineWidth: 1)

            let framesPerColumn = max(1, Int(ceil(Double(span) / Double(size.width))))
            let columns = Int(ceil(Double(span) / Double(framesPerColumn)))
            let columnWidth = size.width / CGFloat(columns)

            var past = Path()
            var future = Path()
            for column in 0 ..< columns {
                let start = window.lowerBound + column * framesPerColumn
                let end = min(start + framesPerColumn, window.upperBound)
                var peak: Float = 0
                for frame in start ..< end {
                    if frame >= 0, frame < levels.count {
                        peak = max(peak, levels[frame])
                    }
                }
                let height = max(1, CGFloat(peak) * (size.height - 6))
                let rect = CGRect(
                    x: CGFloat(column) * columnWidth,
                    y: (size.height - height) / 2,
                    width: max(0.7, columnWidth - 0.45),
                    height: height)
                if end <= fillThrough {
                    past.addRect(rect)
                } else {
                    future.addRect(rect)
                }
            }
            context.fill(past, with: .color(fill))
            context.fill(future, with: .color(pending))

            let x = min(
                max(TimelineGeometry.x(forFrame: playhead, in: size.width, window: window), 0),
                size.width - 1.5)
            var needle = Path()
            needle.addRect(CGRect(x: x, y: 0, width: 1.5, height: size.height))
            context.fill(needle, with: .color(playheadColor))
        }
    }
}

struct TimeRuler: View {
    let window: Range<Int>
    let playhead: Int

    var body: some View {
        Canvas { context, size in
            guard window.count > 0 else { return }
            var baseline = Path()
            baseline.move(to: CGPoint(x: 0, y: size.height / 2))
            baseline.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            context.stroke(baseline, with: .color(Palette.pending), lineWidth: 1)

            let startSecond = Int(floor(Double(window.lowerBound) * AudioLevel.secondsPerFrame / 5) * 5)
            let endSecond = Int(ceil(Double(window.upperBound) * AudioLevel.secondsPerFrame))
            var second = max(0, startSecond)
            while second <= endSecond {
                let frame = Int((Double(second) / AudioLevel.secondsPerFrame).rounded())
                if window.contains(frame) || frame == window.lowerBound {
                    let x = TimelineGeometry.x(forFrame: frame, in: size.width, window: window)
                    if x >= 0, x <= size.width - 2 {
                        var tick = Path()
                        tick.move(to: CGPoint(x: x, y: size.height / 2 - 4))
                        tick.addLine(to: CGPoint(x: x, y: size.height / 2 + 4))
                        context.stroke(tick, with: .color(Palette.pending), lineWidth: 1)
                        let text = Text("\(second)s")
                            .font(.system(size: 9, weight: .semibold, design: .monospaced))
                            .foregroundColor(Palette.inkFaint)
                        context.draw(text, at: CGPoint(x: min(x + 12, size.width - 14), y: size.height / 2 - 8))
                    }
                }
                second += 5
            }

            let x = min(
                max(TimelineGeometry.x(forFrame: playhead, in: size.width, window: window), 0),
                size.width - 1.5)
            var needle = Path()
            needle.addRect(CGRect(x: x, y: 0, width: 1.5, height: size.height))
            context.fill(needle, with: .color(Palette.playhead))
        }
    }
}

struct CaptionRibbon: View {
    let textFrames: [(frame: Int, piece: String)]
    let playhead: Int
    let placeholder: String
    let failure: String?

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Group {
                    if let failure {
                        Text(failure)
                            .font(Typeface.body)
                            .foregroundStyle(Palette.danger)
                    } else if textFrames.isEmpty {
                        Text(placeholder)
                            .font(Typeface.body)
                            .foregroundStyle(Palette.ribbonPending)
                    } else {
                        Text(attributed)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 18)
                Color.clear.frame(height: 1).id("ribbon-tail")
            }
            .onChange(of: textFrames.count) { _, _ in
                proxy.scrollTo("ribbon-tail", anchor: .bottom)
            }
        }
        .mask(
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .black, location: 0.08),
                    .init(color: .black, location: 0.78),
                    .init(color: .clear, location: 1),
                ],
                startPoint: .top,
                endPoint: .bottom))
    }

    private var attributed: AttributedString {
        var result = AttributedString()
        let cutoff = textFrames.lastIndex { $0.frame < playhead } ?? -1
        for (index, item) in textFrames.enumerated() {
            var piece = AttributedString(item.piece)
            if index > cutoff {
                piece.foregroundColor = Palette.ribbonPending
                piece.font = Typeface.body
            } else if index >= cutoff - 3 {
                piece.foregroundColor = Palette.ink
                piece.backgroundColor = Palette.highlight
                piece.font = Typeface.body.weight(.semibold)
            } else {
                piece.foregroundColor = Palette.inkMuted
                piece.font = Typeface.body
            }
            result.append(piece)
        }
        return result
    }
}

struct ProgressTrack: View {
    let progress: Double
    let isScrubbingEnabled: Bool
    let onScrub: (Double) -> Void

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Palette.track)
                Capsule()
                    .fill(Palette.ink)
                    .frame(width: max(0, geo.size.width * CGFloat(clamped)))
                Capsule()
                    .fill(Palette.playhead)
                    .frame(width: 3)
                    .offset(x: max(0, min(geo.size.width - 3, geo.size.width * CGFloat(clamped) - 1.5)))
            }
            .contentShape(Rectangle())
            .gesture(drag(in: geo.size.width), including: isScrubbingEnabled ? .all : .subviews)
        }
        .frame(height: 32)
        .accessibilityLabel(isScrubbingEnabled ? "Scrub timeline" : "Translation progress")
        .accessibilityValue("\(Int((clamped * 100).rounded())) percent")
    }

    private var clamped: Double { min(max(progress, 0), 1) }

    private func drag(in width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard isScrubbingEnabled, width > 0 else { return }
                onScrub(min(max(Double(value.location.x / width), 0), 1))
            }
    }
}

struct MetricCell: View {
    let label: String
    let value: String
    let unit: String
    var emphasis: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(Typeface.metricLabel)
                .textCase(.uppercase)
                .tracking(0.8)
                .foregroundStyle(Palette.inkFaint)
            HStack(alignment: .firstTextBaseline, spacing: 1) {
                Text(value)
                    .font(Typeface.metric)
                    .foregroundStyle(emphasis ? Palette.target : Palette.ink)
                Text(unit)
                    .font(Typeface.metricUnit)
                    .foregroundStyle(Palette.inkFaint)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

struct SourcePickerSheet: View {
    @Binding var selection: SourceRecording
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Source recording")
                .font(Typeface.eyebrow)
                .tracking(1.2)
                .textCase(.uppercase)
                .foregroundStyle(Palette.inkFaint)
                .padding(.horizontal, 20)
                .padding(.top, 18)
            Text("Bundled French source audio. All five long-form recordings are 24 kHz mono.")
                .font(Typeface.note)
                .foregroundStyle(Palette.inkFaint)
                .padding(.horizontal, 20)
                .padding(.top, 3)
                .padding(.bottom, 10)

            HStack {
                Text("File")
                Spacer()
                Text("Secs").frame(width: 46, alignment: .trailing)
                Text("Frames").frame(width: 52, alignment: .trailing)
                Text("Rate").frame(width: 44, alignment: .trailing)
            }
            .font(Typeface.metricLabel.weight(.bold))
            .tracking(0.8)
            .textCase(.uppercase)
            .foregroundStyle(Palette.pending)
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .overlay(alignment: .bottom) { Palette.hairline.frame(height: 1) }

            ForEach(SourceRecording.all) { recording in
                let info = recording.info
                Button {
                    selection = recording
                    dismiss()
                } label: {
                    HStack {
                        Text(recording.displayName)
                            .foregroundStyle(Palette.ink)
                        Spacer()
                        Text(info.map { String(format: "%.1f", $0.seconds) } ?? "—")
                            .frame(width: 46, alignment: .trailing)
                        Text(info.map { "\($0.frames)" } ?? "—")
                            .frame(width: 52, alignment: .trailing)
                        Text(info.map { "\(Int($0.sampleRate / 1000))k" } ?? "—")
                            .frame(width: 44, alignment: .trailing)
                    }
                    .font(.system(size: 13).monospacedDigit())
                    .foregroundStyle(Palette.inkMuted)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 11)
                    .background(recording == selection ? Palette.highlight : Color.clear)
                }
                .buttonStyle(.plain)
                .disabled(recording.url == nil)
                .overlay(alignment: .bottom) { Palette.wash.frame(height: 1) }
            }
            Spacer()
        }
        .background(Palette.surface)
    }
}
