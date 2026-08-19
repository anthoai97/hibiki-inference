import SwiftUI
#if canImport(AppKit)
import AppKit
#endif
#if canImport(UIKit)
import UIKit
#endif

/// Visual tokens from the winning dual-timeline prototype (`prototype-phone-ui.html`
/// variant B, recorded in `docs/ios-dual-timeline-ui.md`). Light values are the
/// source of truth; dark values keep the same roles at readable contrast.
enum Palette {
    static let surface = Color(light: 0xF7F6F3, dark: 0x121214)
    static let ink = Color(light: 0x14141A, dark: 0xF7F6F3)
    static let inkMuted = Color(light: 0x6B6558, dark: 0xA8A193)
    static let inkFaint = Color(light: 0x8A8578, dark: 0x7A7568)
    static let hairline = Color(light: 0xE2DED4, dark: 0x2A2A31)
    static let pending = Color(light: 0xDDD8CB, dark: 0x3A3832)
    static let wash = Color(light: 0xEFEADE, dark: 0x1C1C22)
    static let track = Color(light: 0xEAE5D9, dark: 0x1C1C22)
    static let target = Color(light: 0x2F7D68, dark: 0x4AA88A)
    static let playhead = Color(light: 0xC8632F, dark: 0xE07A42)
    static let highlight = Color(light: 0xF5E2B8, dark: 0x4A3D22)
    static let ribbonPending = Color(light: 0xD6D0C2, dark: 0x5A564C)
    static let danger = Color(light: 0xB42318, dark: 0xF97066)
}

enum Typeface {
    static let eyebrow = Font.system(size: 11.5, weight: .bold)
    static let title = Font.system(size: 16.5, weight: .semibold)
    static let metric = Font.system(size: 19, weight: .semibold).monospacedDigit()
    static let metricUnit = Font.system(size: 11, weight: .semibold)
    static let metricLabel = Font.system(size: 9.5, weight: .regular)
    static let lane = Font.system(size: 10.5, weight: .regular)
    static let instrument = Font.system(size: 11, weight: .semibold, design: .monospaced)
    static let body = Font.system(size: 16)
    static let note = Font.system(size: 11.5)
    static let button = Font.system(size: 15, weight: .semibold)
}

extension Color {
    init(hex: UInt, alpha: Double = 1) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: alpha)
    }

    init(light: UInt, dark: UInt) {
        #if os(macOS)
        self.init(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return NSColor(Color(hex: isDark ? dark : light))
        })
        #else
        self.init(uiColor: UIColor { traits in
            UIColor(Color(hex: traits.userInterfaceStyle == .dark ? dark : light))
        })
        #endif
    }
}

struct InkButtonStyle: ButtonStyle {
    var fill: Color = Palette.ink
    var foreground: Color = Palette.surface

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Typeface.button)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .background(fill.opacity(configuration.isPressed ? 0.82 : 1))
            .foregroundStyle(foreground)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
