import SwiftUI

/// Placeholder screen for the Hibiki Edge prototype.
///
/// This is the skeleton from issue #18: a single static screen that only has to
/// launch. MLX Swift is a linked dependency of the app target (declared in the
/// Xcode project and pinned in `Package.resolved`); it is first exercised by the
/// native inference tickets, not here, so nothing MLX runs during rendering.
/// Later tickets replace this body with the real one-screen flow (model
/// download, source picker, Play French, Translate, transcript, Play English).
struct ContentView: View {
    var body: some View {
        VStack(spacing: 16) {
            Text("Hibiki Edge")
                .font(.largeTitle.bold())

            Text("On-device French → English speech translation")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Text("Model: Hibiki 1B · CC-BY 4.0 (Kyutai)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
