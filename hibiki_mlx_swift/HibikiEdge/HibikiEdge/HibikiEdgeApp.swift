import SwiftUI

/// Entry point for the Hibiki Edge prototype.
///
/// Skeleton from issue #18: one window, one placeholder screen. The real
/// single-screen flow (model download, source picker, Play French, Translate,
/// transcript, Play English) is built by the later tickets.
@main
struct HibikiEdgeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
