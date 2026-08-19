import Foundation
import MLX
import XCTest

enum MLXTestSupport {
    /// `swift test` runs without MLX's Metal shader bundle (metallib), so the
    /// GPU backend can't initialize here. Force the CPU backend for all MLX ops
    /// in tests: parity is a numerical property, not a performance one, and the
    /// real Metal GPU path is verified on the device (ticket #26).
    static func forceCPUDevice() {
        Device.setDefault(device: Device(.cpu))
    }

    /// The repo root, five levels up from any file in `Tests/HibikiCoreTests/`.
    static func repoRoot() -> URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 5 { root.deleteLastPathComponent() }
        return root
    }

    /// A fixture file under `Tests/HibikiCoreTests/Fixtures/`.
    static func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/\(name)")
    }

    /// A bundle directory relative to the repo root (e.g. `artifacts/hibiki-1b-mlx-q8`).
    static func bundleURL(_ subpath: String) -> URL {
        repoRoot().appendingPathComponent(subpath, isDirectory: true)
    }

    /// The `$HIBIKI_ARTIFACTS` override if set, otherwise the given repo bundle.
    static func defaultBundleURL(fallback subpath: String) -> URL {
        if let override = ProcessInfo.processInfo.environment["HIBIKI_ARTIFACTS"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return bundleURL(subpath)
    }

    /// Skip the calling test unless the bundle's `config.json` is present; the
    /// multi-gigabyte weights are gitignored, so weight-backed tests skip rather
    /// than fail when the bundle has not been downloaded.
    static func requireBundle(_ directory: URL) throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: directory.appendingPathComponent("config.json").path),
            "artifact bundle not present at \(directory.path); download it first")
    }
}
