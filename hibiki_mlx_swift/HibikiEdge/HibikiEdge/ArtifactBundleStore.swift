import Foundation

/// Downloads and retains the one pinned Hibiki **artifact bundle** — the
/// configuration, Hibiki weights, Mimi weights, and SentencePiece tokenizer
/// from a single hard-coded canonical source (Hugging Face, pinned revision).
///
/// The bundle is retained under Application Support so later launches reuse it
/// instead of downloading ~4 GB again. This is the minimal prototype behaviour
/// from ticket #20: basic progress, a plain error with retry, per-file resume
/// (a file already on disk is kept), and a presence check — no background
/// download, no partial-file resume, no automatic updates.
///
/// The type is `@MainActor` and uses a main delegate queue, so its published
/// state and the download continuation are only ever touched on the main
/// thread — no cross-thread races.
@MainActor
final class ArtifactBundleStore: NSObject, ObservableObject {
    enum Phase: Equatable {
        case idle
        case downloading(label: String, fraction: Double)
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    // Pinned canonical source.
    private static let repo = "kyutai/hibiki-1b-mlx-bf16"
    private static let revision = "b3d6291f3dcf7954e1a502e4d66f32e3556f17ae"
    private static let fileNames = [
        "config.json",
        "hibiki-mlx-dc2cf5a5@80.safetensors",
        "mimi-dbaa9758@125.safetensors",
        "tokenizer_spm_48k_multi6_2.model",
    ]

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        return URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }()

    private var isRunning = false
    private var activeContinuation: CheckedContinuation<URL, Error>?
    private var reportProgress: (Double) -> Void = { _ in }

    /// Directory the bundle is retained in, pinned to the revision.
    var bundleDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base
            .appendingPathComponent("ArtifactBundle", isDirectory: true)
            .appendingPathComponent(Self.revision, isDirectory: true)
    }

    private func localURL(for name: String) -> URL { bundleDirectory.appendingPathComponent(name) }

    private func remoteURL(for name: String) -> URL {
        // '@' is left unescaped by `.urlPathAllowed`, but the server expects %40.
        let encoded = (name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name)
            .replacingOccurrences(of: "@", with: "%40")
        return URL(string: "https://huggingface.co/\(Self.repo)/resolve/\(Self.revision)/\(encoded)")!
    }

    /// Every expected file is present locally.
    var isComplete: Bool {
        Self.fileNames.allSatisfy { FileManager.default.fileExists(atPath: localURL(for: $0).path) }
    }

    /// Reflect on-disk state without downloading (call when the screen appears).
    func refresh() { phase = isComplete ? .ready : .idle }

    /// Start (or retry) downloading any missing files.
    func start() {
        guard !isRunning else { return }
        isRunning = true
        Task { await run() }
    }

    private func run() async {
        defer { isRunning = false }
        if isComplete { phase = .ready; return }
        do {
            try FileManager.default.createDirectory(at: bundleDirectory, withIntermediateDirectories: true)
            excludeFromBackup(bundleDirectory)

            let total = Self.fileNames.count
            for (index, name) in Self.fileNames.enumerated() {
                let destination = localURL(for: name)
                if FileManager.default.fileExists(atPath: destination.path) { continue }

                let label = "File \(index + 1) of \(total): \(name)"
                phase = .downloading(label: label, fraction: 0)
                let temporary = try await downloadOne(name: name) { fraction in
                    self.phase = .downloading(label: label, fraction: fraction)
                }
                do {
                    try FileManager.default.moveItem(at: temporary, to: destination)
                } catch {
                    try? FileManager.default.removeItem(at: temporary)
                    throw error
                }
            }

            // Minimal validation: every expected file is present.
            guard isComplete else {
                phase = .failed("Download finished but some files are missing.")
                return
            }
            phase = .ready
        } catch {
            phase = .failed(Self.message(for: error))
        }
    }

    private func downloadOne(name: String, progress: @escaping (Double) -> Void) async throws -> URL {
        reportProgress = progress
        var request = URLRequest(url: remoteURL(for: name))
        request.setValue("HibikiEdge", forHTTPHeaderField: "User-Agent")
        return try await withCheckedThrowingContinuation { continuation in
            self.activeContinuation = continuation
            self.session.downloadTask(with: request).resume()
        }
    }

    private func excludeFromBackup(_ url: URL) {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutable = url
        try? mutable.setResourceValues(values)
    }

    private func finish(_ result: Result<URL, Error>) {
        guard let continuation = activeContinuation else { return }
        activeContinuation = nil
        continuation.resume(with: result)
    }

    private static func message(for error: Error) -> String {
        if let urlError = error as? URLError {
            return "Network error: \(urlError.localizedDescription)"
        }
        return error.localizedDescription
    }
}

extension ArtifactBundleStore: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64,
                                totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        MainActor.assumeIsolated { reportProgress(fraction) }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        // `location` is removed once this method returns, so move it somewhere we
        // own before hopping actors.
        let response = downloadTask.response as? HTTPURLResponse
        let moved: Result<URL, Error>
        if let response, !(200...299).contains(response.statusCode) {
            moved = .failure(NSError(domain: "HibikiEdge", code: response.statusCode,
                                     userInfo: [NSLocalizedDescriptionKey: "Server returned HTTP \(response.statusCode)."]))
        } else {
            let stable = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            do {
                try FileManager.default.moveItem(at: location, to: stable)
                moved = .success(stable)
            } catch {
                moved = .failure(error)
            }
        }
        MainActor.assumeIsolated { finish(moved) }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // Success is resumed in didFinishDownloadingTo. If the task ended without
        // delivering a file, resume here too so the run never hangs and Retry works.
        // `finish` is a no-op once the continuation has already been resumed.
        let result: Result<URL, Error> = .failure(error ?? URLError(.unknown))
        MainActor.assumeIsolated { finish(result) }
    }
}
