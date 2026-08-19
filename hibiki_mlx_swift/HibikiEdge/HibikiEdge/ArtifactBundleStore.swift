import Foundation

/// Downloads and retains the one pinned Hibiki **artifact bundle** — the
/// configuration, Hibiki weights, Mimi weights, and SentencePiece tokenizer
/// from a single hard-coded canonical source (Hugging Face, pinned revision).
///
/// The bundle is retained under Application Support so later launches reuse it
/// instead of downloading ~2 GB again. This is the minimal prototype behaviour
/// from ticket #20: basic progress, a plain error with retry, per-file resume
/// (a file already on disk is kept), and a presence check.
///
/// URLSession delivers its callbacks on a background serial queue (not the main
/// thread), progress is published to the UI only when the whole-percent value
/// changes, and the download continuation is guarded by a lock — so a
/// multi-gigabyte download never floods or stalls the main thread.
final class ArtifactBundleStore: NSObject, ObservableObject {
    enum Phase: Equatable {
        case idle
        case downloading(label: String, fraction: Double)
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    // Pinned canonical source: the Q8-quantized bundle (~2.1 GB).
    private static let repo = "anquachdev/hibiki-1b-mlx-q8"
    private static let revision = "417bed8fc89290a5299cc31ca595e467ff8ac84a"
    private static let fileNames = [
        "config.json",
        "hibiki-mlx-dc2cf5a5@80.q8.safetensors",
        "mimi-dbaa9758@125.safetensors",
        "tokenizer_spm_48k_multi6_2.model",
    ]

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        // delegateQueue nil -> a background serial queue; callbacks stay off the main thread.
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    private let lock = NSLock()
    private var isRunning = false
    private var activeContinuation: CheckedContinuation<URL, Error>?
    private var currentLabel = ""
    private var lastReportedPercent = -1

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
    func refresh() { publish(isComplete ? .ready : .idle) }

    /// Start (or retry) downloading any missing files.
    func start() {
        lock.lock()
        if isRunning { lock.unlock(); return }
        isRunning = true
        lock.unlock()
        Task { await run() }
    }

    private func run() async {
        defer {
            lock.lock(); isRunning = false; lock.unlock()
        }
        if isComplete { publish(.ready); return }
        do {
            try FileManager.default.createDirectory(at: bundleDirectory, withIntermediateDirectories: true)
            excludeFromBackup(bundleDirectory)

            let total = Self.fileNames.count
            for (index, name) in Self.fileNames.enumerated() {
                let destination = localURL(for: name)
                if FileManager.default.fileExists(atPath: destination.path) { continue }

                lock.lock()
                currentLabel = "File \(index + 1) of \(total): \(name)"
                lastReportedPercent = -1
                let label = currentLabel
                lock.unlock()
                publish(.downloading(label: label, fraction: 0))

                let temporary = try await downloadOne(name: name)
                do {
                    try FileManager.default.moveItem(at: temporary, to: destination)
                } catch {
                    try? FileManager.default.removeItem(at: temporary)
                    throw error
                }
            }

            // Minimal validation: every expected file is present.
            guard isComplete else {
                publish(.failed("Download finished but some files are missing."))
                return
            }
            publish(.ready)
        } catch {
            publish(.failed(Self.message(for: error)))
        }
    }

    private func downloadOne(name: String) async throws -> URL {
        var request = URLRequest(url: remoteURL(for: name))
        request.setValue("HibikiEdge", forHTTPHeaderField: "User-Agent")
        return try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            activeContinuation = continuation
            lock.unlock()
            session.downloadTask(with: request).resume()
        }
    }

    private func finish(_ result: Result<URL, Error>) {
        lock.lock()
        let continuation = activeContinuation
        activeContinuation = nil
        lock.unlock()
        continuation?.resume(with: result)
    }

    private func excludeFromBackup(_ url: URL) {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutable = url
        try? mutable.setResourceValues(values)
    }

    private func publish(_ newPhase: Phase) {
        DispatchQueue.main.async { self.phase = newPhase }
    }

    private static func message(for error: Error) -> String {
        if let urlError = error as? URLError {
            return "Network error: \(urlError.localizedDescription)"
        }
        return error.localizedDescription
    }
}

extension ArtifactBundleStore: URLSessionDownloadDelegate {
    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let percent = Int(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) * 100)
        lock.lock()
        let changed = percent != lastReportedPercent
        if changed { lastReportedPercent = percent }
        let label = currentLabel
        lock.unlock()
        // Publish at most once per whole percent, so a multi-GB file makes ~100
        // UI updates rather than thousands.
        if changed {
            publish(.downloading(label: label, fraction: Double(percent) / 100))
        }
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        guard let http = downloadTask.response as? HTTPURLResponse else {
            finish(.failure(URLError(.badServerResponse)))
            return
        }
        guard (200...299).contains(http.statusCode) else {
            finish(.failure(NSError(domain: "HibikiEdge", code: http.statusCode,
                                    userInfo: [NSLocalizedDescriptionKey: "Server returned HTTP \(http.statusCode)."])))
            return
        }
        // `location` is removed once this method returns, so move it somewhere we own.
        let stable = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: stable)
            finish(.success(stable))
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // Success is resumed in didFinishDownloadingTo (same serial queue, ordered
        // before this). If the task ended without delivering a file, resume here
        // so the run never hangs; finish is a no-op once already resumed.
        finish(.failure(error ?? URLError(.unknown)))
    }
}
