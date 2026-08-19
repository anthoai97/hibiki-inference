import AVFoundation
import Foundation
import HibikiCore
import MLX
import MLXRandom

// A macOS command-line runner for the Hibiki S2S pipeline. Unlike the tests
// (which force the CPU backend for deterministic parity), this runs on the
// Mac's default device — the Metal GPU — so a real translation runs at speed
// without a phone. It loads the bundle, streams a French WAV through the
// inference session, prints the English transcript as it arrives, and writes
// the English audio to a WAV.
//
// Usage:
//   swift run hibiki-translate <bundle-dir> <input.wav> [output.wav]

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    fail("usage: hibiki-translate <bundle-dir> <input.wav> [output.wav]")
}
let bundleDirectory = URL(fileURLWithPath: arguments[1])
let inputURL = URL(fileURLWithPath: arguments[2])
let outputURL = arguments.count >= 4 ? URL(fileURLWithPath: arguments[3]) : nil

/// Read a WAV as 24 kHz mono float samples (`processingFormat` is float32).
func readWav(_ url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    guard let buffer = AVAudioPCMBuffer(
        pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
        file.length > 0 else {
        fail("could not read \(url.lastPathComponent)")
    }
    try file.read(into: buffer)
    guard let channel = buffer.floatChannelData else { fail("\(url.lastPathComponent) is not float PCM") }
    return Array(UnsafeBufferPointer(start: channel[0], count: Int(buffer.frameLength)))
}

/// Write mono float samples as a 24 kHz 16-bit PCM WAV.
func writeWav(_ samples: [Float], to url: URL, sampleRate: Int) throws {
    func littleEndian<T: FixedWidthInteger>(_ value: T) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }
    var data = Data()
    let dataBytes = samples.count * 2
    data.append(Data("RIFF".utf8))
    data.append(littleEndian(UInt32(36 + dataBytes)))
    data.append(Data("WAVE".utf8))
    data.append(Data("fmt ".utf8))
    data.append(littleEndian(UInt32(16)))                 // PCM header size
    data.append(littleEndian(UInt16(1)))                  // PCM
    data.append(littleEndian(UInt16(1)))                  // mono
    data.append(littleEndian(UInt32(sampleRate)))
    data.append(littleEndian(UInt32(sampleRate * 2)))     // byte rate
    data.append(littleEndian(UInt16(2)))                  // block align
    data.append(littleEndian(UInt16(16)))                 // bits per sample
    data.append(Data("data".utf8))
    data.append(littleEndian(UInt32(dataBytes)))
    for sample in samples {
        data.append(littleEndian(Int16(max(-1, min(1, sample)) * 32767)))
    }
    try data.write(to: url)
}

do {
    MLXRandom.seed(0)
    print("device: \(Device.defaultDevice())")
    print("loading model from \(bundleDirectory.path) …")
    let started = Date()
    let model = try LoadedModel.load(directory: bundleDirectory)
    print(String(format: "loaded in %.1fs; translating %@ …", -started.timeIntervalSinceNow, inputURL.lastPathComponent))

    let session = try InferenceSession(model: model)
    let pcm = try readWav(inputURL)
    session.warmup()

    let translateStarted = Date()
    var output: [Float] = []
    let chunk = model.mimi.cfg.frameSize * 25 // ~2 s of audio per push
    var offset = 0
    func consume(_ results: [StepResult]) {
        for result in results {
            if let text = result.text {
                FileHandle.standardOutput.write(Data(text.utf8))
            }
            if let block = result.pcm { output.append(contentsOf: block) }
        }
    }
    while offset < pcm.count {
        let end = min(offset + chunk, pcm.count)
        consume(try session.pushPCM(Array(pcm[offset ..< end])))
        offset = end
    }
    consume(session.finish())

    let elapsed = -translateStarted.timeIntervalSinceNow
    let audioSeconds = Double(pcm.count) / model.mimi.cfg.sampleRate
    print("\n\n--- transcript ---\n\(session.text)")
    print(String(format: "\ntranslated %.1fs of audio in %.1fs (%.2fx real time)",
                 audioSeconds, elapsed, audioSeconds / max(elapsed, 0.001)))

    if let outputURL {
        try writeWav(output, to: outputURL, sampleRate: 24_000)
        print("wrote \(output.count) samples of English audio to \(outputURL.path)")
    }
} catch {
    fail("error: \((error as? LocalizedError)?.errorDescription ?? error.localizedDescription)")
}
