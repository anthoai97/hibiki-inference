import Foundation

/// The header index of a `.safetensors` file: a map from tensor name to its
/// dtype and shape, read without loading the (multi-gigabyte) tensor data.
///
/// The format is an 8-byte little-endian header length, then that many bytes of
/// JSON mapping each tensor name to `{dtype, shape, data_offsets}`, plus an
/// optional `__metadata__` entry.
public struct SafetensorsIndex {
    public struct Entry {
        public let dtype: String
        public let shape: [Int]
    }

    public let entries: [String: Entry]

    public func shape(of name: String) -> [Int]? { entries[name]?.shape }
    public func contains(_ name: String) -> Bool { entries[name] != nil }
    public var count: Int { entries.count }

    public init(fileURL: URL) throws {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: fileURL)
        } catch {
            throw ModelLoadError.missingFile("\(fileURL.lastPathComponent) could not be opened: \(error.localizedDescription).")
        }
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw ModelLoadError.shapeMismatch("\(fileURL.lastPathComponent) is too short to be a safetensors file.")
        }
        let headerLength = lengthData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        guard headerLength > 0, headerLength < 100_000_000 else {
            throw ModelLoadError.shapeMismatch("\(fileURL.lastPathComponent) has an implausible safetensors header length.")
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)), headerData.count == Int(headerLength) else {
            throw ModelLoadError.shapeMismatch("\(fileURL.lastPathComponent) safetensors header is truncated.")
        }

        guard let raw = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw ModelLoadError.shapeMismatch("\(fileURL.lastPathComponent) safetensors header is not a JSON object.")
        }

        var entries: [String: Entry] = [:]
        for (name, value) in raw where name != "__metadata__" {
            guard let object = value as? [String: Any],
                  let dtype = object["dtype"] as? String,
                  let shape = object["shape"] as? [Int] else {
                continue
            }
            entries[name] = Entry(dtype: dtype, shape: shape)
        }
        self.entries = entries
    }
}
