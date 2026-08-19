import Foundation

/// A bounded mono PCM ring that a render callback can pull from.
///
/// The audio thread must never wait: `read` takes a short unfair lock, copies,
/// and returns zeros on underrun. The producer may sleep if the ring is full.
/// A shared `NSCondition` is not used — waiting on it from the render callback
/// overloads Core Audio (`HALC_ProxyIOContext … skipping cycle due to overload`)
/// and then stalls the next Metal encode for hundreds of milliseconds.
public final class PCMRing: @unchecked Sendable {
    public let capacity: Int

    private let samples: UnsafeMutablePointer<Float>
    private var readPos = 0
    private var writePos = 0
    private var count = 0
    private var aborted = false
    private var lock = os_unfair_lock()

    public init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.samples = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        self.samples.initialize(repeating: 0, count: capacity)
    }

    deinit {
        samples.deinitialize(count: capacity)
        samples.deallocate()
    }

    /// Samples currently waiting to be read.
    public var available: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return count
    }

    /// Clear the ring and allow writes again after `abort()`.
    public func reset() {
        os_unfair_lock_lock(&lock)
        readPos = 0
        writePos = 0
        count = 0
        aborted = false
        os_unfair_lock_unlock(&lock)
    }

    /// Unblock a writer waiting on a full ring. Further `write` calls return
    /// immediately until `reset()`.
    public func abort() {
        os_unfair_lock_lock(&lock)
        aborted = true
        os_unfair_lock_unlock(&lock)
    }

    /// Copy `source` into the ring. If the ring is full the producer sleeps
    /// briefly off the lock so the audio thread can keep draining.
    public func write(_ source: [Float]) {
        guard !source.isEmpty else { return }
        var offset = 0
        while offset < source.count {
            os_unfair_lock_lock(&lock)
            if aborted {
                os_unfair_lock_unlock(&lock)
                return
            }
            if count == capacity {
                os_unfair_lock_unlock(&lock)
                Thread.sleep(forTimeInterval: 0.001)
                continue
            }
            let n = min(source.count - offset, capacity - count, capacity - writePos)
            source.withUnsafeBufferPointer { src in
                samples.advanced(by: writePos).update(from: src.baseAddress! + offset, count: n)
            }
            writePos = (writePos + n) % capacity
            count += n
            offset += n
            os_unfair_lock_unlock(&lock)
        }
    }

    /// Pull up to `requested` samples into `dest`. Never waits. Returns how
    /// many real samples were copied; any shortfall is filled with zeros.
    @discardableResult
    public func read(into dest: UnsafeMutablePointer<Float>, count requested: Int) -> Int {
        os_unfair_lock_lock(&lock)
        let n = min(requested, count)
        if n > 0 {
            let first = min(n, capacity - readPos)
            dest.update(from: samples.advanced(by: readPos), count: first)
            if first < n {
                dest.advanced(by: first).update(from: samples, count: n - first)
            }
            readPos = (readPos + n) % capacity
            count -= n
        }
        os_unfair_lock_unlock(&lock)
        if n < requested {
            dest.advanced(by: n).update(repeating: 0, count: requested - n)
        }
        return n
    }
}
