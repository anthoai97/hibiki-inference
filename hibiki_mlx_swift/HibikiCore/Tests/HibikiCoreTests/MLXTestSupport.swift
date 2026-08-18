import MLX

enum MLXTestSupport {
    /// `swift test` runs without MLX's Metal shader bundle (metallib), so the
    /// GPU backend can't initialize here. Force the CPU backend for all MLX ops
    /// in tests: parity is a numerical property, not a performance one, and the
    /// real Metal GPU path is verified on the device (ticket #26).
    static func forceCPUDevice() {
        Device.setDefault(device: Device(.cpu))
    }
}
