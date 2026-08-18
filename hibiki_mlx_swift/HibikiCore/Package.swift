// swift-tools-version: 6.0
import PackageDescription

// HibikiCore holds the native MLX Swift implementation of the S2S Edge
// inference path (config, artifact-bundle loading, the Mimi codec, and the
// Hibiki generator). It is a separate package so the same code builds into the
// iOS app *and* runs under `swift test` on macOS, where MLX has a real Metal
// GPU — that is how the loaded-model and inference-session seams are verified
// without a phone.
let package = Package(
    name: "HibikiCore",
    platforms: [
        .iOS(.v18),
        .macOS(.v14),
    ],
    products: [
        .library(name: "HibikiCore", targets: ["HibikiCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
    ],
    targets: [
        .target(
            name: "HibikiCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "HibikiCoreTests",
            dependencies: ["HibikiCore"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
