"""Tests for the reusable BF16/Q8 benchmark command."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from hibiki_mlx.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkSettings,
    TranscriptComparison,
    VariantResult,
    benchmark_variant,
    build_parser,
    compare_transcripts,
    discover_assets,
    load_reference_transcripts,
    main,
    write_report,
)
from hibiki_mlx.session import StepResult


class AssetDiscoveryTests(unittest.TestCase):
    def test_discovers_every_wav_recursively_in_stable_order(self) -> None:
        with TemporaryDirectory() as temporary:
            assets = Path(temporary)
            (assets / "long-form").mkdir()
            (assets / "short-form").mkdir()
            (assets / "long-form" / "b.wav").write_bytes(b"")
            (assets / "short-form" / "a.wav").write_bytes(b"")
            (assets / "short-form" / "ignored.txt").write_text("")

            paths = discover_assets(assets)

        self.assertEqual(paths, [assets / "long-form" / "b.wav", assets / "short-form" / "a.wav"])


class ReportWritingTests(unittest.TestCase):
    def test_writes_machine_readable_per_file_and_summary_results(self) -> None:
        report = BenchmarkReport(
            assets_directory=Path("assets"),
            variants=(
                VariantResult(
                    name="q8",
                    artifact_directory=Path("artifacts/q8"),
                    load_seconds=1.5,
                    warmup_seconds=0.2,
                ),
            ),
            cases=(
                BenchmarkCase(
                    variant="q8",
                    asset=Path("assets/short-form/a.wav"),
                    input_seconds=10.0,
                    steps=132,
                    processing_seconds=5.0,
                    output_seconds=9.6,
                    transcript="Hello",
                    mlx_active_bytes=4 * 2**20,
                    mlx_peak_bytes=5 * 2**20,
                ),
            ),
            reference_transcripts={"short-form/a.wav": "Hello"},
            reference_transcript_file=Path("assets/transcripts.json"),
        )
        with TemporaryDirectory() as temporary:
            output_directory = Path(temporary) / "results"

            write_report(report, output_directory)

            document = json.loads((output_directory / "benchmark.json").read_text())
            with (output_directory / "benchmark.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            with (output_directory / "accuracy.csv").open(newline="") as handle:
                accuracy_rows = list(csv.DictReader(handle))

        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["summary"]["q8"]["steps_per_second"], 26.4)
        self.assertEqual(rows[0]["asset"], "assets/short-form/a.wav")
        self.assertEqual(rows[0]["real_time_factor"], "0.500000")
        self.assertEqual(document["accuracy"]["q8"]["word_error_rate"], 0.0)
        self.assertEqual(accuracy_rows[0]["reference_key"], "short-form/a.wav")


class TranscriptComparisonTests(unittest.TestCase):
    def test_compares_each_variant_against_the_reference_transcript(self) -> None:
        cases = (
            BenchmarkCase(
                variant="q8",
                asset=Path("assets/a.wav"),
                input_seconds=1.0,
                steps=1,
                processing_seconds=1.0,
                output_seconds=1.0,
                transcript="hello word",
                mlx_active_bytes=0,
                mlx_peak_bytes=0,
            ),
        )

        comparisons = compare_transcripts(
            cases,
            assets_directory=Path("assets"),
            references={"a.wav": "Hello, world!"},
        )

        self.assertEqual(
            comparisons,
            (
                TranscriptComparison(
                    asset=Path("assets/a.wav"),
                    variant="q8",
                    reference_key="a.wav",
                    reference_transcript="Hello, world!",
                    candidate_transcript="hello word",
                    reference_words=2,
                    word_errors=1,
                ),
            ),
        )
        self.assertEqual(comparisons[0].word_error_rate, 0.5)

    def test_loads_the_text_field_from_the_reference_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "transcripts.json"
            manifest.write_text(
                json.dumps({"short/a.wav": {"language": "fr", "text": "Hello world"}})
            )

            references = load_reference_transcripts(manifest)

        self.assertEqual(references, {"short/a.wav": "Hello world"})


def step(index: int, pcm: np.ndarray | None = None) -> StepResult:
    return StepResult(
        text_frame_index=index,
        text_token=index,
        text=str(index),
        audio_frame_index=index - 2 if pcm is not None else None,
        pcm=pcm,
        seconds_per_frame=0.08,
    )


class RecordingSession:
    frame_size = 3

    def __init__(self) -> None:
        self.warmed = False
        self.calls = 0
        self.text = "translated"

    def warmup(self) -> None:
        self.warmed = True

    def push_pcm(self, _pcm: np.ndarray) -> list[StepResult]:
        self.calls += 1
        return [step(self.calls, np.zeros(3, dtype=np.float32))]

    def finish(self) -> list[StepResult]:
        return [step(3)]


class RecordingRuntime:
    def __init__(self) -> None:
        self.loaded: list[Path] = []
        self.sessions: list[RecordingSession] = []
        self.seeds: list[int] = []
        self.peak_resets = 0
        self._times = iter((0.0, 1.5, 2.0, 2.2, 3.0, 5.0))

    def load_model(self, artifact_directory: Path) -> object:
        self.loaded.append(artifact_directory)
        return object()

    def make_session(self, _model: object, _settings: BenchmarkSettings) -> RecordingSession:
        session = RecordingSession()
        self.sessions.append(session)
        return session

    def read_pcm(self, _asset: Path) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def reset_peak_memory(self) -> None:
        self.peak_resets += 1

    def memory_bytes(self) -> tuple[int, int]:
        return (4 * 2**20, 5 * 2**20)

    def now(self) -> float:
        return next(self._times)


class VariantBenchmarkTests(unittest.TestCase):
    def test_loads_and_warms_once_then_measures_each_asset_in_a_fresh_session(self) -> None:
        runtime = RecordingRuntime()
        asset = Path("assets/short-form/a.wav")
        progress: list[str] = []

        variant, cases = benchmark_variant(
            runtime,
            "q8",
            Path("artifacts/q8"),
            [asset],
            BenchmarkSettings(seed=17),
            progress=progress.append,
        )

        self.assertEqual(variant.name, "q8")
        self.assertEqual(variant.load_seconds, 1.5)
        self.assertAlmostEqual(variant.warmup_seconds, 0.2)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].asset, asset)
        self.assertEqual(cases[0].steps, 3)
        self.assertEqual(cases[0].processing_seconds, 2.0)
        self.assertEqual(cases[0].mlx_peak_bytes, 5 * 2**20)
        self.assertEqual(runtime.loaded, [Path("artifacts/q8")])
        self.assertTrue(runtime.sessions[0].warmed)
        self.assertEqual(len(runtime.sessions), 2)
        self.assertEqual(runtime.seeds, [17, 17])
        self.assertEqual(runtime.peak_resets, 1)
        self.assertEqual(progress[0], "q8: loading artifacts/q8")
        self.assertIn("q8: [1/1] translating assets/short-form/a.wav", progress)
        self.assertIn("q8: [1/1] 3 steps in 2.0s", progress[-1])


class CommandInterfaceTests(unittest.TestCase):
    def test_accepts_explicit_bf16_q8_assets_and_output_paths(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--bf16-artifacts",
                "artifacts/bf16",
                "--q8-artifacts",
                "artifacts/q8",
                "--assets",
                "assets",
                "--transcripts",
                "assets/transcripts.json",
                "--output-dir",
                "benchmarks/result",
            ]
        )

        self.assertEqual(arguments.bf16_artifacts, Path("artifacts/bf16"))
        self.assertEqual(arguments.q8_artifacts, Path("artifacts/q8"))
        self.assertEqual(arguments.assets, Path("assets"))
        self.assertEqual(arguments.transcripts, Path("assets/transcripts.json"))
        self.assertEqual(arguments.output_dir, Path("benchmarks/result"))

    def test_runs_isolated_variant_workers_and_merges_their_reports(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            (assets / "source.wav").write_bytes(b"")
            output_directory = root / "results"

            def run_worker(command: list[str], *, check: bool) -> None:
                variant = command[command.index("--_worker-variant") + 1]
                worker_output = Path(command[command.index("--_worker-result") + 1])
                report = BenchmarkReport(
                    assets_directory=assets,
                    variants=(
                        VariantResult(
                            name=variant,
                            artifact_directory=root / variant,
                            load_seconds=1.0,
                            warmup_seconds=0.1,
                        ),
                    ),
                    cases=(
                        BenchmarkCase(
                            variant=variant,
                            asset=assets / "source.wav",
                            input_seconds=1.0,
                            steps=13,
                            processing_seconds=0.5,
                            output_seconds=0.8,
                            transcript="text",
                            mlx_active_bytes=1,
                            mlx_peak_bytes=2,
                        ),
                    ),
                )
                worker_output.write_text(json.dumps(report.as_dict()))

            with patch("hibiki_mlx.benchmark.subprocess.run", side_effect=run_worker):
                exit_code = main(
                    [
                        "--bf16-artifacts",
                        str(root / "bf16"),
                        "--q8-artifacts",
                        str(root / "q8"),
                        "--assets",
                        str(assets),
                        "--output-dir",
                        str(output_directory),
                    ]
                )

            document = json.loads((output_directory / "benchmark.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(document["summary"]["bf16"]["cases"], 1)
        self.assertEqual(document["summary"]["q8"]["cases"], 1)

    def test_resume_merges_existing_worker_reports_without_running_models(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            (assets / "source.wav").write_bytes(b"")
            output_directory = root / "results"
            output_directory.mkdir()
            for variant in ("bf16", "q8"):
                report = BenchmarkReport(
                    assets_directory=assets,
                    variants=(
                        VariantResult(
                            name=variant,
                            artifact_directory=root / variant,
                            load_seconds=1.0,
                            warmup_seconds=0.1,
                        ),
                    ),
                    cases=(
                        BenchmarkCase(
                            variant=variant,
                            asset=assets / "source.wav",
                            input_seconds=1.0,
                            steps=13,
                            processing_seconds=0.5,
                            output_seconds=0.8,
                            transcript="text",
                            mlx_active_bytes=1,
                            mlx_peak_bytes=2,
                        ),
                    ),
                )
                (output_directory / f"{variant}.worker.json").write_text(
                    json.dumps(report.as_dict())
                )

            with patch("hibiki_mlx.benchmark.subprocess.run") as run:
                exit_code = main(
                    [
                        "--bf16-artifacts",
                        str(root / "bf16"),
                        "--q8-artifacts",
                        str(root / "q8"),
                        "--assets",
                        str(assets),
                        "--output-dir",
                        str(output_directory),
                        "--resume",
                    ]
                )

            document = json.loads((output_directory / "benchmark.json").read_text())

        self.assertEqual(exit_code, 0)
        run.assert_not_called()
        self.assertEqual(document["summary"]["bf16"]["cases"], 1)
        self.assertEqual(document["summary"]["q8"]["cases"], 1)
