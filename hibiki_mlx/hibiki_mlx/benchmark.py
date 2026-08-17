"""Benchmark Hibiki artifact bundles against a directory of WAV inputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, read_pcm
from .inference import LoadedModel, load_model
from .sampling import Sampler
from .session import DEFAULT_CONDITION, InferenceSession, StepResult


@dataclass(frozen=True)
class VariantResult:
    """One artifact bundle's load and warmup measurements."""

    name: str
    artifact_directory: Path
    load_seconds: float
    warmup_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifact_directory": str(self.artifact_directory),
            "load_seconds": self.load_seconds,
            "warmup_seconds": self.warmup_seconds,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    """One model variant's measurement for one source WAV."""

    variant: str
    asset: Path
    input_seconds: float
    steps: int
    processing_seconds: float
    output_seconds: float
    transcript: str
    mlx_active_bytes: int
    mlx_peak_bytes: int

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.processing_seconds if self.processing_seconds else 0.0

    @property
    def real_time_factor(self) -> float:
        return self.processing_seconds / self.input_seconds if self.input_seconds else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "asset": str(self.asset),
            "input_seconds": self.input_seconds,
            "steps": self.steps,
            "processing_seconds": self.processing_seconds,
            "steps_per_second": self.steps_per_second,
            "real_time_factor": self.real_time_factor,
            "output_seconds": self.output_seconds,
            "transcript": self.transcript,
            "mlx_active_bytes": self.mlx_active_bytes,
            "mlx_peak_bytes": self.mlx_peak_bytes,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """All variant results for one deterministic set of source WAV files."""

    assets_directory: Path
    variants: tuple[VariantResult, ...]
    cases: tuple[BenchmarkCase, ...]
    settings: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "assets_directory": str(self.assets_directory),
            "settings": self.settings,
            "variants": [variant.as_dict() for variant in self.variants],
            "cases": [case.as_dict() for case in self.cases],
            "summary": _summarize_cases(self.cases),
        }


@dataclass(frozen=True)
class BenchmarkSettings:
    """The fixed inference settings shared by every benchmark case."""

    seed: int = 299792458
    condition: str = DEFAULT_CONDITION
    temp: float = 0.8
    text_top_k: int = 25
    audio_top_k: int = 250


class BenchmarkSession(Protocol):
    """The session interface required to time one PCM input."""

    frame_size: int
    text: str

    def warmup(self) -> None: ...

    def push_pcm(self, pcm: np.ndarray) -> list[StepResult]: ...

    def finish(self) -> list[StepResult]: ...


class BenchmarkRuntime(Protocol):
    """The MLX runtime adapter used by the reusable benchmark runner."""

    def load_model(self, artifact_directory: Path) -> object: ...

    def make_session(self, model: object, settings: BenchmarkSettings) -> BenchmarkSession: ...

    def read_pcm(self, asset: Path) -> np.ndarray: ...

    def seed(self, seed: int) -> None: ...

    def reset_peak_memory(self) -> None: ...

    def memory_bytes(self) -> tuple[int, int]: ...

    def now(self) -> float: ...


class MlxBenchmarkRuntime:
    """The production adapter for the local MLX model loader and session."""

    def load_model(self, artifact_directory: Path) -> LoadedModel:
        return load_model(artifact_directory=artifact_directory)

    def make_session(
        self, model: object, settings: BenchmarkSettings
    ) -> InferenceSession:
        if not isinstance(model, LoadedModel):
            raise TypeError("the MLX benchmark runtime requires a LoadedModel")
        return InferenceSession(
            model,
            condition=settings.condition,
            text_sampler=Sampler(temp=settings.temp, top_k=settings.text_top_k),
            audio_sampler=Sampler(temp=settings.temp, top_k=settings.audio_top_k),
        )

    def read_pcm(self, asset: Path) -> np.ndarray:
        return read_pcm(asset)

    def seed(self, seed: int) -> None:
        mx.random.seed(seed)

    def reset_peak_memory(self) -> None:
        mx.reset_peak_memory()

    def memory_bytes(self) -> tuple[int, int]:
        return int(mx.get_active_memory()), int(mx.get_peak_memory())

    def now(self) -> float:
        return time.perf_counter()


def build_parser() -> argparse.ArgumentParser:
    """Build the public command interface for a BF16/Q8 comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-artifacts",
        type=Path,
        default=Path("artifacts/hibiki-1b-mlx-bf16"),
        help="BF16 artifact bundle directory",
    )
    parser.add_argument(
        "--q8-artifacts",
        type=Path,
        default=Path("artifacts/hibiki-1b-mlx-q8"),
        help="Q8 artifact bundle directory",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("assets"),
        help="directory recursively containing source WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for benchmark.json and benchmark.csv",
    )
    parser.add_argument("--seed", type=int, default=BenchmarkSettings.seed)
    parser.add_argument("--temp", type=float, default=BenchmarkSettings.temp)
    parser.add_argument("--text-top-k", type=int, default=BenchmarkSettings.text_top_k)
    parser.add_argument("--audio-top-k", type=int, default=BenchmarkSettings.audio_top_k)
    parser.add_argument("--condition", default=BenchmarkSettings.condition)
    parser.add_argument("--_worker-variant", choices=("bf16", "q8"), help=argparse.SUPPRESS)
    parser.add_argument("--_worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def discover_assets(directory: Path) -> list[Path]:
    """Return every WAV file below ``directory`` in a deterministic order."""
    return sorted(path for path in directory.rglob("*.wav") if path.is_file())


def benchmark_variant(
    runtime: BenchmarkRuntime,
    name: str,
    artifact_directory: Path,
    assets: list[Path],
    settings: BenchmarkSettings,
) -> tuple[VariantResult, tuple[BenchmarkCase, ...]]:
    """Load and warm one variant once, then benchmark each asset in a fresh session."""
    runtime.seed(settings.seed)
    started = runtime.now()
    model = runtime.load_model(artifact_directory)
    load_seconds = runtime.now() - started

    warmup_session = runtime.make_session(model, settings)
    started = runtime.now()
    warmup_session.warmup()
    warmup_seconds = runtime.now() - started

    cases = []
    for asset in assets:
        runtime.seed(settings.seed)
        pcm = runtime.read_pcm(asset)
        session = runtime.make_session(model, settings)
        runtime.reset_peak_memory()
        started = runtime.now()
        results = _translate(session, pcm)
        processing_seconds = runtime.now() - started
        mlx_active_bytes, mlx_peak_bytes = runtime.memory_bytes()
        cases.append(
            BenchmarkCase(
                variant=name,
                asset=asset,
                input_seconds=len(pcm) / SAMPLE_RATE,
                steps=len(results),
                processing_seconds=processing_seconds,
                output_seconds=sum(
                    len(result.pcm) for result in results if result.pcm is not None
                )
                / SAMPLE_RATE,
                transcript=session.text,
                mlx_active_bytes=mlx_active_bytes,
                mlx_peak_bytes=mlx_peak_bytes,
            )
        )
    return (
        VariantResult(
            name=name,
            artifact_directory=artifact_directory,
            load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,
        ),
        tuple(cases),
    )


def _translate(session: BenchmarkSession, pcm: np.ndarray) -> list[StepResult]:
    results = []
    for start in range(0, len(pcm), session.frame_size):
        results.extend(session.push_pcm(pcm[start : start + session.frame_size]))
    results.extend(session.finish())
    return results


def write_report(report: BenchmarkReport, directory: Path) -> None:
    """Write the complete benchmark evidence as JSON and per-file CSV rows."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark.json").write_text(json.dumps(report.as_dict(), indent=2) + "\n")

    fields = [
        "variant",
        "asset",
        "input_seconds",
        "steps",
        "processing_seconds",
        "steps_per_second",
        "real_time_factor",
        "output_seconds",
        "mlx_active_bytes",
        "mlx_peak_bytes",
        "transcript",
    ]
    with (directory / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in report.cases:
            row = case.as_dict()
            for name in (
                "input_seconds",
                "processing_seconds",
                "steps_per_second",
                "real_time_factor",
                "output_seconds",
            ):
                row[name] = f"{row[name]:.6f}"
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    """Run BF16 and Q8 workers in clean processes, then merge their reports."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    settings = BenchmarkSettings(
        seed=arguments.seed,
        condition=arguments.condition,
        temp=arguments.temp,
        text_top_k=arguments.text_top_k,
        audio_top_k=arguments.audio_top_k,
    )
    if arguments._worker_variant or arguments._worker_result:
        if not arguments._worker_variant or arguments._worker_result is None:
            parser.error("internal benchmark workers require both worker arguments")
        return _run_worker(arguments, settings)

    assets = discover_assets(arguments.assets)
    if not assets:
        parser.error(f"no WAV files found below {arguments.assets}")
    output_directory = arguments.output_dir or _default_output_directory()
    if output_directory.exists():
        parser.error(f"output directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)

    variants = (("bf16", arguments.bf16_artifacts), ("q8", arguments.q8_artifacts))
    reports = []
    for name, artifact_directory in variants:
        worker_result = output_directory / f"{name}.worker.json"
        subprocess.run(
            _worker_command(arguments, name, worker_result),
            check=True,
        )
        reports.append(_read_worker_report(worker_result))

    report = BenchmarkReport(
        assets_directory=arguments.assets,
        variants=tuple(variant for worker in reports for variant in worker.variants),
        cases=tuple(case for worker in reports for case in worker.cases),
        settings=_settings_dict(settings),
    )
    write_report(report, output_directory)
    print(f"[hibiki-benchmark] wrote {output_directory / 'benchmark.json'}")
    print(f"[hibiki-benchmark] wrote {output_directory / 'benchmark.csv'}")
    return 0


def _run_worker(arguments: argparse.Namespace, settings: BenchmarkSettings) -> int:
    assets = discover_assets(arguments.assets)
    if not assets:
        raise ValueError(f"no WAV files found below {arguments.assets}")
    variants = {
        "bf16": arguments.bf16_artifacts,
        "q8": arguments.q8_artifacts,
    }
    name = arguments._worker_variant
    assert name is not None
    variant, cases = benchmark_variant(
        MlxBenchmarkRuntime(),
        name,
        variants[name],
        assets,
        settings,
    )
    report = BenchmarkReport(
        assets_directory=arguments.assets,
        variants=(variant,),
        cases=cases,
        settings=_settings_dict(settings),
    )
    result_path = arguments._worker_result
    assert result_path is not None
    result_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    print(f"[hibiki-benchmark] completed {name}: {len(cases)} files")
    return 0


def _worker_command(
    arguments: argparse.Namespace, name: str, worker_result: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "hibiki_mlx.benchmark",
        "--bf16-artifacts",
        str(arguments.bf16_artifacts),
        "--q8-artifacts",
        str(arguments.q8_artifacts),
        "--assets",
        str(arguments.assets),
        "--seed",
        str(arguments.seed),
        "--temp",
        str(arguments.temp),
        "--text-top-k",
        str(arguments.text_top_k),
        "--audio-top-k",
        str(arguments.audio_top_k),
        "--condition",
        arguments.condition,
        "--_worker-variant",
        name,
        "--_worker-result",
        str(worker_result),
    ]


def _read_worker_report(path: Path) -> BenchmarkReport:
    document = json.loads(path.read_text())
    variants = tuple(
        VariantResult(
            name=item["name"],
            artifact_directory=Path(item["artifact_directory"]),
            load_seconds=item["load_seconds"],
            warmup_seconds=item["warmup_seconds"],
        )
        for item in document["variants"]
    )
    cases = tuple(
        BenchmarkCase(
            variant=item["variant"],
            asset=Path(item["asset"]),
            input_seconds=item["input_seconds"],
            steps=item["steps"],
            processing_seconds=item["processing_seconds"],
            output_seconds=item["output_seconds"],
            transcript=item["transcript"],
            mlx_active_bytes=item["mlx_active_bytes"],
            mlx_peak_bytes=item["mlx_peak_bytes"],
        )
        for item in document["cases"]
    )
    return BenchmarkReport(
        assets_directory=Path(document["assets_directory"]),
        variants=variants,
        cases=cases,
        settings=document.get("settings", {}),
    )


def _settings_dict(settings: BenchmarkSettings) -> dict[str, object]:
    return {
        "seed": settings.seed,
        "condition": settings.condition,
        "temp": settings.temp,
        "text_top_k": settings.text_top_k,
        "audio_top_k": settings.audio_top_k,
    }


def _default_output_directory() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmarks") / f"hibiki-bf16-q8-{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())


def _summarize_cases(cases: tuple[BenchmarkCase, ...]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for variant in sorted({case.variant for case in cases}):
        matching = [case for case in cases if case.variant == variant]
        input_seconds = sum(case.input_seconds for case in matching)
        processing_seconds = sum(case.processing_seconds for case in matching)
        steps = sum(case.steps for case in matching)
        summary[variant] = {
            "cases": len(matching),
            "input_seconds": input_seconds,
            "processing_seconds": processing_seconds,
            "steps": steps,
            "steps_per_second": steps / processing_seconds if processing_seconds else 0.0,
            "real_time_factor": processing_seconds / input_seconds if input_seconds else 0.0,
        }
    return summary
