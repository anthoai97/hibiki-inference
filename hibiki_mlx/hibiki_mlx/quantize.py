"""Create a self-contained Q8 or Q4 Hibiki MLX artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import QuantizationSpec, convert_bundle, validate_quantization_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BF16 artifact bundle directory")
    parser.add_argument("destination", type=Path, help="new Q8/Q4 artifact bundle directory")
    parser.add_argument("--bits", choices=(4, 8), type=int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate paths without converting")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    spec = QuantizationSpec.for_bits(arguments.bits)
    if arguments.dry_run:
        validate_quantization_request(arguments.source, arguments.destination)
        print(
            f"would create Q{spec.bits} bundle at {arguments.destination} "
            f"with group_size={spec.group_size}"
        )
        return 0

    destination = convert_bundle(arguments.source, arguments.destination, spec)
    print(f"created Q{spec.bits} bundle at {destination}")
    return 0
