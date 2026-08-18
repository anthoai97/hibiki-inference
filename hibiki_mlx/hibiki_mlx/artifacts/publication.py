"""Publish validated quantized artifact bundles through a Hub adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .bundle import validate_quantized_bundle
from .quantization import QuantizationSpec


class HubApi(Protocol):
    """The small Hugging Face adapter required to publish one bundle."""

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> object: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        path_in_repo: str,
        revision: str,
        commit_message: str,
    ) -> object: ...


@dataclass(frozen=True)
class Publication:
    """The result of publishing one validated quantized bundle."""

    spec: QuantizationSpec
    commit: object


def publish_quantized_bundle(
    api: HubApi,
    directory: Path,
    *,
    repo_id: str,
    private: bool,
    revision: str,
    commit_message: str,
) -> Publication:
    """Validate, create if necessary, and upload one complete artifact bundle."""
    spec = validate_quantized_bundle(directory)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(directory),
        path_in_repo=".",
        revision=revision,
        commit_message=commit_message,
    )
    return Publication(spec=spec, commit=commit)
