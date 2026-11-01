"""Global configuration for the Vietnamese History GraphRAG system.

Settings are loaded from ``configs/default.yaml`` and can be overridden by
environment variables.  Nothing in the codebase reads YAML directly -- always
go through :func:`get_settings` so that a single source of truth is preserved.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _load_dotenv() -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


@dataclass
class Settings:
    """Flattened view of ``configs/default.yaml`` plus env overrides."""

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = REPO_ROOT

    # ---------------------------------------------------------------- paths
    @property
    def data_raw(self) -> Path:
        return self.root / self.raw["paths"]["data_raw"]

    @property
    def data_processed(self) -> Path:
        return self.root / self.raw["paths"]["data_processed"]

    @property
    def artifacts(self) -> Path:
        return self.root / self.raw["paths"]["artifacts"]

    @property
    def reports(self) -> Path:
        return self.root / self.raw["paths"]["reports"]

    @property
    def cache_dir(self) -> Path:
        return self.root / self.raw["paths"]["cache"]

    # ------------------------------------------------------------------ llm
    @property
    def llm_provider(self) -> str:
        return os.getenv("HGRAG_LLM_PROVIDER", self.raw["llm"]["provider"])

    @property
    def llm_model(self) -> str:
        return os.getenv("GEMINI_MODEL", self.raw["llm"]["model"])

    @property
    def embed_model(self) -> str:
        return os.getenv("GEMINI_EMBED_MODEL", self.raw["llm"]["embed_model"])

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw["llm"]

    # ------------------------------------------------------------- embedding
    @property
    def embedding(self) -> dict[str, Any]:
        return self.raw["embedding"]

    @property
    def embed_provider(self) -> str:
        return os.getenv("HGRAG_EMBED_PROVIDER", self.raw["embedding"]["provider"])

    @property
    def embed_dim(self) -> int:
        return int(self.raw["embedding"]["dim"])

    # ----------------------------------------------------------------- neo4j
    @property
    def neo4j(self) -> dict[str, Any]:
        return self.raw["neo4j"]

    @property
    def neo4j_uri(self) -> str:
        port = os.getenv("NEO4J_BOLT_PORT", str(self.raw["neo4j"]["uri"].rsplit(":", 1)[-1]))
        return f"bolt://localhost:{port}"

    @property
    def neo4j_password(self) -> str:
        return os.getenv(
            "NEO4J_PASSWORD", self.raw["neo4j"].get("default_password", "history2025")
        )

    # --------------------------------------------------------------- qdrant
    @property
    def qdrant(self) -> dict[str, Any]:
        return self.raw["qdrant"]

    @property
    def qdrant_url(self) -> str:
        port = os.getenv("QDRANT_PORT", str(self.raw["qdrant"]["url"].rsplit(":", 1)[-1]))
        return f"http://localhost:{port}"

    # -------------------------------------------------------------- others
    @property
    def ingest(self) -> dict[str, Any]:
        return self.raw["ingest"]

    @property
    def chunking(self) -> dict[str, Any]:
        return self.raw["chunking"]

    @property
    def extraction(self) -> dict[str, Any]:
        return self.raw["extraction"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def seed(self) -> int:
        return int(self.raw["evaluation"]["seed"])

    def ensure_dirs(self) -> None:
        for p in (
            self.data_raw,
            self.data_processed,
            self.artifacts,
            self.reports,
            self.cache_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Load (and cache) the settings object."""
    _load_dotenv()
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(raw=raw, root=REPO_ROOT)
