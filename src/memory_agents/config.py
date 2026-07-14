"""Finds the vLLM endpoint.

SLURM assigns the GPU node at submit time, so we don't know the server's
hostname in advance. The serve job writes its host:port to a small JSON file
once it is healthy (see scripts/serve_vllm.sbatch); this reads it back.
Environment variables win over the file, so you can also point the driver at
any OpenAI-compatible endpoint by hand.
"""

import json
import os
from pathlib import Path

# parents[2] = .../bsc-thesis-memory-agents (this file is src/memory_agents/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_FILE = PROJECT_ROOT / ".server" / "latest.json"


def resolve_server() -> tuple[str, str]:
    """Return (base_url, model) for the LLM endpoint.

    Order: LLM_BASE_URL / LLM_MODEL env vars first, then the discovery file.
    """
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if base_url and model:
        return base_url, model

    if not DISCOVERY_FILE.exists():
        raise RuntimeError(
            f"No LLM server found. Set LLM_BASE_URL and LLM_MODEL, or start the "
            f"server with scripts/serve_vllm.sbatch (writes {DISCOVERY_FILE})."
        )
    info = json.loads(DISCOVERY_FILE.read_text())
    return (base_url or f"http://{info['host']}:{info['port']}/v1",
            model or info["model"])
