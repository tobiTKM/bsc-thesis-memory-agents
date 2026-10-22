"""Finds the vLLM endpoint from LLM_BASE_URL / LLM_MODEL.

scripts/serve_and_run.sbatch exports both once vLLM is healthy on the node SLURM
allocated; set them by hand to target any other OpenAI-compatible endpoint.
"""

import os


def resolve_server() -> tuple[str, str]:
    """Return (base_url, model) for the LLM endpoint, from the environment."""
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not base_url or not model:
        missing = ", ".join(name for name, value in
                            (("LLM_BASE_URL", base_url), ("LLM_MODEL", model))
                            if not value)
        raise RuntimeError(
            f"No LLM server configured: {missing} not set. "
            f"scripts/serve_and_run.sbatch exports both once vLLM is healthy; set "
            f"them by hand to target another OpenAI-compatible endpoint."
        )
    return base_url, model
