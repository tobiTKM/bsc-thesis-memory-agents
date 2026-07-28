"""Finds the vLLM endpoint.

SLURM assigns the GPU node at submit time, so the server's hostname is not known
in advance. The serve job derives a port from its job id, waits until vLLM is
healthy, then exports LLM_BASE_URL / LLM_MODEL for the driver it launches (see
scripts/serve_and_run.sbatch). Reading them back here keeps the driver pointed at
whatever node SLURM allocated, and lets you aim it at any other OpenAI-compatible
endpoint by hand.

There used to be a fallback that read a `.server/latest.json` discovery file
written by a `scripts/serve_vllm.sbatch`. Neither has existed since the serve and
run steps were merged into one job, so the branch was unreachable and its error
message told you to run a script that is not in the repo.
"""

import os


def resolve_server() -> tuple[str, str]:
    """Return (base_url, model) for the LLM endpoint, from the environment."""
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not base_url or not model:
        # Name the missing one: setting only half the pair used to produce an
        # error that pointed somewhere else entirely.
        missing = ", ".join(name for name, value in
                            (("LLM_BASE_URL", base_url), ("LLM_MODEL", model))
                            if not value)
        raise RuntimeError(
            f"No LLM server configured: {missing} not set. "
            f"scripts/serve_and_run.sbatch exports both once vLLM is healthy; set "
            f"them by hand to target another OpenAI-compatible endpoint."
        )
    return base_url, model
