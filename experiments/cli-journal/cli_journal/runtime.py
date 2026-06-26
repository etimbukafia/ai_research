from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings


def configure_quiet_runtime() -> None:
    """Keep noisy third-party memory libraries from writing to the CLI."""

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")
    os.environ.setdefault("CHROMADB_TELEMETRY", "False")
    os.environ.setdefault("POSTHOG_DISABLED", "True")

    warnings.filterwarnings("ignore", message=".*model_fields.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"chromadb(\.|$).*")
    warnings.filterwarnings("ignore", message=".*does not support keyword search.*")
    warnings.filterwarnings("ignore", message=".*Number of requested results.*")

    for name in [
        "chromadb",
        "chromadb.telemetry",
        "chromadb.telemetry.product",
        "chromadb.telemetry.product.posthog",
        "posthog",
    ]:
        logger = logging.getLogger(name)
        logger.disabled = True
        logger.propagate = False


@contextlib.contextmanager
def quiet_third_party_output():
    """Capture stdout/stderr and warnings from mem0/Chroma calls."""

    configure_quiet_runtime()
    stdout = io.StringIO()
    stderr = io.StringIO()
    with warnings.catch_warnings():
        configure_quiet_runtime()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield stdout, stderr


def captured_output(streams: tuple[io.StringIO, io.StringIO]) -> str:
    output = "\n".join(part.strip() for part in (streams[0].getvalue(), streams[1].getvalue()) if part.strip())
    return "\n".join(line for line in output.splitlines() if line.strip())
