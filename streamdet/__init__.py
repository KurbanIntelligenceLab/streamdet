"""streamdet: streaming, compute-adaptive AI-generated-video detection.

Reference implementation for "Detect-as-It-Streams: Anytime, Compute-Adaptive
Detection of AI-Generated Video from the Compressed Bitstream".

Each module in this package is a runnable stage of the pipeline (see README.md
for the end-to-end reproduction). They import one another as siblings, so this
__init__ puts the package directory on sys.path.

External dependency: the VidAudit toolkit supplies the codec motion-vector
feature extractor (the 13-d TemporalSpec feature), the audited leave-one-
generator-out protocol, and the detector zoo (CLIP, Ivy-xDetector). Install it,
or point VIDAUDIT_PATH at a checkout:

    export VIDAUDIT_PATH=/path/to/vidaudit
"""
import contextlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Sibling modules (e.g. `import streamdet_metrics`, `from analyze_streaming
# import score_matrix`) resolve against the package directory.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Optional checkout of the VidAudit toolkit, if it is not already importable.
_VIDAUDIT = os.environ.get("VIDAUDIT_PATH")
if _VIDAUDIT and os.path.isdir(_VIDAUDIT) and _VIDAUDIT not in sys.path:
    sys.path.insert(0, _VIDAUDIT)

PROJECT_ROOT = os.path.dirname(_HERE)


@contextlib.contextmanager
def atomic_out(path):
    """Open `path` for writing via a tmp file + atomic rename.

    Duplicate tasks can race the same output shard (e.g. a re-queued array
    wave); a plain open('w') interleaves and corrupts it. With rename, the last
    finished writer wins with a consistent file, and a killed task never leaves
    a truncated shard that looks complete.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    f = open(tmp, "w", newline="")
    try:
        yield f
        f.close()
        os.replace(tmp, path)
    except BaseException:
        f.close()
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
