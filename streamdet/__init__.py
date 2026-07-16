"""streamdet: streaming, compute-adaptive AI-generated-video detection.

Reference implementation for "Detect-as-It-Streams: Anytime, Compute-Adaptive
Detection of AI-Generated Video from the Compressed Bitstream".

Subpackages, each a stage of the pipeline (see README.md for the end-to-end
reproduction; every stage module is runnable with `python -m`):

    streamdet.features    codec motion vectors / pixel embeddings -> per-chunk features
    streamdet.scoring     leave-one-generator-out readouts (streaming and clip-level)
    streamdet.escalation  stage-2 VLM scoring
    streamdet.analysis    anytime curves, cascade/deferral, motion-bias, paper numbers
    streamdet.data        cell manifests and censuses
    streamdet.bench       latency / MACs measurement
    streamdet.metrics     NumPy streaming metrics + synthetic proofs of the theory

External dependency: a third-party codec-forensics toolkit ("VidAudit") supplies
the motion-vector feature extractor (the 13-d spectral feature), the audited
leave-one-generator-out protocol, and the detector zoo (CLIP, Ivy-xDetector).
Install it, or point VIDAUDIT_PATH at a checkout:

    export VIDAUDIT_PATH=/path/to/vidaudit
"""
import contextlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Optional checkout of the third-party toolkit, if it is not already importable.
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
