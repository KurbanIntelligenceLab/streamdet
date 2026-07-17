# Detect-as-It-Streams

Reference implementation for **"Detect-as-It-Streams: Anytime, Compute-Adaptive Detection of
AI-Generated Video from the Compressed Bitstream."**

Detectors for AI-generated video (AIGV) are evaluated almost entirely offline: decode a whole clip
to pixels, score it once, increasingly with a large VLM. Deployment is the opposite — video arrives
as a stream, under a latency budget, at a scale where running a heavy detector on every clip is
uneconomical. This repo recasts AIGV detection as a **streaming** problem:

* an **anytime** front-end that scores the codec motion-vector field per chunk (CPU-only,
  ~10<sup>5</sup> MACs/chunk, no pixel-domain forward pass) and exits as soon as it is confident,
  under a single end-calibrated threshold that is *anytime-valid*;
* **reasoning on demand** — only near-boundary segments (~15%) escalate to a pixel CNN or a small
  VLM, tracing a measured compute–accuracy frontier.

This code produces every number in the paper (E1–E7).

---

## Repository layout

```
streamdet/                   the package; every stage runs with `python -m`
  metrics.py                 NumPy streaming metrics + a synthetic proof-by-assertion of
                             Propositions 1-3 and Corollary 1 (self-contained)
  features/                  clips -> per-chunk features
    gop_features.py            codec motion vectors -> per-chunk 13-d features (stage 1)
    clip_features_from_npy.py  clip-level 13-d features from precomputed MV arrays
    pixel_chunk_features.py    stage-2 pixel arm: per-chunk CLIP ViT-B/32 embeddings
    restrav_features.py        baseline: ReStraV (NeurIPS 2025) 21-d features per clip
  scoring/                   leave-one-generator-out readouts
    streaming_scores.py        per-chunk LOGO scoring + abstain protocol
    e1_readout.py              audited LOGO/RvR readout (the E1 positive control)
    restrav_readout.py         baseline: LOGO readout over the ReStraV features
  escalation/                stage-2 reasoning
    vlm_scores.py              Ivy-xDetector (Qwen2.5-VL-3B) p(generated) per clip
    vv_scores.py               baseline: VideoVeritas (ICML 2026, Qwen3-VL-8B) verdicts
  analysis/                  measurement and figures
    streaming.py               anytime AUC(t), gate calibration, stopping-time FPR, latency
    cascade.py                 deferral sweep, E[C] accounting, deferral gain (Prop. 3 / Cor. 1)
    deferral_rule.py           learned deferral rule vs the confidence band
    motionbias.py              motion-bias control (matched + within-bin AUC)
    vlm.py                     VLM baseline + codec->VLM cascade
    paper_numbers.py           recompute every number the paper cites
    significance.py            bootstrap CIs + paired tests (McNemar) for every
                               decision-accuracy point; AUC@N reconciliation;
                               aggregator ablation (max vs mean vs last)
  data/                      cells and censuses
    manifest_longform.py       long-form (length-matched) cell
    manifest_videos.py         manifest from a video tree (e.g. the cross-dataset cell)
    subsample_manifest.py      balanced subsample (for the VLM arm)
    survey_cell.py             per-clip frame/chunk census of a cell
  bench/
    latency.py                 per-chunk wall-clock and MACs, codec vs pixel
scripts/                     one script per stage, grouped by experiment
  e1_control/  e2_latency_accuracy/  e3e4_compute_accuracy/
  e5_earliness/  e6_cross_dataset/  e7_ablations/  e8_baselines/  common/
tests/
  smoke_local.py             end-to-end test on synthesized clips (no cluster, ~1 min)
```

## Third-party dependencies

Two external toolkits are used and are **not** included here. Both are independent projects; obtain
them from their own distributions.

| Toolkit | What this repo uses it for | Required? |
|---|---|---|
| **VidAudit** — an audited benchmark/toolkit for AI-generated-video detection | The codec motion-vector feature extractor (the 13-d spectral feature over a 16×16 macroblock grid), the audited leave-one-generator-out protocol and its splits/seeds, the canonical H.264 re-encode, and the detector zoo (CLIP ViT-B/32; Ivy-xDetector). Cited in the paper as concurrent work. | **Yes** |
| **mv-extractor** — a batch extractor for codec motion vectors | Produces the per-clip `[T, H_mb, W_mb, 5]` motion-vector `.npy` arrays (plus `frame_types.txt`) that `features/gop_features.py` consumes in its `--npy-path` input mode, and the canonical re-encode recipe those arrays assume. Only needed if you precompute MV arrays rather than decoding mp4s directly. | Optional |

> **Anonymity note.** Both are third-party projects with respect to this submission; their
> repositories are referenced in the paper as anonymized concurrent/related work, and direct links
> are omitted here for double-blind review. They will be linked in the camera-ready. If you are a
> reviewer and need them to reproduce, request them through the submission system.

## Setup

**Prerequisites:** [conda](https://docs.conda.io/) (Miniforge/Miniconda) and `git`. A GPU is needed
only for the stage-2 arms (pixel CNN, VLM); stage 1 is CPU-only. `ffmpeg` comes with the
environment and is required for the canonical re-encode path.

```bash
# 1. Environment
conda env create -f environment.yml
conda activate streamdet

# 2. Point at the third-party toolkit (see the table above)
export VIDAUDIT_PATH=/path/to/vidaudit

# 3. Verify the install end-to-end: synthesizes clips, runs the full stage-1
#    pipeline, asserts the plumbing. ~1 min on CPU.
PYTHONPATH="$PWD" python tests/smoke_local.py

# 4. Verify the theory. Self-contained (NumPy only); proves Propositions 1-3
#    and Corollary 1 by assertion on synthetic data.
PYTHONPATH="$PWD" python -m streamdet.metrics    # must print ALL SMOKE TESTS PASSED
```

## Data

We do not redistribute source clips. Bring your own:

* **GenVidBench** — the matched comparison cell (7 generators + 2 real sources).
* **AIGVDBench** — the cross-dataset replication (E6).

Both are prepared through the third-party toolkit's canonical H.264 re-encode (`crf 23`, closed GOP
16), which normalizes the codec fingerprints the motion-vector features read; it also emits the
matched subset CSV and the clip-level feature table that E1 reproduces.

```bash
export DATA_DIR=/path/to/data                # contains GenVidBench/, aigvdbench/
export VIDAUDIT_RESULTS=/path/to/vidaudit/results   # subset CSVs + feature tables
export REPO_ROOT=$PWD
```

Every stage writes one CSV to `$OUT`, so each is `OUT=results/<stage>/sh_0000.csv bash scripts/<group>/<stage>.sh`.
Array stages shard by clip: set `NUM_SHARDS` and `TASK_ID` (0-based) and run one process per shard.

## Reproducing the paper

Stages are ordered; later ones consume earlier outputs from `results/<stage>/`.

| Experiment | Stage scripts (`scripts/<group>/`) | Produces |
|---|---|---|
| **E1** positive control | `e1_control/`: `e1_readout` ; `e1_refeat` → `e1_refeat_readout` | LOGO AUC reproduces the published codec-MV number |
| **E2** latency–accuracy | `e2_latency_accuracy/`: `gopfeat_27k` → `stream_scores` ; `pixel_27k` → `pixel_scores` ; `bench_latency` | anytime AUC(t) per system; per-chunk ms and MACs |
| **E3/E4** compute–accuracy | `e3e4_compute_accuracy/`: `cascade_pixel` ; `manifest_vlm` → `vlm_scores` → `vlm_analysis` ; `deferral_rule` | deferral sweep, the frontier and its knee, pixel vs VLM escalation |
| **E5** earliness | `e5_earliness/`: `manifest_longform` → `gopfeat_longform` → `stream_scores_longform` | length-matched anytime curve, stopping-time FPR |
| **E6** cross-dataset | `e6_cross_dataset/`: `manifest_aigvd` → `gopfeat_aigvd` → `stream_scores_aigvd` | AIGVDBench replication |
| **E7** ablations | `e7_ablations/`: `gopfeat_27k_gop{8,32}` → `stream_scores_gop{8,32}` ; `ablate_featureset` ; `motionbias` | chunk size, feature set, motion-bias control |
| **Baselines + statistics** | `e8_baselines/`: `restrav_feat` → `restrav_readout` ; `vv_scores` ; `significance` | ReStraV and VideoVeritas as full-prefix points; CIs and paired tests (McNemar) on every decision-accuracy operating point |

`common/survey_cell.sh` is a cheap read-only census of a cell. Once the stages have run:

```bash
PYTHONPATH="$PWD" python -m streamdet.analysis.paper_numbers --results results
```

## Notes and caveats

* **Motion vectors and decoding.** The compressed-domain feature is the codec's motion field, and
  the scorer runs no pixel-domain forward pass. Obtaining the field is a bitstream parse in
  principle; this reference implementation reads it through the decoder's `export_mvs` side data,
  which pays a full frame decode. We therefore report the ~10<sup>5</sup>-MAC parse-and-score as the
  compute the stage *adds*, and treat the decode as a shared cost. The compute axis, not wall-clock,
  is where the advantage is unambiguous.
* **Abstain protocol.** A clip whose windows are all low-motion can yield no scorable chunk. Those
  clips abstain (floor score) rather than being dropped, so the evaluated population stays the
  audited cell.
* **Frame-count leakage.** The 13-d feature is pinned to a fixed number of motion frames
  (`--target-frames`); without it, clip length leaks into the readout and inflates AUC.
* **Chunks vs GOPs.** The arrival unit is a fixed 16-frame window, which coincides with an encoder
  GOP only under the canonical re-encode; in the wild a clip may carry a single I-frame.
* **Seeds.** Principal numbers are leave-one-generator-out means at seed 42 with percentile
  bootstrap intervals.

## License

MIT (see `LICENSE`). The escalation checkpoints carry their own licenses: CLIP ViT-B/32 (OpenAI) and
Ivy-xDetector (Qwen2.5-VL-3B base, Qwen Research License, non-commercial). The third-party toolkits
above are licensed by their respective authors.
