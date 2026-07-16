# Detect-as-It-Streams

Reference implementation for **"Detect-as-It-Streams: Anytime, Compute-Adaptive Detection of
AI-Generated Video from the Compressed Bitstream."**

Detectors for AI-generated video (AIGV) are evaluated almost entirely offline: decode a whole clip
to pixels, score it once, increasingly with a large VLM. Deployment is the opposite — video arrives
as a stream, under a latency budget, at a scale where running a heavy detector on every clip is
uneconomical. This repo recasts AIGV detection as a **streaming** problem:

* an **anytime** front-end that scores the codec motion-vector field per group-of-pictures
  (CPU-only, ~10<sup>5</sup> MACs/chunk, no pixel-domain forward pass) and exits as soon as it is
  confident, under a single end-calibrated threshold that is *anytime-valid*;
* **reasoning on demand** — only near-boundary segments escalate to an expensive pixel CNN or a
  small VLM, tracing a measured compute–accuracy frontier.

This code produces every number in the paper (E1–E7).

---

## Repository layout

```
streamdet/                  the implementation (each module is a runnable stage)
  streamdet_metrics.py      NumPy-only streaming metrics + a synthetic proof-by-assertion
                            smoke test for Propositions 1-3 and Corollary 1
  gop_features.py           codec motion vectors -> per-chunk 13-d features (stage 1)
  streaming_scores.py       leave-one-generator-out per-chunk scoring + abstain protocol
  analyze_streaming.py      anytime AUC(t), gate calibration, stopping-time FPR, latency
  analyze_cascade.py        deferral sweep, E[C] accounting, deferral gain (Prop. 3 / Cor. 1)
  analyze_deferral_rule.py  learned deferral rule vs the confidence band
  analyze_motionbias.py     motion-bias control (matched + within-bin AUC)
  pixel_chunk_features.py   stage-2 pixel arm: per-chunk CLIP ViT-B/32 embeddings
  vlm_scores.py             stage-2 VLM arm: Ivy-xDetector (Qwen2.5-VL-3B) p(generated)
  vlm_analysis.py           VLM baseline + codec->VLM cascade
  bench_latency.py          per-chunk wall-clock and MACs, codec vs pixel
  e1_readout.py             audited LOGO/RvR readout (the E1 positive control)
  clip_features_from_npy.py clip-level 13-d features from precomputed MV arrays
  survey_cell.py            per-clip frame/GOP census of a cell
  manifest_*.py             cell manifests (long-form, video-tree, balanced subsample)
  paper_numbers.py          recompute every number the paper cites
  smoke_local.py            end-to-end local test on synthesized clips (no cluster, ~1 min)
scripts/                    one shell script per experiment stage (see Reproducing)
environment.yml             conda environment
```

## Setup

**Prerequisites:** [conda](https://docs.conda.io/) (Miniforge/Miniconda) and `git`. A GPU is needed
only for the stage-2 arms (pixel CNN, VLM); stage 1 is CPU-only.

```bash
# 1. Clone and create the environment
git clone <this-repo> streamdet && cd streamdet
conda env create -f environment.yml
conda activate streamdet

# 2. The codec-MV feature extractor, the audited LOGO protocol, and the detector
#    zoo (CLIP, Ivy-xDetector) come from the VidAudit toolkit. Clone it and point
#    VIDAUDIT_PATH at the checkout (or install it on your PYTHONPATH).
git clone https://github.com/KurbanIntelligenceLab/vidaudit.git ../vidaudit
export VIDAUDIT_PATH=$(cd ../vidaudit && pwd)

# 3. Verify the install end-to-end (synthesizes clips, runs the full stage-1
#    pipeline, asserts the plumbing). Takes about a minute on CPU.
PYTHONPATH="$PWD" python streamdet/smoke_local.py

# 4. Verify the theory. This is self-contained (NumPy only) and proves
#    Propositions 1-3 and Corollary 1 by assertion on synthetic data.
python streamdet/streamdet_metrics.py    # must print ALL SMOKE TESTS PASSED
```

`ffmpeg` is pulled in by the environment and is required for the canonical re-encode path.

## Data

We do not redistribute source clips. Bring your own:

* **GenVidBench** — the matched comparison cell (7 generators + 2 real sources).
* **AIGVDBench** — the cross-dataset replication (E6).

Both are prepared through VidAudit's canonical H.264 re-encode (`crf 23`, closed GOP 16), which
normalizes the codec fingerprints the motion-vector features read. VidAudit also emits the matched
subset CSV and the clip-level feature table that E1 reproduces.

Point the scripts at your data with two environment variables:

```bash
export DATA_DIR=/path/to/data              # contains aigvdbench/, GenVidBench/
export VIDAUDIT_RESULTS=/path/to/vidaudit/results   # subset CSVs + feature tables
```

Every stage writes one CSV to `$OUT`, so each script is `OUT=… bash scripts/<stage>.sh`.

## Reproducing the paper

Stages are ordered; later ones consume earlier outputs from `results/<stage>/`.

| Experiment | Stage scripts | What it produces |
|---|---|---|
| **E1** positive control | `e1_readout.sh`; `e1_refeat.sh` → `e1_refeat_readout.sh` | LOGO AUC reproduces the published codec-MV number |
| **E2** latency–accuracy | `gopfeat_27k.sh` → `stream_scores.sh`; `pixel_27k.sh` → `pixel_scores.sh`; `bench_latency.sh` | anytime AUC(t) per system; per-chunk ms and MACs |
| **E3/E4** compute–accuracy | `cascade_pixel.sh`; `manifest_vlm.sh` → `vlm_scores.sh` → `vlm_analysis.sh`; `deferral_rule.sh` | deferral sweep, the frontier and its knee, pixel vs VLM escalation |
| **E5** earliness | `manifest_longform.sh` → `gopfeat_longform.sh` → `stream_scores_longform.sh` | length-matched anytime curve, stopping-time FPR |
| **E6** cross-dataset | `manifest_aigvd.sh` → `gopfeat_aigvd.sh` → `stream_scores_aigvd.sh` | AIGVDBench replication |
| **E7** ablations | `gopfeat_27k_gop{8,32}.sh` → `stream_scores_gop{8,32}.sh`; `ablate_featureset.sh`; `motionbias.sh`; `deferral_rule.sh` | chunk size, feature set, motion-bias control, gate/deferral rule |

Array stages shard by clip: set `NUM_SHARDS` and `TASK_ID` (0-based) and run one process per shard,
e.g. under a scheduler. `survey_cell.sh` is a cheap read-only census of a cell.

Once the stages have run, recompute every number the paper cites in one pass:

```bash
PYTHONPATH="$PWD" python streamdet/paper_numbers.py
```

## Notes and caveats

* **Motion vectors and decoding.** The compressed-domain feature is the codec's motion field, and
  the scorer runs no pixel-domain forward pass. Obtaining the field is a bitstream parse in
  principle; this reference implementation reads it through the decoder's `export_mvs` side data,
  which pays a full frame decode. We therefore report the ~10<sup>5</sup>-MAC parse-and-score as the
  compute the stage *adds*, and treat the decode as a shared cost (a platform already decodes for
  playback). The compute axis, not wall-clock, is where the advantage is unambiguous.
* **Abstain protocol.** A clip whose windows are all low-motion can yield no scorable chunk. Those
  clips abstain (floor score) rather than being dropped, so the evaluated population stays the
  audited cell.
* **Frame-count leakage.** The 13-d feature is pinned to a fixed number of motion frames
  (`--target-frames`); without it, clip length leaks into the readout and inflates AUC.
* **Seeds.** Principal numbers are leave-one-generator-out means at seed 42 with percentile
  bootstrap intervals.

## License

MIT. The escalation checkpoints carry their own licenses: CLIP ViT-B/32 (OpenAI) and Ivy-xDetector
(Qwen2.5-VL-3B base, Qwen Research License, non-commercial).
