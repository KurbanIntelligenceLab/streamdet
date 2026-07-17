# Figure tooling

Everything in the paper's Figure 1 is real data; these scripts reproduce it:

1. `fig1_pick.py` / `fig1_write_spec.py` — choose exemplar clips from the
   stream-scores logs (an early-exiting generated clip, a low real clip, a
   deferred clip) and record their running-max trajectories + the fold's tau.
2. `fig1_data.py` — cluster/offline stage: extract GOP-aligned frames (JPEG)
   and the per-macroblock codec motion-vector field (npz) for the chosen clips
   via PyAV's side-data export.
3. `fig1_portrait.py` — compose the single-column teaser (film strips, the
   generated-vs-real motion-vector quiver pair, the running-score chart with
   the calibrated threshold and deferral band). `fig1_compose.py` is the
   full-width variant of the same design.
