#!/usr/bin/env python3
"""
streamdet_metrics.py
--------------------
Reference metrics for STREAMING, compute-adaptive AI-generated-video (AIGV)
detection (the "Detect-as-It-Streams" paper). NumPy only.

Implements the paper's quantities:
  * roc_auc, recall_at_fpr                          (standard)
  * running_max / anytime_tpr_fpr / anytime_auc     (anytime front-end)
  * streaming_auc_at_budget                          (latency-budgeted sAUC)
  * expected_compute / deferral_rate                 (cascade cost)
  * cascade_decision / cascade_error                 (reasoning on demand)
  * compute_accuracy_frontier                        (Corollary 1 sweep)
  * bootstrap_ci

The smoke test at the bottom is SYNTHETIC and proves, by assertion, the
theory claims in the paper:
  Proposition 1 (anytime monotonicity): under the disjunctive running-max
      aggregate with a fixed threshold, TPR(t) and FPR(t) are non-decreasing
      in the observed prefix t.
  Proposition 2 (anytime-valid false-positive control): a SINGLE threshold
      calibrated to the (1-alpha) quantile of the final-prefix score M_N holds
      the false-positive rate at the data-dependent stopping time AND at every
      prefix (FPR <= alpha); naive per-prefix recalibration controls the
      per-prefix rate but INFLATES the stopping-time FPR (multiple looks).
  Proposition 3 (cascade compute + exact accuracy decomposition):
      E[C] = tbar*C1 + Pr(s in W)*C2, and
      err_casc = err_1 - (err_1^W - err_2^W) exactly, so cascade error <=
      stage-1 error iff stage 2 is at least as accurate on the deferred set.
  Corollary 1: choosing the window so Pr(s in W) = (B - tbar*C1)/C2 meets the
      expected-compute budget B; the frontier is monotone under the region-wise
      deferral condition (verified), not unconditionally (negative control).

These are synthetic checks of the math/implementation, NOT empirical results.
Run:  python3 streamdet_metrics.py
"""

import numpy as np

# ----------------------------------------------------------------------
# standard detection metrics (numpy only)
# ----------------------------------------------------------------------

def _avg_ranks(x):
    """Average ranks (1-based), ties averaged. NumPy-only."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    n = len(x)
    pos = np.arange(1, n + 1, dtype=float)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = pos[i:j + 1].mean()
        i = j + 1
    return ranks


def roc_auc(scores, labels):
    """AUC = P(score(pos) > score(neg)) via the Mann-Whitney U statistic."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(scores)
    sum_pos = ranks[labels == 1].sum()
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def recall_at_fpr(scores, labels, target_fpr=0.05):
    """Max recall (TPR) achievable at FPR <= target_fpr, and the threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    # threshold = (1 - target_fpr) quantile of negative scores
    tau = np.quantile(neg, 1.0 - target_fpr)
    tpr = float((pos >= tau).mean())
    return tpr, float(tau)


# ----------------------------------------------------------------------
# anytime front-end (disjunctive running-max aggregate)
# ----------------------------------------------------------------------

def running_max(perchunk_scores):
    """
    perchunk_scores: (N_clips, T) per-GOP scores in stream order.
    returns M: (N_clips, T) where M[:, t] = max_{i<=t} score[:, i].
    """
    s = np.asarray(perchunk_scores, dtype=float)
    return np.maximum.accumulate(s, axis=1)


def anytime_tpr_fpr(perchunk_scores, labels, tau):
    """
    Returns (tpr_t, fpr_t), each length T, using the disjunctive rule
    flag-generated iff M_t >= tau. This is the object of Proposition 1.
    """
    M = running_max(perchunk_scores)
    labels = np.asarray(labels).astype(int)
    flag = (M >= tau)                      # (N, T) boolean
    tpr_t = flag[labels == 1].mean(axis=0)
    fpr_t = flag[labels == 0].mean(axis=0)
    return tpr_t, fpr_t


def anytime_auc_curve(perchunk_scores, labels):
    """AUC of the running-max score M_t at each prefix t -> length-T array."""
    M = running_max(perchunk_scores)
    T = M.shape[1]
    return np.array([roc_auc(M[:, t], labels) for t in range(T)])


def streaming_auc_at_budget(perchunk_scores, labels, budget_prefix):
    """Latency-budgeted sAUC: AUC of M_t at the largest prefix within budget."""
    M = running_max(perchunk_scores)
    t = min(int(budget_prefix), M.shape[1]) - 1
    t = max(t, 0)
    return roc_auc(M[:, t], labels)


# ----------------------------------------------------------------------
# anytime-valid false-positive control (Proposition 2)
# ----------------------------------------------------------------------

def anytime_valid_threshold(perchunk_scores, labels, alpha=0.05):
    """
    Single end-calibrated threshold tau_alpha = (1-alpha) quantile of the
    FINAL-prefix running-max score M_N under the real (null) distribution.
    By Proposition 2, thresholding the running max at this tau controls the
    false-positive rate at the data-dependent stopping time and at every prefix.
    """
    M = running_max(perchunk_scores)
    labels = np.asarray(labels).astype(int)
    M_last_real = M[labels == 0, -1]
    return float(np.quantile(M_last_real, 1.0 - alpha))


def stopping_time_fpr(perchunk_scores, labels, tau):
    """
    FPR at the alarm time of the running-max detector with a FIXED threshold
    tau: a real clip is a false positive iff M_t >= tau for some t<=N, which
    (by monotonicity) equals M_N >= tau. Returns that probability over reals.
    """
    M = running_max(perchunk_scores)
    labels = np.asarray(labels).astype(int)
    ever = (M[labels == 0] >= tau).any(axis=1)
    return float(ever.mean())


def stopping_time_fpr_perprefix(perchunk_scores, labels, alpha=0.05):
    """
    Naive per-prefix recalibration: tau_t = (1-alpha) quantile of M_t under the
    null at EACH prefix t. A real clip is a false positive iff M_t >= tau_t for
    some t (first-crossing alarm). Returns the resulting stopping-time FPR,
    which Proposition 2(ii) shows is >= alpha (multiple-looks inflation).
    """
    M = running_max(perchunk_scores)
    labels = np.asarray(labels).astype(int)
    real = M[labels == 0]
    taus = np.quantile(real, 1.0 - alpha, axis=0)   # length-T per-prefix thresholds
    ever = (real >= taus[None, :]).any(axis=1)
    return float(ever.mean()), taus


# ----------------------------------------------------------------------
# reasoning-on-demand cascade (compute + accuracy)
# ----------------------------------------------------------------------

def deferral_rate(stage1_scores, window):
    """Fraction of inputs whose stage-1 score lands in the deferral window."""
    s = np.asarray(stage1_scores, dtype=float)
    a, b = window
    return float(((s >= a) & (s <= b)).mean())


def expected_compute(C1, C2, tbar, p_defer):
    """E[C] = tbar*C1 + p_defer*C2  (Proposition 2)."""
    return float(tbar * C1 + p_defer * C2)


def cascade_decision(stage1_scores, tau, window, stage2_pred):
    """
    Confident region: decide by stage-1 threshold tau.
    Deferred region (stage-1 score in window): use stage-2 prediction.
    Returns final binary predictions.
    """
    s = np.asarray(stage1_scores, dtype=float)
    a, b = window
    deferred = (s >= a) & (s <= b)
    pred = (s >= tau).astype(int)               # stage-1 decision everywhere
    pred[deferred] = np.asarray(stage2_pred, dtype=int)[deferred]
    return pred, deferred


def error_rate(pred, labels):
    pred = np.asarray(pred).astype(int)
    labels = np.asarray(labels).astype(int)
    return float((pred != labels).mean())


def error_mass(pred, labels, region_mask):
    """Error MASS on a region: P(wrong AND in region) = mean of (pred!=label)&mask."""
    pred = np.asarray(pred).astype(int)
    labels = np.asarray(labels).astype(int)
    region_mask = np.asarray(region_mask).astype(bool)
    return float(((pred != labels) & region_mask).mean())


def cascade_error(stage1_scores, tau, window, stage2_pred, labels):
    pred, _ = cascade_decision(stage1_scores, tau, window, stage2_pred)
    return error_rate(pred, labels)


def compute_accuracy_frontier(stage1_scores, tau, stage2_pred, labels,
                              C1, C2, tbar, half_widths):
    """
    Sweep symmetric deferral windows [tau-w, tau+w] over half_widths.
    Returns list of (p_defer, E[C], accuracy).
    """
    out = []
    for w in half_widths:
        window = (tau - w, tau + w)
        p = deferral_rate(stage1_scores, window)
        ec = expected_compute(C1, C2, tbar, p)
        err = cascade_error(stage1_scores, tau, window, stage2_pred, labels)
        out.append((float(w), p, ec, 1.0 - err))
    return out


def window_for_budget(B, C1, C2, tbar):
    """Corollary 1: target deferral probability to meet expected-compute B."""
    p = (B - tbar * C1) / C2
    return float(np.clip(p, 0.0, 1.0))


# ----------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------

def bootstrap_ci(metric_fn, scores, labels, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(scores[idx], labels[idx]))
    vals = np.array(vals, dtype=float)
    lo = float(np.nanpercentile(vals, 100 * alpha / 2))
    hi = float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))
    return float(np.nanmean(vals)), lo, hi


# ----------------------------------------------------------------------
# SYNTHETIC smoke test: proves Prop 1, Prop 2, Corollary 1 by assertion
# ----------------------------------------------------------------------

def _make_streaming_data(seed=0, n=4000, T=12):
    """
    Synthetic per-GOP scores. Generated clips accumulate evidence: their
    per-chunk scores trend upward; real clips stay low. Labels balanced.
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)]).astype(int)
    scores = np.empty((n, T), dtype=float)
    for k in range(n):
        if labels[k] == 1:                       # generated: rising mean
            mu = np.linspace(-0.6, 1.4, T)
            scores[k] = rng.normal(mu, 0.8)
        else:                                    # real: flat low mean
            scores[k] = rng.normal(-0.5, 0.8, T)
    return scores, labels


def _smoke():
    print("=" * 64)
    print("streamdet_metrics smoke test (SYNTHETIC; proves the theory)")
    print("=" * 64)
    scores, labels = _make_streaming_data()

    # ---- sanity: full-prefix AUC is meaningful, with bootstrap CI ----
    M = running_max(scores)
    auc_full = roc_auc(M[:, -1], labels)
    mean, lo, hi = bootstrap_ci(roc_auc, M[:, -1], labels, n_boot=300, seed=1)
    print(f"[sanity] full-prefix anytime AUC = {auc_full:.3f} "
          f"(bootstrap 95% CI [{lo:.3f}, {hi:.3f}])")
    assert 0.5 < auc_full <= 1.0

    # ---- Proposition 1: TPR(t) and FPR(t) non-decreasing in prefix ----
    tau = 0.5
    tpr_t, fpr_t = anytime_tpr_fpr(scores, labels, tau)
    d_tpr = np.diff(tpr_t)
    d_fpr = np.diff(fpr_t)
    print(f"[Prop 1] TPR(t): {tpr_t[0]:.3f} -> {tpr_t[-1]:.3f} ; "
          f"FPR(t): {fpr_t[0]:.3f} -> {fpr_t[-1]:.3f}")
    assert np.all(d_tpr >= -1e-12), "TPR must be non-decreasing in prefix"
    assert np.all(d_fpr >= -1e-12), "FPR must be non-decreasing in prefix"
    auc_t = anytime_auc_curve(scores, labels)
    print(f"[Prop 1] anytime AUC(t): {auc_t[0]:.3f} -> {auc_t[-1]:.3f} "
          f"(rises as more stream arrives)")
    print("[Prop 1] PASS: running-max aggregate is monotone in the prefix; "
          "fixed-tau FPR also rises -> the worst prefix is the last one.")

    # ---- Proposition 2: anytime-valid false-positive control ----
    alpha = 0.05
    tau_av = anytime_valid_threshold(scores, labels, alpha=alpha)
    # (i) single end-calibrated threshold: stopping-time FPR and per-prefix FPR
    st_fpr = stopping_time_fpr(scores, labels, tau_av)
    _, fpr_t_av = anytime_tpr_fpr(scores, labels, tau_av)
    print(f"[Prop 2] end-calibrated tau={tau_av:.3f} (alpha={alpha}): "
          f"stopping-time FPR={st_fpr:.3f}; max per-prefix FPR={fpr_t_av.max():.3f}")
    assert st_fpr <= alpha + 1e-9, "end-calibrated stopping-time FPR must be <= alpha"
    assert fpr_t_av.max() <= alpha + 1e-9, "per-prefix FPR must be <= alpha for all t"
    # (ii) naive per-prefix recalibration inflates the stopping-time FPR
    st_fpr_naive, _ = stopping_time_fpr_perprefix(scores, labels, alpha=alpha)
    print(f"[Prop 2] naive per-prefix recalibration: stopping-time "
          f"FPR={st_fpr_naive:.3f} (>= alpha; multiple-looks inflation)")
    assert st_fpr_naive >= alpha - 1e-9, "per-prefix recalibration should not be below alpha"
    assert st_fpr_naive >= st_fpr - 1e-9, "per-prefix recalibration must not beat end-calibration"
    print("[Prop 2] PASS: monotonicity makes a single end-calibrated threshold "
          "anytime-valid; per-prefix recalibration inflates the alarm-time FPR.")

    # ---- Proposition 3: expected compute formula ----
    C1, C2 = 1.0, 200.0          # stage-1 per-GOP cost << one stage-2 call
    tbar = float(scores.shape[1])  # worst case: all GOPs processed
    stage1_scores = M[:, -1]     # stage-1 score = final running-max aggregate
    tau1 = tau_av                # decision threshold, calibrated as in Prop 2
    w = 0.8                      # one-sided deferral width: W = [tau - w, tau)
    window = (tau1 - w, tau1)    # band just below the threshold
    p_def = deferral_rate(stage1_scores, window)
    ec_formula = expected_compute(C1, C2, tbar, p_def)
    # Monte-Carlo compute: tbar*C1 always, +C2 on deferred fraction
    a, b = window
    deferred = (stage1_scores >= a) & (stage1_scores <= b)
    ec_mc = float(tbar * C1 + deferred.mean() * C2)
    print(f"[Prop 3] deferral rate Pr(s in W) = {p_def:.3f} ; "
          f"E[C] formula = {ec_formula:.2f} ; Monte-Carlo = {ec_mc:.2f}")
    assert abs(ec_formula - ec_mc) < 1e-9, "compute formula must match"
    print("[Prop 3] PASS: E[C] = tbar*C1 + Pr(s in W)*C2 holds exactly.")

    # ---- Proposition 3: exact error decomposition + deferral guarantee ----
    # stage-1 prediction everywhere; stage-2 is MORE accurate on deferred set.
    stage1_pred = (stage1_scores >= tau1).astype(int)
    err1 = error_rate(stage1_pred, labels)
    rng = np.random.default_rng(7)
    stage2_pred = labels.copy()
    flip = rng.random(len(labels)) < 0.05        # 5% noise -> still better
    stage2_pred[flip] = 1 - stage2_pred[flip]
    err_casc = cascade_error(stage1_scores, tau1, window, stage2_pred, labels)
    # EXACT decomposition uses error MASSES on the deferred region:
    err1_W_mass = error_mass(stage1_pred, labels, deferred)
    err2_W_mass = error_mass(stage2_pred, labels, deferred)
    decomp = err1 - (err1_W_mass - err2_W_mass)
    print(f"[Prop 3] deferred-set error mass: stage-1 = {err1_W_mass:.4f}, "
          f"stage-2 = {err2_W_mass:.4f}")
    print(f"[Prop 3] overall stage-1 err = {err1:.4f}, cascade err = {err_casc:.4f}, "
          f"err1-(err1^W-err2^W) = {decomp:.4f}")
    assert abs(err_casc - decomp) < 1e-12, \
        "exact decomposition err_casc = err1 - (err1^W - err2^W) must hold"
    assert err_casc <= err1 + 1e-12, "cascade error must be <= stage-1 error"
    print("[Prop 3] PASS: exact decomposition holds and (stage 2 better on "
          "deferred) => cascade error <= stage-1 error.")

    # ---- Corollary 1: window hits a compute budget ----
    B = tbar * C1 + 0.10 * C2                    # budget allowing ~10% deferral
    p_target = window_for_budget(B, C1, C2, tbar)
    # find symmetric window whose deferral rate ~ p_target
    widths = np.linspace(0.0, 3.0, 600)
    best = min(widths, key=lambda w: abs(
        deferral_rate(stage1_scores, (tau1 - w, tau1 + w)) - p_target))
    p_got = deferral_rate(stage1_scores, (tau1 - best, tau1 + best))
    ec_got = expected_compute(C1, C2, tbar, p_got)
    print(f"[Cor 1] target Pr(defer)={p_target:.3f} -> window half-width "
          f"{best:.3f} gives Pr(defer)={p_got:.3f}, E[C]={ec_got:.2f} "
          f"(budget B={B:.2f})")
    assert abs(ec_got - B) < 0.05 * C2, "selected window should meet budget"
    print("[Cor 1] PASS: a compute budget selects a deferral window.")

    # ---- frontier is monotone ONLY when stage 2 dominates stage 1 on every
    #      escalated region. With a globally-noisy stage 2, widening the window
    #      eventually escalates points far from the boundary where stage 1 is
    #      already near-perfect, which can REDUCE accuracy. We therefore verify
    #      the monotonicity in the regime where Corollary 1's condition holds
    #      (here: an oracle stage 2 on the deferred set). This is exactly the
    #      conditional claim in the paper, not an unconditional one.
    stage2_oracle = labels.copy()
    frontier = compute_accuracy_frontier(
        stage1_scores, tau1, stage2_oracle, labels, C1, C2, tbar,
        half_widths=np.linspace(0.0, 2.5, 12))
    accs = [acc for (_, _, _, acc) in frontier]
    assert all(accs[i + 1] >= accs[i] - 1e-9 for i in range(len(accs) - 1)), \
        "under the deferral condition, accuracy is non-decreasing as W widens"
    print("[Cor 1] PASS: when stage 2 dominates on the escalated region, the "
          "compute-accuracy frontier is monotone in the budget.")

    # ---- negative control: a globally-noisy stage 2 need NOT give a monotone
    #      frontier, confirming the condition is necessary (informational only).
    accs_noisy = [acc for (_, _, _, acc) in compute_accuracy_frontier(
        stage1_scores, tau1, stage2_pred, labels, C1, C2, tbar,
        half_widths=np.linspace(0.0, 2.5, 12))]
    monotone_noisy = all(accs_noisy[i + 1] >= accs_noisy[i] - 1e-9
                         for i in range(len(accs_noisy) - 1))
    print(f"[Cor 1] negative control: globally-noisy stage 2 monotone? "
          f"{monotone_noisy} (expected False -> condition is necessary).")

    print("=" * 64)
    print("ALL SMOKE TESTS PASSED")
    print("These are synthetic checks of the math, not empirical results.")
    print("=" * 64)


if __name__ == "__main__":
    _smoke()
