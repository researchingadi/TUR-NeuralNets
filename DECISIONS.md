# DECISIONS.md — TUR-NeuralNets Project Decision Log

**Project:** Thermodynamic Uncertainty Relations in Neural Network Training  
**Owner:** Adi Singh (`researchingadi`)  
**Repo:** `researchingadi/tur-neuralnets`  
**Log started:** June 2026  

---

> This document records every significant decision made in this project — architectural, theoretical, experimental, and implementation-level. No decision is too small. Every entry must include what was chosen, what was rejected, why, and what it affects downstream. Future contributors (and future Adi) should be able to reconstruct the entire reasoning chain of this project from this file alone.

---

## [2026-06-14] — Core Physical Framing: SGD as a Langevin Process

**Decision:** Model stochastic gradient descent as a continuous-time Langevin equation:
```
dθ_t = −∇_θ L(θ_t) dt + √(2T) dW_t
```
where `T = η σ² / 2` is the effective temperature, η is the learning rate, and σ² is gradient variance. This is the foundational mapping that makes stochastic thermodynamics applicable.

**Alternatives considered:**
- Treating SGD as a discrete Markov chain without continuous-time embedding (loses contact with Langevin thermodynamics literature)
- Using the Fokker-Planck formulation directly (mathematically equivalent but harder to connect to per-step computations)
- Framing in terms of free energy / Helmholtz decomposition (more general but requires estimating non-gradient force components)

**Reasoning:** The Langevin mapping is the standard bridge between SGD and stochastic thermodynamics, established rigorously by Mandt, Hoffman & Blei (2017, JMLR). It gives us direct access to the TUR literature (Barato & Seifert 2015; Horowitz & Gingrich 2020) without additional theoretical machinery. Every subsequent physical interpretation in this paper depends on this mapping.

**Impact:** Determines the form of the entropy production estimator (gradient-norm based), the choice of TUR formulation, the noise injection model for the toy experiment, and the entire Section 3 (Framework) of the paper.

---

## [2026-06-14] — Primary Observable Current: Loss-Decrease Current

**Decision:** Define the primary thermodynamic current J as the loss-decrease rate:
```
J_t = (L(θ_{t−1}) − L(θ_t)) / η
```
This is the current used to compute both sides of the TUR inequality in all experiments.

**Alternatives considered:**
- Weight update magnitude: `J_t = ‖Δθ_t‖` (physically natural as flow in weight space, but dimensionally inconsistent with loss-space interpretation)
- Gradient norm: `J_t = ‖∇L(θ_t)‖` (the driving force, not a current — conflates cause and effect)
- Generalization current: rate of test loss decrease (most scientifically interesting but requires per-step test evaluation, expensive and noisy)
- All three in parallel (done in toy experiment as a secondary validation; see W=200 decision below)

**Reasoning:** The loss-decrease current is (1) directly computable at every training step, (2) physically interpretable as work extracted from the loss landscape, (3) the most natural analog of the chemical currents in Barato & Seifert (2015), and (4) dimensionally consistent with the entropy production estimator in units of [loss / step]. The generalization current is the scientifically central quantity for the paper's claim but is introduced as a secondary current computed at evaluation checkpoints (not every step).

**Impact:** Drives the TUR ratio formula implemented in `src/metrics/tur_ratio.py`. Sets the primary axis of all TUR-diagnostic figures. The choice of J must be declared explicitly in every figure caption to be defensible to reviewers.

---

## [2026-06-14] — Gradient Variance Estimator: Both Ground Truth and Empirical (Sliding Window)

**Decision:** Implement two estimators for σ² (gradient variance) and run them in parallel throughout the toy experiment:
1. **Ground truth σ²**: Known analytically (or set exactly) from the Langevin noise injection magnitude — `σ² = 2T` by construction
2. **Empirical sliding window**: Running variance of gradient norms over the last W steps
3. **Multi-batch (reserved for neural net experiments)**: Variance across K mini-batches sampled at the same θ_t

**Alternatives considered:**
- Sliding window only (cheap but no ground truth to validate against)
- Fixed estimate from initialization (pathological — gradient variance changes dramatically over training)
- Multi-batch only (expensive: requires K extra forward passes per step; impractical during toy verification)

**Reasoning:** The ground truth σ² gives us a calibration anchor in the toy experiment where we control the noise exactly. The empirical sliding window is the method that will actually be used in the neural network experiments (where true σ² is unknown). Comparing them in the toy lets us quantify the estimator bias before we rely on it in high-stakes experiments. This is the "both" choice that validates the cheaper method against the exact one. Multi-batch estimation is architecturally reserved for Experiment 3 (optimizer comparison) where per-step cost is more acceptable.

**Impact:** Requires `src/metrics/entropy_production.py` to expose two code paths. The comparison panel (σ²_true vs σ²_empirical entropy production) becomes a validation figure in the toy experiment. All neural net experiments use sliding window by default with multi-batch as an optional config flag.

---

## [2026-06-14] — Window Size W: W=200 Primary, Overlay at [50, 200, 500]

**Decision:** The TUR ratio computation uses a sliding window of W steps to estimate `Var(J)` and `⟨J⟩`. Primary window size: **W = 200**. All figures show overlaid curves for W ∈ {50, 200, 500} with W=200 as the thicker/primary line.

**Alternatives considered:**
- W = 50 only (too noisy for the TUR ratio to be readable; TUR ratio fluctuates wildly)
- W = 500 only (smooths over the transient physics in early training; masks the initial non-stationarity)
- Adaptive W based on local stationarity (theoretically principled but introduces another free parameter)
- W fixed as a fraction of total steps (e.g. 5%) — equivalent to W=500 for a 10,000-step run

**Reasoning:** The W-sensitivity of the TUR ratio is itself scientifically interesting — it reveals the timescale over which the system approaches quasi-stationarity. Showing all three window sizes explicitly (1) demonstrates robustness of the W=200 result, (2) provides visual evidence of the transient-to-stationary transition, and (3) preempts the obvious reviewer question "does your result depend on the window size?" The answer becomes: "No — see Figure 1d." W=200 is chosen as primary because it balances noise suppression with temporal resolution at the step sizes used.

**Impact:** The window size must be documented in every figure caption. The `tur_ratio` function in `src/metrics/tur_ratio.py` takes `window_sizes: List[int]` as a config-driven parameter. The multi-W overlay is the default output; single-W is a subset case.

---

## [2026-06-14] — Numerical Integration: Euler-Maruyama, dt=0.01, Config-Driven

**Decision:** Integrate the Langevin SDE using the **Euler-Maruyama scheme**:
```
θ_{t+1} = θ_t − ∇L(θ_t) dt + √(2T · dt) · ξ_t,   ξ_t ~ N(0, I)
```
with `dt = 0.01` as the default timestep. `dt` is a YAML config parameter, never hardcoded.

**Alternatives considered:**
- Heun / Runge-Kutta (higher order): More accurate for stiff SDEs, but overkill for a quadratic landscape where EM is exact in law; adds implementation complexity for no physical insight gain
- dt = 0.001: Smaller timestep, lower discretization error, but 10× more steps for the same simulated time — reserved as a convergence-check config option
- dt = 0.1: Too large; Euler-Maruyama becomes unstable near the curvature of the anisotropic well

**Reasoning:** For a quadratic potential, the Langevin SDE has an exact Ornstein-Uhlenbeck solution, and Euler-Maruyama converges to it correctly in the limit dt→0. dt=0.01 is well inside the stability region for the landscape parameters we're using (a=10, b=1 in the anisotropic well). Heun would be the right choice for a double-well with a stiff barrier — flagged in code comments as `# TODO: consider Heun for double-well if dt instability observed`. The config-driven dt means any convergence test (dt=0.01 vs dt=0.001) can be run without code changes.

**Impact:** Sets the timestep everywhere in `src/langevin/simulator.py`. All entropy production estimates are computed per dt-step. The `dt` value appears in every figure's axis label when time is plotted. This decision is citable against Kloeden & Platen (numerical SDE textbook) and consistent with practice in Boffi & Vanden-Eijnden (2024).

---

## [2026-06-14] — Two-Landscape Design: Anisotropic Quadratic + Double-Well

**Decision:** The toy 2D experiment uses two loss landscapes, presented as two figure sets:
1. **Anisotropic quadratic:** `L(θ) = a·θ₁² + b·θ₂²` with `a=10, b=1` (near-stationary Ornstein-Uhlenbeck; clean analytics; primary TUR verification landscape)
2. **Double-well:** `L(θ) = (θ₁² − 1)² + θ₂²` (nonequilibrium, saddle-crossing dynamics; physically richer; tests TUR in a non-convex setting)

**Alternatives considered:**
- Isotropic quadratic only (a=b=1): Too symmetric; doesn't mimic the anisotropic curvature of real neural network loss landscapes; misses the flat-direction physics central to SGD generalization
- Double-well only: Not near-stationary; harder to verify TUR analytically; better as a second panel once the isotropic/anisotropic cases establish the baseline
- Rosenbrock or Rastrigin: Interesting but too complex for a sanity-check experiment; save for paper appendix if needed

**Reasoning:** The anisotropic quadratic landscape is where we *verify* TUR analytically and establish ground truth. It mimics the flat (θ₂) vs sharp (θ₁) curvature directions seen in real loss landscapes (connecting to Feng & Tu 2021, inverse variance-flatness). The double-well introduces barrier-crossing, which makes the system genuinely nonequilibrium and tests the finite-time TUR (Pietzonka et al. 2017) in a regime closer to what neural networks encounter when escaping sharp minima. Showing both makes the toy section paper-worthy rather than a footnote.

**Impact:** `configs/toy_langevin.yaml` has two top-level sections: `landscape.anisotropic` and `landscape.double_well`. The landscape factory in `src/langevin/landscapes.py` dispatches on a config flag. Both landscapes use identical TUR computation code — the landscape is a plugin.

---

## [2026-06-14] — TUR Framing: Near-Steady-State for Toy, Finite-Time for Neural Nets

**Decision:** Apply two distinct TUR framings depending on the experiment type:

- **Toy Langevin experiment:** Verify TUR in the **near-stationary regime** after the particle has relaxed to the vicinity of the potential minimum. This invokes the steady-state TUR (Barato & Seifert 2015) in its domain of validity. Transient initialization phase is excluded from the TUR ratio computation via a `burn_in` config parameter.

- **Neural network training experiments:** Invoke the **finite-time generalization** of Pietzonka et al. (2017), replacing instantaneous entropy production rate with cumulative entropy production over training windows. Paper methods sentence (locked):
  > *"In the toy Langevin experiment we verify TUR in the near-stationary regime following relaxation; for training dynamics we invoke the finite-time generalization of Pietzonka et al. (2017), replacing instantaneous entropy production rate with cumulative entropy production over training windows."*

**Alternatives considered:**
- Apply finite-time TUR everywhere (theoretically safer but requires more complex implementation in the toy where the simpler bound suffices)
- Apply steady-state TUR everywhere and flag transience as a limitation (scientifically weaker — a reviewer will immediately flag that training is not a steady state)
- Ignore the steady-state vs transient distinction entirely (not defensible to a Physical Review E referee)

**Reasoning:** The honest framing distinguishes the toy (where we can achieve near-stationarity by design) from the neural network case (where training is inherently transient). This demonstrates theoretical awareness and preempts the most obvious referee objection. The methods sentence is already written and cite-ready.

**Impact:** The `burn_in` parameter in `configs/toy_langevin.yaml` sets how many steps to skip before TUR ratio computation begins. All TUR ratio figures for the toy experiment are annotated with a vertical line at t=burn_in. Neural network TUR ratios are computed over cumulative windows using the Pietzonka et al. (2017) formulation.

---

## [2026-06-14] — YAML Config Structure: Fully Config-Driven, No Magic Numbers

**Decision:** Every experiment is driven by a YAML configuration file. No numerical constants appear in Python source files except as default values with explicit comments. The canonical config structure is:

```yaml
# configs/toy_langevin.yaml
experiment:
  name: "toy_langevin_anisotropic"
  seed: 42
  author: "Adi Singh"

langevin:
  dt: 0.01
  n_steps: 50000
  temperature: 0.5        # T = η·σ²/2; controls noise amplitude
  burn_in: 5000           # steps excluded from TUR computation

landscape:
  type: "anisotropic_quadratic"
  a: 10.0                 # curvature along θ₁
  b: 1.0                  # curvature along θ₂
  theta_init: [3.0, 3.0]  # starting position

noise:
  sigma2_true: null       # if null, computed as 2T from Langevin; else override
  window_sizes: [50, 200, 500]
  primary_window: 200

tur:
  current: "loss_decrease"  # options: loss_decrease, weight_update, grad_norm
  n_current_estimates: 3    # how many J definitions to compare

figures:
  dpi: 300
  format: ["pdf", "png"]
  output_dir: "figures/"
  style: "paper"            # uses src/plotting/paper_style.py rcParams
```

**Alternatives considered:**
- Argparse-only CLI (no audit trail, no reproducibility guarantee)
- Hydra (more powerful but heavy dependency for an early-stage research project)
- Hardcoded constants with a `params.py` file (no version control on parameter choices)

**Reasoning:** YAML configs are (1) human-readable and diff-able in git, (2) directly reproducible by anyone who clones the repo, (3) sufficient for this project's complexity level, and (4) the standard in serious ML research codebases. Hydra is flagged as a future upgrade if the hyperparameter sweep in Experiment 2 requires grid/random search infrastructure. Every config file is committed alongside its output figures — a figure without a config is not a result.

**Impact:** All scripts in `scripts/` take `--config path/to/config.yaml` as their only required argument. The config is logged to wandb/tensorboard at the start of every run. Config files are named to match their output figures (e.g. `toy_anisotropic_T0.5_dt0.01.yaml`).

---

## [2026-06-14] — Figure Layout: Single 4-Panel Figure as Primary Template

**Decision:** The primary output figure for the toy experiment is a **single figure with 4 panels** in a 2×2 layout:
- **Panel (a):** Loss landscape contours + particle trajectory heatmap + streamplot of estimated probability currents
- **Panel (b):** Entropy production rate `Ṡ_t` over time (both σ²_true and σ²_empirical overlaid)
- **Panel (c):** LHS vs RHS of TUR inequality over time (`Var(J)/⟨J⟩²` vs `2/Ṡ`)
- **Panel (d):** TUR ratio over time for W ∈ {50, 200, 500}, with horizontal dashed line at ratio=2

This layout is the **template for all subsequent experiment figures** in the paper.

**Alternatives considered:**
- Two separate figures (landscape + diagnostics): Cleaner separation but wastes a figure slot in a journal with tight limits
- 6-panel figure including σ² estimator comparison as its own panel: Too dense for a first figure; the σ² comparison is shown in Panel (b) instead (two overlaid lines)
- TUR ratio as the only figure (skip landscape visualization): Loses the physical intuition of showing what the particle is doing spatially

**Reasoning:** The 4-panel layout tells a complete physical story: (a) here is the system, (b) here is the cost being paid, (c) here is the inequality, (d) here is our diagnostic. A reviewer can follow the physical argument from left-to-right, top-to-bottom without leaving the figure. This layout structure — system → cost → bound → diagnostic — will be reused for all experiment figures, making the paper visually coherent.

**Impact:** `src/plotting/paper_style.py` defines the rcParams and panel layout grid. All figures use `figsize=(12, 10)` for 4-panel layouts, exported at 300 DPI as both PDF and PNG. Panel labels use uppercase bold (A, B, C, D) in the Physical Review E style.

---

*End of initialization entries. All future decisions appended below this line in chronological order.*

---
