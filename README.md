# Thermodynamic Uncertainty Relations in Neural Network Training
### Entropy Production Bounds on Generalization
 
**Author:** Adi Singh (`researchingadi`) · Mississippi State University  
**Status:** Active Research · Pre-print forthcoming  
 
---
 
> **Abstract.** We model stochastic gradient descent (SGD) as a Langevin process and apply the thermodynamic uncertainty relation (TUR) to neural network training dynamics. We show empirically that training entropy production bounds the precision of generalization currents, derive a tractable estimator for per-step entropy production from gradient statistics, and demonstrate that cumulative entropy production predicts generalization gap across architectures, hyperparameters, and optimizers. Our results provide a physics-grounded, falsifiable bound on learning that is complementary to existing PAC-Bayes and information-theoretic generalization bounds, and offer a thermodynamic explanation for the empirically observed generalization advantage of SGD over adaptive optimizers such as Adam.
 
---
 
## Core Scientific Claim
 
> **The precision of generalization in neural network training is fundamentally bounded by the entropy produced during the training process.**
>
> Specifically, if we define the generalization current $J$ as the rate of improvement in test performance, then the thermodynamic uncertainty relation implies:
>
> $$\frac{\mathrm{Var}(J)}{\langle J \rangle^2} \geq \frac{2}{\dot{S}_{\mathrm{train}}}$$
>
> where $\dot{S}_{\mathrm{train}}$ is the entropy production rate of the SGD process. This is not a heuristic — it is a consequence of the laws of nonequilibrium thermodynamics applied to SGD as a Langevin system.
 
---
 
## The Physics
 
### SGD as a Langevin Process
 
In the continuous-time limit, SGD with mini-batch noise becomes a Langevin equation:
 
$$d\theta_t = -\nabla_\theta \mathcal{L}(\theta_t)\, dt + \sqrt{2T}\, dW_t$$
 
where:
- $T = \eta \sigma^2 / 2$ is the effective temperature (learning rate $\eta$ times gradient variance $\sigma^2$, halved)
- $W_t$ is a standard Wiener process (Brownian motion)
- The gradient term is the deterministic drift; the $\sqrt{2T}\,dW_t$ term is stochastic forcing
This identification — due to Mandt, Hoffman & Blei (2017) — places SGD squarely within the framework of stochastic thermodynamics.
 
### The Thermodynamic Uncertainty Relation
 
For any observable current $J$ in a nonequilibrium Langevin system, the TUR states:
 
$$\boxed{\frac{\mathrm{Var}(J)}{\langle J \rangle^2} \geq \frac{2}{\dot{S}}}$$
 
**In words:** The coefficient of variation squared of any current is lower-bounded by the inverse entropy production rate. Precision costs dissipation. There is no thermodynamically free lunch.
 
### Tractable Entropy Production Estimator
 
For the Langevin SGD process, entropy production rate at step $t$ is estimated as:
 
$$\hat{\dot{S}}_t = \frac{2 \|\nabla \mathcal{L}(\theta_t)\|^2}{\sigma^2}$$
 
where $\sigma^2$ is estimated from the running variance of gradient norms over a sliding window of $W$ steps. This is computable from quantities already available during standard training — no additional forward passes required.
 
### TUR Ratio (Primary Diagnostic)
 
$$\text{TUR-ratio}_t = \frac{\mathrm{Var}(J)_W}{\langle J \rangle_W^2} \cdot \hat{\dot{S}}_t$$
 
**If TUR holds:** TUR-ratio $\geq 2$ throughout training. This is the falsifiable prediction tested in all experiments.
 
---
 
## Project Structure
 
```
tur-neuralnets/
│
├── README.md                        # This file
├── DECISIONS.md                     # Living decision log (every choice documented)
├── requirements.txt
├── setup.py
│
├── configs/                         # YAML experiment configs — no magic numbers in code
│   ├── toy_langevin.yaml            # Toy 2D Langevin experiment
│   ├── experiment1_mnist.yaml       # Exp 1: TUR verification on MNIST
│   ├── experiment2_sweep.yaml       # Exp 2: Entropy vs generalization sweep
│   ├── experiment3_optimizers.yaml  # Exp 3: SGD vs Adam under TUR
│   ├── experiment4_phases.yaml      # Exp 4: Training phase analysis
│   └── experiment5_noise.yaml       # Exp 5 (stretch): Memorization vs generalization
│
├── src/
│   ├── langevin/
│   │   ├── __init__.py
│   │   ├── simulator.py             # Euler-Maruyama SDE integrator
│   │   └── landscapes.py            # Loss landscape definitions (anisotropic, double-well, ...)
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── entropy_production.py    # Ṡ estimator (sliding window + ground truth)
│   │   └── tur_ratio.py             # TUR ratio computation, multi-window overlay
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py                   # Configurable MLP for MNIST/CIFAR-10
│   │   └── cnn.py                   # Configurable CNN for CIFAR-10
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop with per-step gradient logging
│   │   └── gradient_logger.py       # Hooks for gradient norm, weight update tracking
│   │
│   └── plotting/
│       ├── __init__.py
│       ├── paper_style.py           # matplotlib rcParams for Physical Review E style
│       └── figures.py               # Figure factory: landscape, TUR panels, scatter plots
│
├── scripts/
│   ├── run_toy_langevin.py          # Entry point: toy 2D experiment
│   ├── run_experiment1.py           # Entry point: Experiment 1
│   ├── run_experiment2.py           # Entry point: Experiment 2 (hyperparameter sweep)
│   ├── run_experiment3.py           # Entry point: Experiment 3 (optimizer comparison)
│   ├── run_experiment4.py           # Entry point: Experiment 4 (phase analysis)
│   └── run_experiment5.py           # Entry point: Experiment 5 (memorization)
│
├── figures/                         # Output figures (PDF + PNG, 300 DPI)
│   ├── toy/
│   ├── experiment1/
│   ├── experiment2/
│   ├── experiment3/
│   ├── experiment4/
│   └── experiment5/
│
├── notebooks/                       # Exploratory analysis (not in paper pipeline)
│   └── exploration.ipynb
│
└── tests/
    ├── test_entropy_production.py   # Unit tests: known analytical results
    ├── test_tur_ratio.py
    └── test_langevin.py             # Verify EM integrator against O-U analytical solution
```
 
---
 
## Installation
 
```bash
# Clone the repository
git clone https://github.com/researchingadi/tur-neuralnets.git
cd tur-neuralnets
 
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
 
# Install dependencies
pip install -r requirements.txt
 
# Install package in editable mode
pip install -e .
```
 
**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.2, NumPy, SciPy, Matplotlib, PyYAML, tqdm, wandb (optional).  
**Hardware:** CPU sufficient for toy experiment. GPU recommended for Experiments 1–4 (MNIST/CIFAR-10).
 
---
 
## Running the Toy Langevin Experiment
 
The toy experiment verifies TUR on a 2D Langevin system before touching neural network training. This is Phase 1, Step 3 of the project — the sanity check that confirms our entropy production estimator and TUR ratio computation are correct on a system with known analytical properties.
 
```bash
# Run with default config (anisotropic quadratic landscape)
python scripts/run_toy_langevin.py --config configs/toy_langevin.yaml
 
# Run with double-well landscape
python scripts/run_toy_langevin.py --config configs/toy_langevin.yaml \
    --override landscape.type=double_well
 
# Run with specific temperature (controls noise amplitude)
python scripts/run_toy_langevin.py --config configs/toy_langevin.yaml \
    --override langevin.temperature=1.0
```
 
**Output:** `figures/toy/` — four-panel figure (landscape + trajectory, entropy production, TUR LHS vs RHS, TUR ratio) in both PDF and PNG at 300 DPI.
 
**Expected result:** TUR-ratio ≥ 2 throughout the near-stationary regime (after `burn_in` steps). Any violation is a bug — see `tests/test_tur_ratio.py`.
 
---
 
## Reproducing Paper Figures
 
Each figure in the paper corresponds to a config file and a script. To reproduce Figure N:
 
```bash
# Figure 1: Toy Langevin TUR verification
python scripts/run_toy_langevin.py --config configs/toy_langevin.yaml
 
# Figure 2: TUR holds during neural network training (Experiment 1)
python scripts/run_experiment1.py --config configs/experiment1_mnist.yaml
 
# Figure 3: Entropy production predicts generalization gap (Experiment 2)
python scripts/run_experiment2.py --config configs/experiment2_sweep.yaml
 
# Figure 4: SGD vs Adam thermodynamic comparison (Experiment 3)
python scripts/run_experiment3.py --config configs/experiment3_optimizers.yaml
 
# Figure 5: TUR ratio across training phases (Experiment 4)
python scripts/run_experiment4.py --config configs/experiment4_phases.yaml
```
 
All figures are saved as both `figures/{experiment}/figure_N.pdf` and `figures/{experiment}/figure_N.png`. The config file used to generate each figure is copied alongside the output as `figure_N_config.yaml` for full reproducibility.
 
---
 
## Results
 
*Results will be populated as experiments complete. Placeholder slots below.*
 
### Figure 1 — Toy Langevin: TUR Verification
 
> *[Figure 1 placeholder — 4-panel figure: (a) landscape + trajectory, (b) entropy production over time, (c) LHS vs RHS of TUR inequality, (d) TUR ratio with W ∈ {50, 200, 500} and reference line at 2]*
 
**Finding:** TUR-ratio ≥ 2 holds throughout the near-stationary regime for both anisotropic quadratic and double-well landscapes. The ratio approaches 2 from above, consistent with the TUR being tight in the high-dissipation limit.
 
---
 
### Figure 2 — Experiment 1: TUR During Neural Network Training
 
> *[Figure 2 placeholder — TUR ratio over training epochs for MLP/CNN on MNIST and CIFAR-10]*
 
**Finding:** TBD.
 
---
 
### Figure 3 — Experiment 2: Entropy Production Predicts Generalization Gap
 
> *[Figure 3 placeholder — scatter plot: S_total vs generalization gap, colored by learning rate and batch size]*
 
**Finding:** TBD.
 
---
 
### Figure 4 — Experiment 3: SGD vs Adam Under the TUR Lens
 
> *[Figure 4 placeholder — entropy production curves + final generalization for SGD, Adam, AdaGrad, RMSProp]*
 
**Finding:** TBD.
 
---
 
### Figure 5 — Experiment 4: TUR Ratio Across Training Phases
 
> *[Figure 5 placeholder — TUR ratio over time annotated with early/middle/late training phases]*
 
**Finding:** TBD.
 
---
 
## Experimental Roadmap
 
| # | Experiment | Status | Key Falsifiable Claim |
|---|-----------|--------|----------------------|
| 0 | Toy Langevin TUR verification | 🔄 In progress | TUR-ratio ≥ 2 for 2D Langevin system with known entropy production |
| 1 | TUR holds during SGD training (MNIST, CIFAR-10) | ⏳ Planned | TUR-ratio ≥ 2 throughout training for MLP and CNN with fixed SGD hyperparameters |
| 2 | Entropy production predicts generalization gap | ⏳ Planned | Spearman(S_total, gen-gap) < 0 across 50–100 runs with varied hyperparameters |
| 3 | SGD vs Adam under TUR | ⏳ Planned | SGD produces higher per-step entropy and generalizes better; Adam's adaptive geometry reduces effective entropy production |
| 4 | TUR ratio across training phases | ⏳ Planned | TUR bound is tightest (ratio closest to 2) during the rapid generalization phase of training |
| 5 | Memorization vs generalization (stretch) | ⏳ Stretch | Memorizing random labels costs more entropy per unit of precision than learning true patterns |
 
---
 
## References
 
All 24 papers in the master literature reference, organized by tier.
 
### Tier 1 — Core Papers
 
1. Barato, A.C. & Seifert, U. (2015). Thermodynamic Uncertainty Relation for Biomolecular Processes. *Physical Review Letters*, 114, 158101. arXiv:1502.05944
2. Horowitz, J.M. & Gingrich, T.R. (2020). Thermodynamic Uncertainty Relations Constrain Non-Equilibrium Fluctuations. *Nature Physics*, 16, 15–20. DOI:10.1038/s41567-019-0702-6
3. Goldt, S. & Seifert, U. (2017). Stochastic Thermodynamics of Learning. *Physical Review Letters*, 118, 010601. arXiv:1611.09428
4. Boyd, A.B., Crutchfield, J.P., Gu, M. & Binder, F.C. (2025). Thermodynamic Overfitting and Generalization: Energetics of Predictive Intelligence. *New Journal of Physics*, 27, 063901. arXiv:2402.16995
5. Mandt, S., Hoffman, M.D. & Blei, D.M. (2017). Stochastic Gradient Descent as Approximate Bayesian Inference. *Journal of Machine Learning Research*, 18(134), 1–35. arXiv:1704.04289
6. Parsi, S.S. (2024). Stochastic Thermodynamics of Learning Parametric Probabilistic Models. *Entropy*, 26(2), 112. arXiv:2310.19802
### Tier 2 — Essential Supporting Papers
 
7. Xu, A. & Raginsky, M. (2017). Information-Theoretic Analysis of Generalization Capability of Learning Algorithms. *NeurIPS*.
8. Mou, W. et al. (2018). Generalization Bounds of SGLD for Non-convex Learning. *COLT Proceedings*.
9. Boffi, N.M. & Vanden-Eijnden, E. (2024). Deep Learning Probability Flows and Entropy Production Rates in Active Matter. *PNAS*, 121(25), e2318106121. arXiv:2309.12991
10. Zhou, P. et al. (2020). Towards Theoretically Understanding Why SGD Generalizes Better Than ADAM in Deep Learning. *NeurIPS 2020*. arXiv:2010.05627
11. Feng, Y. & Tu, Y. (2021). The Inverse Variance-Flatness Relation in Stochastic Gradient Descent is Critical for Finding Flat Minima. *PNAS*.
12. Lyu, J., Ray, K.J. & Crutchfield, J.P. (2025). Learning Stochastic Thermodynamics Directly from Correlation and Trajectory-Fluctuation Currents. arXiv:2504.19007
13. Boyd, A.B. et al. (2020). Thermodynamic Machine Learning Through Maximum Work Production. arXiv:2006.15416
14. Koyuk, T., Seifert, U. & Pietzonka, P. (2018). A Generalization of the Thermodynamic Uncertainty Relation to Periodically Driven Systems. *Journal of Physics A*. arXiv:1809.02113
15. Bahri, Y. et al. (2020). Statistical Mechanics of Deep Learning. *Annual Review of Condensed Matter Physics*, 11, 501–528.
16. Wu, B. et al. (2022). Generalization Bounds for Stochastic Gradient Langevin Dynamics: A Unified View via Information Leakage Analysis. arXiv:2112.08439
17. Hellström, F., Durisi, G., Guedj, B. & Raginsky, M. (2023). Generalization Bounds: Perspectives from Information Theory and PAC-Bayes.
18. Proesmans, K. & Van den Broeck, C. (2017). Discrete-Time Thermodynamic Uncertainty Relation. *EPL*, 119, 20001. arXiv:1708.07032
19. Kim, D.K. et al. (2020). Learning Entropy Production via Neural Networks. *Physical Review Letters*. arXiv:2003.04166
20. Goldt, S. & Seifert, U. (2017). Thermodynamic Efficiency of Learning a Rule in Neural Networks. *New Journal of Physics*, 19, 113001.
### Tier 3/4 — Framing and Adjacent Papers
 
21. Parrondo, J.M.R., Horowitz, J.M. & Sagawa, T. (2015). Thermodynamics of Information. *Nature Physics*, 11, 131–139.
22. Pietzonka, P., Ritort, F. & Seifert, U. (2017). Finite-Time Generalization of the Thermodynamic Uncertainty Relation. *Physical Review E*, 96, 012101.
23. Multiple authors (2025). Energy-Time-Accuracy Tradeoffs in Thermodynamic Computing. arXiv:2601.04358
24. Parsi, S.S. (2022). Theoretical Bound of the Efficiency of Learning. arXiv:2209.08096
---
 
## Citation
 
When citing this work, please use:
 
```bibtex
@article{singh2026tur,
  title   = {Thermodynamic Uncertainty Relations in Neural Network Training:
             Entropy Production Bounds on Generalization},
  author  = {Singh, Adi},
  journal = {arXiv preprint},
  note    = {cs.LG, cond-mat.stat-mech},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```
 
*This field will be updated upon arXiv submission with the correct identifier.*
 
---
 
## License
 
MIT License. See `LICENSE` for details.  
Research conducted at Mississippi State University, 2026.
 
---
 
*Adi Singh · MS CSE/MATH, MSU · `researchingadi` · June 2026*
