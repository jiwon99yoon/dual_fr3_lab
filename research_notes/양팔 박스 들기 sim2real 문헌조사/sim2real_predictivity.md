# Sim-to-Real Predictivity: Does Sim Evaluation Predict Real Performance/Ranking, and How to Select Policies via (Targeted) Simulation

Scope note: notes target a paper claim of the form "failure-targeted sim stress tests (low-friction slip, box rotation) predict real ranking better than nominal or wide-DR sim success" for dual-arm plate squeeze-lift (PPO, Isaac Lab). Our own data point: a 100%-nominal-sim policy got 1/12 real; a low-friction-trained/evaluated policy got 8/10; Isaac Sim underestimated box rotation 2-3x vs MuJoCo.

Legend: [V] = verified against the primary source (abstract/full text) in this session. [S] = from search-result snippets/secondary summary, not fully verified. [M] = from background knowledge, not re-checked in this session; verify before citing.

## Q1. Kadian et al. "Sim2Real Predictivity" (Habitat, SRCC): details, metric, findings

### Takeaway
The canonical reference. With 9 navigation policies evaluated in matched sim and real episodes, default Habitat had SRCC_Succ = 0.18 (sim ranking nearly useless) because agents exploited a simulator artifact (wall sliding). Turning off the exploitable artifact and adding real actuation noise raised SRCC_Succ to 0.844. This is the closest precedent for "the sim artifact the policy exploits destroys predictivity". Our friction/rotation result is a contact-rich manipulation analogue.

### Cited Findings
- Paper: "Sim2Real Predictivity: Does Evaluation in Simulation Predict Real-World Performance?" by A. Kadian, J. Truong, A. Gokaslan, A. Clegg, E. Wijmans, S. Lee, M. Savva, S. Chernova, D. Batra. IEEE RA-L 2020 (arXiv Dec 2019). [V] — [arXiv:1912.06321](https://arxiv.org/abs/1912.06321)
- Metric: SRCC = sample Pearson correlation over the paired dataset {(s_i, r_i)} of per-method sim and real accuracy (success rate or SPL) for n methods. A footnote says "other metrics such as rank correlation can also be used." [V] — [PDF](https://arxiv.org/pdf/1912.06321)
- They also treat SRCC as an objective for simulator design: max_θ SRCC(S_n(θ), R_n) over test-time simulator parameters θ (actuation noise, sliding, etc.), with the policies held fixed. [V] — [PDF](https://arxiv.org/pdf/1912.06321)
- Protocol: 9 PointGoal navigation models (DD-PPO, 500M steps), 3 room configurations x 5 spawn-to-goal waypoints x 3 trials = 810 runs total in sim and real combined (405 real runs). About 6 min per real episode, so 40.5 h of real testing (about 5 days with human monitoring). Start/goal poses are identical in sim and real; platform is LoCoBot via PyRobot. [V] — [PDF](https://arxiv.org/pdf/1912.06321)
- Findings: CVPR19-challenge Habitat settings gave SRCC_SPL = 0.603 and SRCC_Succ = 0.18. The cause was agents "abusing collision dynamics to 'slide' along walls", cutting corners through non-navigable space, which does not work in reality. Disabling sliding and adding a LoCoBot-derived actuation-noise model gave SRCC_SPL = 0.875 and SRCC_Succ = 0.844. [V] — [arXiv abstract](https://arxiv.org/abs/1912.06321); [PDF](https://arxiv.org/pdf/1912.06321)
- They argue that a simulator with low SRCC cannot be used for model selection even if mean real performance is high, because sim changes have an unknown sign of effect on real performance. [V] — [PDF](https://arxiv.org/pdf/1912.06321)

### Inferences
- The structure maps directly onto our case. A policy reaches 100% nominal sim success by exploiting an unrealistic contact property (high or idealized friction, underestimated box rotation in PhysX), just as Habitat agents exploited wall sliding. Changing test-time parameters toward the real failure mode (low μ, rotation) should raise sim-real correlation. A Kadian-style table "SRCC/Spearman under nominal vs wide-DR vs targeted-stress evaluation" is a natural headline result.
- Caveat to pre-empt in review: Kadian tuned θ to maximise SRCC on the same 9 models used to report SRCC, so there is no held-out set of policies. A stronger paper would choose the stress-test parameters a priori, from failure analysis or a cross-simulator check, or validate on held-out policies.
- With n = 9 policies, the Pearson CI is wide. Our study will likely have fewer policies, so it should report rank statistics with bootstrap CIs.

### Gaps
- Kadian does not report confidence intervals on SRCC. Statistical uncertainty in the SRCC values could not be verified.

## Q2. Similar studies in manipulation/locomotion: sim-real correlation, real-to-sim evaluation, digital twins, OPE, checkpoint selection

### Takeaway
Since 2024 there has been a wave of real-to-sim evaluation work for generalist (VLA/imitation) manipulation policies: SIMPLER, PolaRiS, Gaussian-splat twins, SureSim. They report Pearson r and MMRV over about 4-10 policies and mostly cover pick/place/drawer tasks with visual-gap emphasis. For RL sim-to-real checkpoint selection, the evidence is thinner: Khor and Weng 2025 for humanoid locomotion, VSDR 2022 for grasping, and off-policy classification 2019 for grasping. Anecdotes that "near-100% sim success does not imply real success" exist for contact-rich assembly, but I found no systematic sim-real ranking study for contact-rich/bimanual RL policies.

### Cited Findings
**SIMPLER (real-to-sim for generalist manipulation policies)**
- "Evaluating Real-World Robot Manipulation Policies in Simulation" by X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, S. Levine, J. Wu, C. Finn, H. Su, Q. Vuong, T. Xiao. arXiv May 2024; published at CoRL 2024 [M for venue]. [V] — [arXiv:2405.05941](https://arxiv.org/abs/2405.05941)
- They identify control and visual disparities as the key sim-real gaps and mitigate them without full-fidelity digital twins. Paired sim-real evaluation shows strong correlation, and SIMPLER also reproduces real sensitivity to distribution shifts. [V] — [arXiv:2405.05941](https://arxiv.org/abs/2405.05941)
- Metrics: MMRV (Mean Maximum Rank Violation) weights each mis-ranked pair by its real success-rate gap, so near-ties caused by eval noise matter less. They report it alongside Pearson r. [V] — [arXiv HTML](https://arxiv.org/html/2405.05941)
- On the Google Robot, 6 checkpoints were evaluated: 3 RT-1, RT-1-X, RT-2-X, and Octo-Base.

  | Selection signal | MMRV | Pearson r |
  |---|---|---|
  | Offline validation MSE | 0.375 | 0.308 |
  | SIMPLER Variant Aggregation | 0.143 | 0.778 |
  | SIMPLER Visual Matching | 0.056 | 0.924 |

  [V] — [arXiv HTML](https://arxiv.org/html/2405.05941)
- Control gap matters for ranking. System-identified PD gains gave the lowest tracking loss (0.131) and the lowest MMRV (0.031) on pick-coke-can. The other gain settings gave MMRV 0.070 and 0.100. Using the real controller's PD values naively gave inaccurate tracking and missed grasps. [V] — [arXiv HTML](https://arxiv.org/html/2405.05941)
- The sim pick-coke-can task uses 75 trials (25 positions x 3 orientations). The real trial count per task was not extracted. [V/partial] — [arXiv HTML](https://arxiv.org/html/2405.05941)

**PolaRiS (2025)**
- "PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies" by A. Jain, M. Zhang, K. Arora, W. Chen, M. Torne, M. Z. Irshad, S. Zakharov, Y. Wang, S. Levine, C. Finn, W.-C. Ma, D. Shah, A. Gupta, K. Pertsch. arXiv Dec 2025. Neural reconstruction from video scans plus brief sim co-training gives a "much stronger correlation to real world generalist policy performance than existing simulated benchmarks." [V] — [arXiv:2512.16881](https://arxiv.org/abs/2512.16881)
- Reported average Pearson r = 0.83 with real performance and r = 0.98 with RoboArena scores. [S] — [PDF](https://arxiv.org/pdf/2512.16881)

**Other real-to-sim / digital-twin evaluation**
- Gaussian-splatting soft-body real-to-sim evaluation (arXiv 2511.04665): sim success rates "generally follow the same progression" as real, so the twin is usable for checkpoint selection. Mismatches are attributed to residual visual/dynamics gaps and to only 16-27 evaluation episodes per checkpoint. [S] — [arXiv:2511.04665](https://arxiv.org/pdf/2511.04665)
- An RSS 2026 paper (RSS XXII p062) on real-to-sim evaluation appeared in search results. Its title and authors were not verified. [S, unverified] — [RSS proceedings](https://roboticsproceedings.org/rss22/p062.html)
- SimFoundry (single-video digital twin) reports mean Pearson 0.911 across 7 tasks and 5 policies, and an "SO-101 Bench" reports identical sim/real ranking of 5 policies (MMRV 0). Both come from news aggregators only. [S, unverified; aggregator source] — [aiweekly SimFoundry](https://aiweekly.co/alerts/simfoundry-predicts-real-robot-policy-success-from-sim-at-0911-correlation); [aiweekly SO-101](https://aiweekly.co/alerts/so-101-bench-shows-sim-policy-rankings-match-real-hardware)
- Runway GWM-Robotics (world-model-based evaluation, company blog) reports Pearson 0.95 over 8 policies and MMRV 0 on the PolaRiS subset. This is not peer-reviewed. [S] — [Runway](https://runway.com/research/accelerating-robot-policy-evaluation)

**Sim + few real trials with statistical guarantees**
- "Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators" (SureSim) by A. Badithela, D. Snyder, L. Zha, J. Mikhail, M. O'Kelly, A. Dixit, A. Majumdar. arXiv Oct 2025. It frames evaluation as prediction-powered inference: a few paired real/sim trials correct the sim bias, and non-asymptotic CIs on real performance follow. It reports 20-25% less hardware effort for similar bounds, on diffusion policy and π0. [V] — [arXiv:2510.04354](https://arxiv.org/abs/2510.04354)
- Per a search summary, SureSim notes that most studies rely on about 20-40 hardware trials and that the method helps only when the sim-real gap is not too large. [S] — [arXiv HTML](https://arxiv.org/html/2510.04354v1)

**RL policy / checkpoint selection for sim-to-real**
- "Post-Convergence Sim-to-Real Policy Transfer: A Principled Alternative to Cherry-Picking" by D. Khor and B. Weng (Iowa State). arXiv Apr 2025. [V] — [arXiv:2504.15414](https://arxiv.org/html/2504.15414v1)
  - Setup: Unitree G1 locomotion, trained in Isaac Gym and evaluated in MuJoCo. The predictor is a worst-case expected reward over distributions within a KL ball of the simulator's stationary distribution, solved as a convex QCLP.
  - Real-ranking Spearman: on 13 undisturbed policies, training reward has negative SCC. Direct sim (SIM-UD) and adversarial sim (SIM-ADV) evaluation both stay below 0.25, while the worst-case estimate reaches 0.55. On the 5-policy disturbed test (foam, balls, plates, pushes up to 20 N), sim-disturbed evaluation reaches SCC 1.0 with reward r1. With a stability reward r2, direct sim is −0.3 and worst-case is up to 0.9.
  - Limitations: finite discretized state space, ad-hoc KL radius k, one robot, small policy sets.
  - This is the closest existing precedent to our claim. A "disturbed" sim test resembling real conditions ranked policies correctly where nominal sim and training reward did not.
- "Validate on Sim, Detect on Real – Model Selection for Domain Randomization" (VSDR) by G. Leibovich, G. Jacob, S. Endrawis, G. Novik, A. Tamar. ICRA 2022. It combines sim evaluation with OOD detection on a fixed set of real images to rank DR-trained grasping policies, and improves ranking accuracy without additional real data. [V] — [arXiv:2111.00765](https://arxiv.org/abs/2111.00765)
  - The ground truth required evaluating every policy on the real robot over 49 grasp attempts, about 130 h of robot time. [S] — [Intel Labs blog](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-Labs-Validate-on-Sim-Detect-on-Real-VSDR-Narrows-Sim2Real/post/1415787)
- "Off-Policy Evaluation via Off-Policy Classification" by A. Irpan, K. Rao, K. Bousmalis, C. Harris, J. Ibarz, S. Levine (author list [M]). NeurIPS 2019. It casts OPE as positive-unlabeled classification. SoftOPC best predicted real grasp success, including ranking sim-trained policies on the real robot from prior real data, to choose which models to test on hardware. [S] — [Google Research blog](https://www.research.google/blog/off-policy-classification-a-new-reinforcement-learning-model-selection-method/)
- "Assessing Transferability from Simulation to Reality for Reinforcement Learning" (SPOTA, simulation optimization-bias estimate) by F. Muratore, M. Gienger, J. Peters. IEEE TPAMI 2021 (arXiv 1907.04685). It estimates an upper confidence bound on the "optimality gap" across randomized sims as a stopping/transfer criterion. [M, unverified this session] — [arXiv:1907.04685](https://arxiv.org/abs/1907.04685)

**Anecdotes: high sim success does not imply real success, contact-rich**
- CoRMA (arXiv 2605.22082, 2026): FORGE baselines "achieve high simulation success but degrade substantially on hardware" on contact-rich insertion. [V, abstract-level] — [arXiv:2605.22082](https://arxiv.org/abs/2605.22082)
  - A search snippet quotes near-perfect sim success vs 25.0% / 12.5% real on GearMesh / PegInsert. [S] — [PDF](https://arxiv.org/pdf/2605.22082)
- IndustReal (Tang et al., RSS 2023): 83-99% real success over 600 trials on contact-rich assembly. [S] — [arXiv:2305.17110](https://arxiv.org/pdf/2305.17110)
- AutoMate (2024): specialist policies drop about 4.2% sim to real, while the generalist is 4.1% higher in real than in sim. Sim-real gaps in contact-rich assembly can therefore be small when sim is carefully calibrated (SDF contact, sys-ID). [S] — [arXiv:2407.08028](https://arxiv.org/pdf/2407.08028v2)
- DREAM (2026): sim success is systematically 7-14 pp below real for most policies. A level offset can co-exist with good ranking. [S] — [arXiv:2608.29078](https://arxiv.org/pdf/2608.29078)

### Inferences
- Nearly all quantitative sim-real ranking studies in manipulation are about imitation/VLA generalists with visual gaps, on quasi-static tasks (pick, move, drawer). The dominant fixes are visual matching and controller sys-ID. Contact-dynamics gaps (friction, compliance, solver artifacts) as the driver of rank inversion are under-studied. SIMPLER's PD-gain result is the closest: dynamics mismatch changed MMRV by about 3x.
- Precedents where nominal sim fails to rank and a perturbation-based sim test succeeds exist only in navigation (Kadian) and humanoid locomotion (Khor and Weng). Our contribution would be the first, as far as I found, for contact-rich bimanual manipulation trained with RL.
- The Khor and Weng result that generic adversarial sim (SIM-ADV) also failed (< 0.25) supports the distinction that targeted tests matching the real failure mode beat generic worst-case/adversarial or wide-DR evaluation. The paper should include a generic-adversarial or wide-DR baseline, not just nominal.

### Gaps
- I found no study reporting sim-real rank correlation for bimanual or squeeze/grasp-friction-dominated tasks.
- Real trial counts for SIMPLER and PolaRiS were not extracted.
- PolaRiS numbers come from a snippet only.

## Q3. Cross-simulator evaluation (sim-to-sim as a proxy for sim-to-real)

### Takeaway
Isaac Gym/Lab to MuJoCo sim-to-sim is a widespread pre-deployment gate, especially in legged/humanoid RL. Multi-simulator training (PolySim) is emerging. However, I found no study quantifying how well cross-simulator performance (or disagreement) predicts real ranking across many policies. Our observation that Isaac underestimates rotation 2-3x vs MuJoCo would be novel quantitative evidence if paired with real outcomes.

### Cited Findings
- Khor and Weng evaluate Isaac Gym-trained G1 policies in MuJoCo as "simulation". The rankings they compare to real are therefore MuJoCo rankings, which implicitly makes a cross-simulator evaluation the predictor. [V] — [arXiv:2504.15414](https://arxiv.org/html/2504.15414v1)
- SKooP (arXiv 2607.11624) uses Isaac Gym to MuJoCo sim-to-sim and finds MuJoCo models joint and contact physics more faithfully. Isaac Gym's default zero armature caused failures in MuJoCo. [S] — [arXiv:2607.11624](https://arxiv.org/pdf/2607.11624)
- H2-COMPACT (arXiv 2505.17627) uses Isaac Gym training, a MuJoCo validation gate, and then a real Unitree G1. [S] — [arXiv:2505.17627](https://arxiv.org/pdf/2505.17627)
- PolySim (arXiv 2510.01708) trains across multiple simulators (multi-simulator dynamics randomization) for humanoid motion tracking. It reports a 52.8% gain in MuJoCo sim-to-sim execution success over an IsaacSim-only baseline, with zero-shot real G1 transfer. [S] — [arXiv:2510.01708](https://arxiv.org/html/2510.01708v3)
- An NVIDIA Isaac Lab article recommends running the same policy on several physics backends, including MuJoCo, to surface solver-specific contact/friction artifacts. This is a vendor source. [S] — [NVIDIA perspectives](https://perspectives.nvidia.com/isaac-lab/which-framework-makes-it-easier-to-test-robot-policies-across/)

### Inferences
- Cross-simulator disagreement can serve as a cheap detector of "exploited simulator artifacts", in the same role as wall-sliding in Kadian. Where solvers disagree (box rotation under squeeze, friction-cone handling), real outcomes are most uncertain, so those axes are where targeted stress tests should be placed. This is a methodological argument, not an established result.
- A paper could report SRCC/Spearman for (a) Isaac nominal, (b) Isaac wide-DR, (c) Isaac targeted low-μ/rotation stress, and (d) MuJoCo nominal, against real outcomes. Cross-sim would be both a predictor and a justification for the stress parameters.

### Gaps
- I found no manipulation paper quantifying cross-simulator ranking vs real ranking.
- I found no published measurement of PhysX vs MuJoCo rotational slip discrepancy for squeeze grasps.

## Q4. Stress testing / adversarial evaluation / falsification of robot policies

### Takeaway
A mature literature (AST, black-box safety validation, falsification) finds failures and estimates failure probabilities in simulation, and robot-manipulation variants exist: RoboFail, RADIUM, geometric red-teaming, ROBOGATE. These works almost never validate that sim-discovered failures or sim stress scores predict real-world ranking or success rates. They focus on finding failures or repairing policies, which is exactly the gap our claim targets.

### Cited Findings
- "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning" by R. Lee, O. J. Mengshoel, A. Saksena, R. W. Gardner, D. Genin, J. Silbermann, M. Owen, M. J. Kochenderfer. JAIR vol. 69, 2020, pp. 1165-1201. It formulates "most likely failure path in simulation" as an MDP solved with RL/MCTS, applied to aircraft collision avoidance. The DAST extension finds failures present in one system but not another. [S, multiple consistent sources] — [JAIR](https://jair.org/index.php/jair/article/view/12190); [arXiv:1811.02188](https://arxiv.org/abs/1811.02188v2)
- "A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems" by A. Corso, R. Moss, M. Koren, R. Lee, M. J. Kochenderfer. JAIR vol. 72, 2021. It formalizes falsification, most-likely failure, and failure-probability estimation via optimization, path planning, RL, and importance sampling. [S] — [arXiv:2005.02979](https://arxiv.org/abs/2005.02979v3); [JAIR](https://jair.org/index.php/jair/article/view/12716)
- RoboFail (arXiv 2412.02818, 2024): a PPO adversary perturbs sim environment attributes to find failure configurations of manipulation policies and estimate failure-mode probabilities. [S] — [arXiv HTML](https://arxiv.org/html/2412.02818v1)
- RADIUM (arXiv 2404.03412, 2024): uses differentiable simulation and gradient-accelerated sampling to predict diverse end-to-end robot failures and repair the policy. Claims up to 10x lower cost than optimization-based falsification. [S] — [arXiv HTML](https://arxiv.org/html/2404.03412v1)
- Geometric red-teaming (CMU, "red and blue teaming for robust manipulation under geometric robustness"): searches object-geometry deformations that break grasping policies, finding failures for over 80% of held-out objects in sim. Retraining on these improved grasp success by up to 21% in sim and 30% on the real robot. The real validation is of the repair, not of ranking. [S] — [CMU RI](https://publications.ri.cmu.edu/red-and-blue-teaming-for-robust-manipulation-under-geometric-robustness)
- ROBOGATE (arXiv 2603.22126, 2026): two-stage boundary-focused adaptive sampling over operating parameters, with 30k runs in Isaac Sim. The two arXiv versions report inconsistent headline numbers (Octo-Small 30.9% vs 0.0%) and inconsistent author names, so treat it with caution. [S] — [arXiv:2603.22126](https://arxiv.org/pdf/2603.22126)
- Falsification of AI manipulation controllers in physical simulators: dual annealing was best and Nelder-Mead was poor. [S] — [project site](https://sites.google.com/view/ai-cps-robotics-manipulation/rq3)
- Khor and Weng's adversarial-sim baseline (SIM-ADV) did not predict real G1 rankings (SCC < 0.25), while a realistic disturbed-sim test did. Generic adversarial pressure is therefore not automatically predictive. [V] — [arXiv:2504.15414](https://arxiv.org/html/2504.15414v1)

### Inferences
- Positioning: stress testing has been used to find failures and repair policies. Our claim instead uses a stress test as a ranking/selection signal validated against real trials. Only Khor and Weng (locomotion) does something close.
- "Failure-targeted" should be operationalized so that it is not post-hoc. One option is to derive the stress axes from (i) real failure observations on a pilot policy, (ii) cross-simulator disagreement, or (iii) physical reasoning (friction-cone margin under squeeze). Then show predictivity on policies not used to choose the axes.

### Gaps
- I found no work validating AST/falsification-derived scores against real-robot rank correlation in manipulation.

## Q5. Typical number of real trials and statistics used

### Takeaway
Typical real evaluations use about 10-50 trials per policy and 4-13 policies. Statistics used are Pearson (SRCC), Spearman, and MMRV for rankings; Wilson/Clopper-Pearson CIs and Fisher's exact or sequential tests for pairwise comparisons. With n ≈ 10-12 trials per policy, only large gaps (e.g., 1/12 vs 8/10) are statistically separable.

### Cited Findings
- Kadian: 9 policies, 405 real runs (45 per policy), Pearson SRCC. [V] — [PDF](https://arxiv.org/pdf/1912.06321)
- SIMPLER: 6 policies on the Google Robot; MMRV + Pearson. [V] — [arXiv HTML](https://arxiv.org/html/2405.05941)
- Khor and Weng: 13 and 5 policies; Spearman. [V] — [arXiv:2504.15414](https://arxiv.org/html/2504.15414v1)
- VSDR: 49 real grasps per policy, about 130 robot-hours. [S] — [Intel blog](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-Labs-Validate-on-Sim-Detect-on-Real-VSDR-Narrows-Sim2Real/post/1415787)
- Gaussian-splat real-to-sim: 16-27 episodes per checkpoint, noted as a source of mismatch. [S] — [arXiv:2511.04665](https://arxiv.org/pdf/2511.04665)
- About 20-40 hardware trials is typical, per SureSim. [S] — [arXiv:2510.04354](https://arxiv.org/html/2510.04354v1)
- STEP ("Is Your Imitation Learning Policy Better than Mine?", arXiv 2503.10966): sequential testing for policy comparison. It reduces trials by up to 40% (v1) or 32% (v4) vs batch tests, and warns that adding trials post hoc is p-hacking. [S] — [arXiv:2503.10966](https://arxiv.org/html/2503.10966v4)
- N-SCORE ("Beyond Binary Success", RSS 2026): anytime-valid sequential comparison for non-binary metrics, claiming up to 70% fewer trials than batch methods. [S] — [RSS 2026](https://roboticsconference.org/program/papers/76/)
- Practitioner guidance recommends Wilson score intervals and Fisher's exact test for small-n success rates. This is a non-academic source. [S] — [claru.ai](https://claru.ai/guides/how-to-evaluate-policy-performance)

### Inferences
- For our data, one-sided Fisher's exact test on 1/12 vs 8/10 gives p ≈ 0.0011 (two-sided 0.0015; computed with scipy, not cited). The Wilson 95% CIs are about [0.015, 0.35] for 1/12 and [0.49, 0.94] for 8/10, so the pairwise reversal is statistically solid.
- Ranking claims need more policies. With k policies, Spearman ρ requires k ≥ 5 even for p < 0.05 at ρ = 1. A paper should evaluate about 5-8 policies (seeds x training-friction settings x checkpoints) with about 10 real trials each. It should report Spearman/Kendall plus MMRV with bootstrap CIs over trials, for each sim predictor: nominal, wide-DR, targeted stress, and cross-sim.

### Gaps
- I found no consensus minimum for the number of policies or trials in sim-real predictivity studies. Most papers do not report CIs on the correlation itself.

## Q6. What remains open for contact-rich bimanual manipulation?

### Takeaway
Open problems: (1) no quantitative sim-real ranking study for contact-rich or bimanual RL manipulation; (2) no evidence comparing nominal vs wide-DR vs failure-targeted sim evaluation as ranking predictors in manipulation; (3) no validated method for choosing stress-test axes a priori (e.g., from cross-simulator disagreement); (4) no statistical standards for small-n predictivity claims. Our study can address (1)-(3) directly.

### Cited Findings
- Real-to-sim predictivity work is concentrated on generalist, mostly single-arm, quasi-static tasks, with visual and controller gaps as the main levers: SIMPLER, PolaRiS, SureSim. [V] — [SIMPLER](https://arxiv.org/abs/2405.05941); [PolaRiS](https://arxiv.org/abs/2512.16881); [SureSim](https://arxiv.org/abs/2510.04354)
- The only RL checkpoint-selection predictivity study found with a disturbed/worst-case sim test beating nominal is humanoid locomotion (Khor and Weng). [V] — [arXiv:2504.15414](https://arxiv.org/html/2504.15414v1)
- In contact-rich insertion, high sim success with poor real transfer is reported (CoRMA/FORGE), but it was not analyzed as a predictivity/ranking problem. [V/S] — [arXiv:2605.22082](https://arxiv.org/abs/2605.22082)
- Kadian shows a simulator artifact exploited by policies drives SRCC toward zero, and removing it restores predictivity. This is the mechanism we hypothesize for friction and rotation in squeeze-lift. [V] — [PDF](https://arxiv.org/pdf/1912.06321)

### Inferences
- A defensible claim would read: "In contact-rich bimanual squeeze-lift, nominal and wide-DR sim success are non-predictive or inversely predictive of real success. A small set of failure-targeted stress tests (low μ, induced rotation), chosen from cross-simulator disagreement and failure physics, recovers the real ranking (Spearman/MMRV)." It should be framed as an extension of Kadian's SRCC and SIMPLER's MMRV to contact dynamics, and contrasted with Khor and Weng (locomotion) and stress-testing work (failure finding, not ranking).
- Threats to validity reviewers will raise:
  - Post-hoc choice of stress axes after seeing real failures. Mitigate with held-out policies or pre-registration-style ordering.
  - Small k (number of policies).
  - Confounding when the low-friction policy is both trained and evaluated at low μ. Separate train-condition from eval-condition in a cross table.
  - Single task. Consider a second object/mass or a second friction surface.

### Gaps
- No bimanual-specific predictivity literature was found. The searches were limited (about 15 queries), so a dedicated search for "bimanual sim-to-real evaluation" and contact-rich digital-twin evaluation is advisable before claiming novelty.
- I could not verify full author lists or venues for several [S] items (PolaRiS numbers, RoboFail, RADIUM, geometric red-teaming, ROBOGATE, STEP, N-SCORE, RSS 2026 p062).
