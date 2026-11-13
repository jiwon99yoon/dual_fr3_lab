# Friction/Contact DR, Real-to-Sim Parameter Identification, and Curricula over DR Difficulty (contact-rich manipulation, RL sim-to-real)

Context: dual-arm FR3 plate squeeze-lift of a box, PPO in Isaac Lab. Nominal-sim policy 100% sim, 1/12 real. Real failure (plate slip, then box rotation) reproduced in sim only at mu ~0.22-0.25, below the training range 0.45-0.85. Retraining with mu in 0.21-0.64 gave 8/10 real. The range was set by hand from observed failures. The staged success-gated curriculum worked; turning on all stages at once collapsed learning. Isaac Sim (PhysX) under-predicted box rotation 2-3x relative to MuJoCo.

Verification legend: [V] = metadata and key claim checked against arXiv, proceedings or project page in this session. [V-abs] = checked against abstract or search summary only. [K] = from prior knowledge of the paper, not re-fetched this session. Treat [K] numbers as needing a check before citing in a paper.

## Q1. Real-to-sim / system identification for choosing DR ranges (SimOpt, BayesSim, DROPO, ASID, TuneNet, differentiable sim)

### Takeaway
The standard baselines fit a distribution, not a point, over simulator parameters (friction, mass, damping, CoM) to real trajectories. They do this by trajectory-discrepancy minimisation (SimOpt), likelihood-free posterior inference (BayesSim), offline likelihood maximisation (DROPO), Fisher-information-driven exploration (ASID), or learned one-shot residuals (TuneNet). All of them match state trajectories. None of them targets reproducing a task failure mode, such as slip followed by rotation, and none was shown on a bimanual squeeze grasp. Most were evaluated on single-arm, single-object tasks with a few real rollouts.

### Cited Findings
**SimOpt.** Chebotar, Handa, Makoviychuk, Macklin, Issac, Ratliff, Fox. "Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience." ICRA 2019. [V-abs] https://arxiv.org/abs/1810.05687
- Method: alternates policy training under a Gaussian DR distribution with an update of that distribution's mean and covariance. The update minimises the discrepancy between real and simulated observation trajectories for the same policy, using relative-entropy policy search over sim parameters. — [arXiv](https://arxiv.org/pdf/1810.05687)
- Tasks: swing-peg-in-hole on an ABB YuMi (rope parameters dominate) and drawer opening on a Franka Panda. Simulator: NVIDIA FleX. Real object states tracked with DART. After 2 SimOpt iterations the swing-peg policy succeeded in 90% of 20 real trials. — [arXiv](https://arxiv.org/pdf/1810.05687); [NVIDIA news](https://news.developer.nvidia.com/?p=13945)
- Drawer failure before adaptation: the finger pushed too hard on the handle and the gripper opened. One iteration fixed the gripper orientation. This is an example of a contact failure mode that disappeared after calibration, but the paper did not set out to target it. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1810.05687)
- Relevance: the closest classic baseline to our loop. It also explicitly argues against making DR ranges too wide, because over-wide distributions give conservative or infeasible policies [K]. That fits our 0.45-0.85 range being "wide but in the wrong place".

**BayesSim.** Ramos, Possas, Fox. "BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators." RSS 2019. [V-abs] https://arxiv.org/abs/1906.01728 , RSS proceedings: https://m.roboticsproceedings.org/rss15/p29.html
- Likelihood-free inference using a Mixture Density Random Fourier Network. Given black-box simulator rollouts and real trajectories, it outputs a Gaussian-mixture posterior over parameters such as friction, mass and damping. The posterior is then used directly as the DR distribution. — [arXiv](https://arxiv.org/abs/1906.01728)
- Policies trained on the posterior beat a uniform prior and were reported to be more robust to the actual real-world friction (for example, a real cart-pole/pushing setting [K: exact task details unverified]). — [deepai summary](https://deepai.org/publication/bayessim-adaptive-domain-randomization-via-probabilistic-inference-for-robotics-simulators)
- Follow-ups: Possas et al., "Online BayesSim" (2020), and Antonova et al., "BayesSimIG" (Isaac Gym integration, 2021). [K, unverified this session]

**DROPO.** Tiboni, Arndt, Kyrki. "DROPO: Sim-to-Real Transfer with Offline Domain Randomization." Robotics and Autonomous Systems vol. 166, 2023 (arXiv 2201.08434). [V-abs] https://arxiv.org/abs/2201.08434 ; code: github.com/gabrieletiboni/dropo
- Uses one offline real dataset and needs no real access during optimisation. It replays logged actions in sim from the real states, models the next-state distribution as Gaussian, and maximises the likelihood of the real transitions with CMA-ES over the DR distribution's mean and variance. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.08434)
- Recovered parameter distributions in sim. In a real pushing task with an unmodelled CoM offset, it found a distribution that compensated, and it outperformed DROID, BayesSim and uniform DR. The authors flag that the gradient-free optimiser limits scaling to higher-dimensional parameter spaces. — [Aalto repository](https://aaltodoc.aalto.fi/items/c66c7c07-c9df-44e7-96df-1b6579d78112/full)
- Relevance: the most practical baseline for us. It shows that variance (range width), not just the mean, can absorb unmodelled effects. That is analogous to a widened low-friction range absorbing PhysX's rotation underestimate.

**ASID.** Memmel, Wagenmaker, Zhu, Yin, Fox, Gupta. "ASID: Active Exploration for System Identification in Robotic Manipulation." ICLR 2024 (oral). [V] https://arxiv.org/abs/2404.12308 ; project: https://weirdlabuw.github.io/asid
- Trains an exploration policy in the (inaccurate) sim to maximise Fisher information about the physical parameters. It runs that policy once on the real robot, identifies the parameters, then trains the task policy in the calibrated sim. — [arXiv html](https://arxiv.org/html/2404.12308v2)
- Parameters identified include rolling friction, tangential surface friction, per-patch ("multi-friction") friction, stiffness, the puck's sliding and torsional friction (shuffleboard), rod inertia / mass distribution (rod balancing), and articulation. Typically a single real episode suffices. — [arXiv html](https://arxiv.org/html/2404.12308v2)
- Real results: rod balancing 6/9 for ASID vs 0/9 for DR. Shuffleboard 4/5 (near zone) and 3/5 (far zone) vs 2/5 and 1/5 for DR. MuJoCo-based (inferred from citations). — [arXiv html](https://arxiv.org/html/2404.12308v2)
- Relevance: the strongest recent baseline that explicitly identifies friction for manipulation. The shuffleboard task identifies torsional friction, which bears on our box-rotation discrepancy.

**TuneNet.** Allevato, Short, Pryor, Thomaz. "TuneNet: One-Shot Residual Tuning for System Identification and Sim-to-Real Robot Task Transfer." CoRL 2019 (PMLR v100, 2020). [V-abs] https://proceedings.mlr.press/v100/allevato20a.html
- A network trained purely on sim-sim observation pairs predicts the parameter residual between two models from one target observation. It can identify values outside the training distribution and was used for sim-to-real transfer on a dynamic manipulation (bouncing/tossing) task. — [PMLR](https://proceedings.mlr.press/v100/allevato20a.html)

**Differentiable / neural real-to-sim** (friction identification by gradients):
- Geilinger et al. "ADD: Analytically Differentiable Dynamics for Multi-Body Systems with Frictional Contact" (SIGGRAPH Asia 2020 [K for venue]). Includes real-to-sim estimation of stiffness, damping and friction, and notes that frictional contact makes the gradients hard. — [arXiv 2007.00987](https://arxiv.org/abs/2007.00987)
- Zhu et al. "Real-to-Sim via End-to-End Differentiable Simulation and Rendering" (arXiv 2412.00259, 2024). Targets properties such as the coefficient of friction that are hard to infer from observation alone. [V-abs] — [arXiv](https://arxiv.org/html/2412.00259v1)
- RigPI (arXiv 2606.25212, 2026). Identifies mass, inertia and friction with a VLM-seeded differentiable sim and validates by replaying trajectories on hardware. Preprint, not peer reviewed. [V-abs] — [arXiv](https://arxiv.org/pdf/2606.25212)
- NVIDIA Isaac Sim Tutorial 15 (sysid with a differentiable Featherstone solver and Adam). Recovers joint friction, damping and mass on synthetic data only. The docs warn this validates the pipeline, not the match to a physical robot. — [Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/tutorial_system_identification_simsim.html)

**Theory.** Chen, Hu, Jin, Li, Wang. "Understanding Domain Randomization for Sim-to-real Transfer." ICLR 2022 (spotlight). It models the simulator as a set of MDPs with tunable parameters such as friction and bounds the sim-to-real gap of DR policies. — [arXiv 2110.03239](https://arxiv.org/abs/2110.03239). Fickinger and Bendahi (2025, arXiv 2506.10133) give a theoretical analysis of offline DR in the DROPO style. [V-abs] — [arXiv](https://arxiv.org/pdf/2506.10133v1)

**Survey.** Muratore, Ramos, Turk, Yu, Gienger, Peters. "Robot Learning from Randomized Simulations: A Review." Frontiers in Robotics and AI 2022. [K for authors; link V] — [Frontiers PDF](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.799893/pdf)

### Inferences
- All of these methods use trajectory-likelihood or trajectory-discrepancy objectives. In our task the informative signal is rare and discrete (slip onset, then rotation). A trajectory loss dominated by nominal arm motion would probably under-weight it. This is why a hand-set range keyed on reproducing the failure worked. A "failure-reproduction likelihood" objective is a natural novel variant of DROPO or BayesSim.
- DROPO and BayesSim output a distribution. Our manual result (shifting the range down and widening it, 0.21-0.64) behaves like their posterior mean plus variance, so either one is a natural quantitative baseline for our hand-tuning.
- ASID's idea of collecting an informative real trajectory maps onto deliberately commanding a squeeze-lift with decreasing grip force until slip occurs, to bracket mu on the real robot.

### Gaps
- Exact real-data budgets for BayesSim and SimOpt (number of real rollouts per iteration) were not verified in full text this session.
- I found no sysid paper evaluated on a bimanual squeeze grasp or a two-contact-patch object.
- I found no paper that calibrates against PhysX/Isaac Lab specifically using real manipulation failures. BayesSimIG targets Isaac Gym, but I did not verify its content.

## Q2. Friction-specific findings and simulator contact discrepancies (PhysX vs MuJoCo vs real)

### Takeaway
There is official Isaac Lab documentation that PhysX's patch friction is "grippier" than MuJoCo-style per-contact friction at the same coefficient. Coefficients are therefore not portable across engines, and a nominal mu in PhysX corresponds to a different effective grip than in reality or in MuJoCo. Peer-reviewed real-vs-sim friction validation for manipulation is sparse. Most engine comparisons are sim-vs-sim or about impacts.

### Cited Findings
- Isaac Lab solver comparison: PhysX TGS uses "Coulomb friction with patch-based anchors". Tangential forces are merged across nearby contacts, and the friction cone is always pyramidal. MJWarp uses MuJoCo's per-contact model with a selectable cone and `impratio`. Tasks tuned for PhysX "can feel 'grippier' than MJWarp's per-contact friction at the same friction coefficient", so expect to raise friction coefficients when porting to MJWarp. — [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/physical-backends/solver-comparison.html)
- Isaac Lab asset migration guide (per search summary; the page has since moved). The same nominal mu does not reproduce the same grasp in PhysX and MJWarp, and MJWarp often slips more. PhysX static/dynamic coefficients and combine modes do not map one-to-one. MuJoCo torsional friction exists only with condim >= 4 (rolling friction with condim 6). — [Isaac Lab docs (moved)](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.html) [V-abs: only the moved-page stub was fetched; content is from the search summary]
- Acosta, Yang, Posa. "Validating Robotics Simulators on Real-World Impacts." RA-L 2022 [K venue]. Drake, MuJoCo and Bullet capture inelastic cube-toss impacts but fail on elastic ones, and model mismatch limits accuracy. It offers a template for fitting contact parameters to real trajectories. — [arXiv 2110.00541](https://arxiv.org/abs/2110.00541)
- Erez, Tassa, Todorov. "Simulation tools for model-based robotics: comparison of Bullet, Havok, MuJoCo, ODE and PhysX." ICRA 2015 [K]. Engine-vs-engine consistency/speed comparison, not real data. Chung and Pollard (2016): no physics engine was best for every task. — [CMU RI PDF](https://www.ri.cmu.edu/app/uploads/2017/12/ChungCAVW2016.pdf)
- Le Lidec et al. "Contact Models in Robotics: a Comparative Analysis." (IEEE T-RO 2024 [K venue]). Compares contact formulations (MuJoCo-style relaxed, LCP, NCP, etc.) and shows they produce different physical behaviours, including spurious effects from relaxations. — [PDF](https://simple-robotics.github.io/publications/contact-models/static/paper/lelidec2024contacts.pdf) [V-abs]
- Hu et al. "Impact of Static Friction on Sim2Real in Robotic RL." arXiv 2503.01255 (2025, preprint, legged robot). Conventional DR omits static (joint) friction. Adding it was necessary for hardware success, compared with DR and an Actuator Net under RMA. It is about joint friction, not contact, but it is an example of a missing friction term dominating sim2real. — [arXiv](https://arxiv.org/abs/2503.01255)
- Josifovski et al. "Analysis of Randomization Effects on Sim2Real Transfer in RL for Robotic Manipulation Tasks." (IROS 2022 [K venue]). More randomization helps transfer but can hurt the ability to find a good policy in sim. This is the range-width trade-off. — [arXiv 2206.06282](https://arxiv.org/abs/2206.06282)
- IndustReal (Tang, Lin, Akinola, Handa, Sukhatme, Ramos, Fox, Narang; RSS 2023). Isaac Gym/Factory, Franka, 83-99% real success on assembly without dynamics randomization. It used a policy-level action integrator for unmodelled dynamics and a sampling-based curriculum. Counterpoint: geometry-dominated insertion transferred without friction DR, unlike friction-dominated squeeze-lift. — [arXiv 2305.17110](https://arxiv.org/abs/2305.17110); [project](https://sites.google.com/usc.edu/industreal)
- Lin et al. "Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids" (arXiv 2502.20396; CoRL 2025 per search snippet). Includes box lift and bimanual handover. Physical parameters including friction are randomized and tuned with real data via an automated real-to-sim tuning module [K for the tuning module detail]. This is the closest bimanual/box-lift sim2real RL work found. — [arXiv](https://arxiv.org/abs/2502.20396)
- Classic friction DR: Peng, Andrychowicz, Zaremba, Abbeel. "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization." ICRA 2018 [K]. Puck pushing on a Fetch, randomizing friction, damping, mass and others. OpenAI "Learning Dexterous In-Hand Manipulation" (IJRR 2020) randomized friction coefficients, noted in Chen et al. — [arXiv 2110.03239](https://arxiv.org/pdf/2110.03239)

### Inferences
- Our finding that the real failure reproduces only at mu ~0.22-0.25 in PhysX, about half the nominal value, is consistent with PhysX patch friction being grippier at a given coefficient. The effective PhysX mu that matches reality can sit well below measured or nominal values, so DR ranges copied from physical datasheets or earlier MuJoCo work would be biased high. This is an inference, not directly tested in any found paper.
- The 2-3x rotation underestimate in Isaac vs MuJoCo may come from patch-merged tangential and torsional resistance across the plate-box contact patch: PhysX merges contacts into patches, while MuJoCo's torsional term is explicit only with condim >= 4. It is unverified. A quick ablation of MuJoCo condim 3 vs 4/6 and PhysX patch vs per-point friction would localise it.

### Gaps
- I found no peer-reviewed real-vs-PhysX measurement of torsional or rotational slip for a squeezed object. The 2-3x discrepancy appears to be new evidence.
- No direct PhysX-vs-MuJoCo-vs-real friction benchmark for manipulation grasps was found. The ComFree-Sim preprint (arXiv 2603.12185) is sim-vs-sim only.

## Q3. Failure-driven / adversarial DR

### Takeaway
Adversarial and active DR search for parameter regions where the current policy is weak or diverges from a reference. They use sim-internal difficulty signals (a discriminator, an adversary's reward, or a success-rate constraint), not real-world failures. Failure-driven calibration anchored to a real observed failure mode is mostly unaddressed.

### Cited Findings
- **Active Domain Randomization.** Mehta, Diaz, Golemo, Pal, Paull. CoRL 2019 (PMLR v100, pp. 1162-1176). SVPG particles propose environment parameters, rewarded by a discriminator that tells randomized rollouts from reference-environment rollouts (a proxy for "informative/difficult"). Uniform DR gave suboptimal, high-variance policies. — [arXiv 1904.04762](https://arxiv.org/abs/1904.04762); [mlanthology](https://mlanthology.org/corl/2019/mehta2019corl-active) [V-abs]
- **Self-Supervised ADR.** Raparthy, Mehta, Golemo, Paull. arXiv 2002.07911 (2020). A coupled environment and goal curriculum via ADR plus self-play. — [arXiv](https://arxiv.org/abs/2002.07911) [V-abs]
- **RARL.** Pinto, Davidson, Sukthankar, Gupta. "Robust Adversarial Reinforcement Learning." ICML 2017. A zero-sum adversary applies disturbance forces, which improves robustness to train/test mismatch. MuJoCo locomotion only, no manipulation or real robot. — [PMLR](https://proceedings.mlr.press/v70/pinto17a.html)
- Adversarial/worst-case DR alternatives such as EPOpt (Rajeswaran et al., ICLR 2017, CVaR over the worst-epsilon fraction of parameter samples). [K, not re-verified]
- Hermann et al. "Adaptive Curriculum Generation from Demonstrations for Sim-to-Real Visuomotor Control" (ICRA 2020 [K venue]). Learns when to increase visual and dynamics randomization. 85% real success on pick-and-stow and 60% on block stacking. — [arXiv 1910.07972](https://arxiv.org/abs/1910.07972) [V-abs]

### Inferences
- Adversarial DR would push toward low friction (the hard region) on its own, but it has no notion of which low-friction value is real. It can over-shoot to infeasible mu where no squeeze force suffices, which is the conservative-policy problem SimOpt warns about. A real-failure-anchored lower bound, our 0.21, is what prevents that.

### Gaps
- I found no paper that uses a real-robot failure episode, rather than full trajectories, as the identification target for DR bounds. This looks like an open niche, with low confidence given the limited search budget.

## Q4. Curricula over DR difficulty: ADR, self-paced DR, DORAEMON; gating and simultaneous-increase failure

### Takeaway
Automatic DR curricula expand randomization ranges only while the policy's performance stays above a threshold. ADR uses per-boundary success buffers. DORAEMON uses an entropy maximisation constrained by success rate, with backtracking. SPDR uses a self-paced KL schedule. ADR adjusts one parameter boundary at a time, which is effectively a staged curriculum. That matches our observation that enabling all difficulty stages simultaneously collapses learning, although I found no paper that ablates simultaneous vs one-at-a-time expansion explicitly.

### Cited Findings
- **ADR (OpenAI).** Akkaya et al. "Solving Rubik's Cube with a Robot Hand." arXiv 1910.07113 (2019). It automatically generates a distribution over environments of ever-increasing difficulty and gave "vastly improved sim2real transfer". — [arXiv](https://arxiv.org/abs/1910.07113) [V-abs]
  - Gating mechanics [K, from paper Sec. 5, not re-fetched]: each parameter has a uniform range [phi_L, phi_H] starting at a point (nominal). With probability p_b, one parameter dimension is fixed at one of its boundaries, and the episode performance is pushed to that boundary's buffer. When the buffer reaches size m, the mean performance is compared to thresholds t_H and t_L: above t_H the boundary is widened by delta, below t_L it is narrowed. Ranges expand per-dimension and per-boundary independently. Paper metric: the "ADR entropy" of the resulting distribution correlated with real-world performance.
- **DORAEMON.** Tiboni, Klink, Peters, Tommasi, D'Eramo, Chalvatzaki. "Domain Randomization via Entropy Maximization." ICLR 2024. Maximises the entropy of the DR (Beta) distribution subject to an in-distribution success-rate constraint alpha, under a KL trust region. If the constraint is violated, a backup step restores performance by backtracking entropy. Sim2real demonstrated on a robotic pushing task [K for the exact real task]. — [arXiv 2311.01885](https://arxiv.org/abs/2311.01885); [mlanthology](https://mlanthology.org/iclr/2024/tiboni2024iclr-domain) [V-abs]
- **Self-Paced DR (SPDR).** Carrasco Damken, Muratore et al. (TU Darmstadt). Builds on Self-Paced RL (Klink et al.). It moves the DR context distribution from easy to target via a KL-penalised self-paced objective with the policy "in the loop". — [TU Darmstadt PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/FabioMuratore/Carrasco_Damken--SelfPacedDomainRandomization.pdf) [V-abs; authors/venue: student thesis/report, uncertain]
- Klink, Abdulsamad, Belousov, D'Eramo, Peters, Pajarinen. "A Probabilistic Interpretation of Self-Paced Learning with Applications to RL" (JMLR 2021) / "Self-Paced Deep RL" (NeurIPS 2020). The underlying self-paced context curriculum. [K]
- Teacher-student curricula: Portelas et al. "Teacher algorithms for curriculum learning of Deep RL in continuous parameterized environments" (ALP-GMM, CoRL 2019), and Matiisen et al. "Teacher-Student Curriculum Learning" (IEEE TNNLS 2019). Both choose tasks by absolute learning progress. [K]
- Domain-specific curricula: IndustReal's sampling-based curriculum (initial-state difficulty) avoids overfitting to an over-constrained early contact stage. — [arXiv 2305.17110](https://arxiv.org/abs/2305.17110). Hermann et al. learn when to increase visual and dynamics DR. — [arXiv 1910.07972](https://arxiv.org/abs/1910.07972)
- Range-width trade-off evidence: Josifovski et al. — [arXiv 2206.06282](https://arxiv.org/abs/2206.06282)

### Inferences
- Our staged curriculum (reward penalties, then goal xy, then low friction, then hover start), gated on success with pausing, is a hand-designed, coarse version of ADR/DORAEMON gating across heterogeneous difficulty axes (reward shaping, task distribution, dynamics, initial state). ADR and DORAEMON only gate dynamics/DR parameters. Mixing reward-term and initial-state stages into one success-gated schedule is less studied.
- Our collapse when everything was enabled at once is consistent with the logic of these methods: ADR moves one boundary per update, and DORAEMON constrains each step by KL and success. Neither starts at full difficulty. A direct baseline would be ADR or DORAEMON on mu only, compared with our staged gate.

### Gaps
- No found paper explicitly ablates "all DR/curriculum axes on at once" against staged expansion in manipulation. That evidence would come from our own experiment.
- ADR thresholds and buffer values and DORAEMON's real-world task details were not re-verified in full text.

## Q5. What gap remains that failure-mode-reproducing calibration in a bimanual contact task could fill?

### Takeaway
Existing real-to-sim methods fit trajectories on mostly single-arm, single-object tasks and assume engine parameters are physically meaningful. Existing DR curricula expand ranges blindly from nominal values. A method that (a) uses observed real failure modes as the calibration target, (b) acknowledges engine-specific effective friction (PhysX grippier than per-contact models), and (c) feeds the identified failure region into a success-gated curriculum on a bimanual squeeze grasp appears to be unoccupied in the literature found.

### Cited Findings
- Sysid baselines match trajectories: SimOpt ([arXiv](https://arxiv.org/abs/1810.05687)), BayesSim ([arXiv](https://arxiv.org/abs/1906.01728)), DROPO ([arXiv](https://arxiv.org/abs/2201.08434)), ASID ([arXiv](https://arxiv.org/abs/2404.12308)). Each was evaluated on single-manipulator tasks (peg, drawer, pushing, shuffleboard, rod balancing).
- Engine friction coefficients are not portable: PhysX patch friction is grippier at the same mu. — [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/physical-backends/solver-comparison.html)
- DR curricula start at nominal and expand: ADR ([arXiv](https://arxiv.org/abs/1910.07113)), DORAEMON ([arXiv](https://arxiv.org/abs/2311.01885)). Neither uses real data to place the range center or bounds.
- The closest bimanual box-lift sim2real RL is Lin et al. 2025 (humanoid hands), which tunes physical parameters with real data but does not target failure modes. — [arXiv](https://arxiv.org/abs/2502.20396)

### Inferences
- Candidate contribution framing: "failure-mode-matched DR". Reproduce a specific real failure signature (plate slip onset, then box yaw/roll) in sim by searching contact parameters. Set the DR range to cover the identified failure-reproducing region plus margin. Then expand with success gating. Our data point (1/12 to 8/10 real by moving mu from 0.45-0.85 to 0.21-0.64) is the empirical hook.
- Natural baselines for that contribution: uniform wide DR, DROPO/BayesSim fitted to the same real logs, ADR/DORAEMON from nominal, and ASID-style exploratory sysid.
- The PhysX-vs-MuJoCo rotation discrepancy (2-3x) adds a second contribution: engine-dependent effective friction means the identified mu is an engine-specific "effective" value, not a material constant. This argues for calibrating per engine against real failure signatures rather than importing measured mu.

### Gaps
- The novelty claim rests on roughly 15 targeted searches. A dedicated Google Scholar sweep for "failure-aware domain randomization", "task-relevant system identification", and Liang et al. 2020 "learning active task-oriented exploration for sysid" [K] is recommended before asserting novelty.
