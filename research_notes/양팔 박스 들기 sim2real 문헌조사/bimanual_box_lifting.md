# Bimanual / Dual-Arm Non-Prehensile Box Lifting (RL + Sim-to-Real) — Prior Work

Our setup, for reference: two Franka FR3 arms with flat 140x140 mm plates on link7 squeeze-lift a 0.5 kg box (310x410x280 mm). The policy is trained with PPO in Isaac Lab and deployed on the real robot. The failure mode we see on hardware: the plates slip 3-8 cm, then the box rotates 20-70 deg about the grasp axis, and the grasp height relative to the CoM decides how much it rotates.

Verification legend:
- **[V]** = verified from the primary page (arXiv abstract or HTML, proceedings page, or project page) during this search.
- **[S]** = only seen in a search-result snippet or abstract; details not checked.
- **[M]** = from prior knowledge and not re-verified in this session. Treat as uncertain.

## Q1. Which papers do RL or learning for dual-arm / bimanual box lifting, whole-arm or palm squeezing, and non-prehensile bimanual grasping with real-robot transfer?

### Takeaway
Several papers report RL policies that squeeze-lift a box with two arms and transfer to hardware: Lin et al. 2025 (humanoid, "grasp-free" box lift, 80% real success), SYMDEX (CoRL 2025, box-lift deployed zero-shot), the Digit box loco-manipulation work, and the 2025-2026 whole-body "embrace" humanoid papers. All of them report only binary success. None reports or analyses how a box slips and rotates under a two-plate squeeze, and none uses flat plates on industrial 7-DoF arms with a heavy, bulky box like ours.

### Cited Findings

**RL / learning, closest to our task**
- **[V] Lin, Sachdev, Fan, Malik, Zhu — "Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids," arXiv 2502.20396 (Feb 2025; later CoRL 2025 per OpenReview listing, venue [S]).** Hardware is a Fourier GR1 humanoid with two 7-DoF arms and multi-fingered hands. The box-lift is a "grasp-free" lift of a box too large for one hand, and box colors, sizes and masses are randomized. The reward is r = r_contact + r_goal, with contact markers placed on the box's side centers. The real box-lift success is 80%, averaged over 10 trials per test object; bimanual handover reached only 52.5%, which the authors attribute to the sim-to-real dynamics gap. The domain-randomization table lists object friction 0.5-1.5 and mass 0.03-0.1 kg (generic, not box-specific). No box-lift failure analysis is given. Relevance: the closest RL squeeze-lift with sim2real, but the box is light, there is no slip or rotation metric, and the contacts are hands rather than flat plates. — [arXiv HTML](https://arxiv.org/html/2502.20396v1/), [OpenReview](https://openreview.net/forum?id=8DHSyMFLbB)
- **[V/S] Li et al. — "Morphologically Symmetric Reinforcement Learning for Ambidextrous Bimanual Manipulation" (SYMDEX), CoRL 2025, PMLR v305.** RL with per-hand subtask policies built as equivariant networks. Box-lift is one of 6 simulated tasks and was deployed zero-shot on a real robot together with table-clean. The sim2real curriculum randomizes the environment and phases in collision and energy penalties. The real-world box-lift success numbers were not confirmed in this search. Relevance: shows that symmetry priors fit two mirrored arms (as our two FR3s are), but it says nothing on slip or rotation. — [PMLR](https://proceedings.mlr.press/v305/li25d.html), [arXiv](https://arxiv.org/abs/2505.05287), [project](https://supersglzc.github.io/projects/symdex)
- **[S] Dao et al. — "Sim-to-Real Learning for Humanoid Box Loco-Manipulation," arXiv 2310.03191 (2023, Digit humanoid; author list [M]).** The authors claim the first learned controller for box pickup and carry on real humanoid hardware. Digit's arms have no grippers, so the box is squeezed between the arms. Relevance: a non-prehensile arm squeeze with sim2real; trial counts were not verified here. — [arXiv](https://arxiv.org/pdf/2310.03191)
- **[V] Zheng, Chen, Bi, Li, Pan, Zhou, Li, Ma — "Embracing Bulky Objects with Humanoid Robots: Whole-Body Manipulation with Reinforcement Learning," arXiv 2509.13534 (Sep 2025, preprint).** RL with a human-motion prior through teacher-student distillation, plus a neural SDF of the object. Arms and torso embrace the object. The robot is a Unitree H1-2 with mocap state, and the object is a 40x60 cm cylinder [S]. The paper reports real-world transfer but no trial counts on the abstract page. — [arXiv](https://arxiv.org/abs/2509.13534)
- **[S] "HULK: Learning Whole-Body Forceful Loco-Manipulation for Humanoids," arXiv 2610.08970 (2026).** MPC-guided RL with two teachers: one for arm motion under wrist forces, one for walking while holding an object against the body. These are distilled into a single policy for large, heavy objects. — [arXiv](https://arxiv.org/html/2610.08970)
- **[S] TRI Punyo — "Learning contact-rich whole-body manipulation with example-guided reinforcement learning," Science Robotics (2025).** A soft upper-body humanoid trained with example-guided RL (one demo per task) and sim2real transfer using domain randomization plus mechanical compliance. Tasks include large boxes and a water jug. Relevance: compliance and soft contact as a sim2real lever, versus our rigid plates. — [Science Robotics](https://www.science.org/doi/10.1126/scirobotics.ads6790)
- **[S] "Zero-shot Whole-Body Manipulation with a Large-Scale Soft Robotic Torso via Guided RL," arXiv 2509.23556.** Reports 88% zero-shot hardware success for whole-body grasping. — [arXiv](https://arxiv.org/pdf/2509.23556)
- **[S] "Learning Dual-Arm Coordination for Grasping Large Flat Objects," arXiv 2504.03500 (2025, IEEE conf. per Xplore link).** A CNN-based PPO policy in Isaac Gym, with transfer to real robots and no fine-tuning claimed. Objects are books and keyboards, and the method is a grasp-and-lift, not a squeeze. — [arXiv](https://arxiv.org/abs/2504.03500), [IEEE](https://ieeexplore.ieee.org/iel8/11127273/11127223/11127654.pdf)
- **[V] Lin, Church, Yang, Li, Lloyd, Zhang, Lepora — "Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning," IEEE RA-L 8(9):5472-5479, 2023.** Two Dobot MG400 arms with TacTip tactile end-effectors, no grippers. Tasks are bi-pushing, bi-reorienting and bi-gathering. Sim2real uses a real-to-sim tactile Pix2Pix GAN. A noted real failure is that the policy "tended to squeeze objects" in bi-reorient, which the authors attribute to the sim2real gap. Relevance: the closest RL work on gripperless dual-arm squeezing with sim2real; the objects are small and are not lifted against gravity. — [arXiv](https://arxiv.org/abs/2307.06423v1), [Bristol](https://research-information.bris.ac.uk/en/publications/bi-touch-bimanual-tactile-manipulation-with-sim-to-real-deep-rein/)
- **[S] "Hybrid Sensor Fusion Mixed with Reinforcement Learning in Autonomous Dual-Arm Lifting Tasks Performed by Humanoid Robots," MDPI Robotics 13(8):121, 2024.** The end-effectors set a pressure (squeeze) force that is held constant while lifting, and the paper also addresses objects with non-parallel contact faces. The full text was blocked (HTTP 403), so the method and trial counts are not verified. — [MDPI](https://www.mdpi.com/2218-6581/13/8/121)
- **[S] BiNoMaP — "Learning Category-Level Bimanual Non-Prehensile Manipulation Primitives," arXiv 2509.21256 (2025).** Bimanual non-prehensile primitives learned from human video, explicitly avoiding RL; the authors cite RL instability and sim2real accuracy loss as the reasons. — [arXiv](https://arxiv.org/html/2509.21256v1)
- **[S] "In the Wild Ungraspable Object Picking with Bimanual Nonprehensile Manipulation," arXiv 2409.15465 (2024).** — [arXiv](https://arxiv.org/pdf/2409.15465)
- **[S] "Learning Cooperative Bimanual Dexterous Grasp of Large Objects with Tactile Sensing," arXiv 2608.10383 (2026).** Humanoid dual-arm grasping with dexterous hands and tactile sensing, learned from teleoperated demonstrations. High success on unseen objects is claimed but numbers were not verified. — [arXiv](https://arxiv.org/html/2608.10383)
- **[S] FlatLab, arXiv 2608.14049 (Zhu et al., 2026).** A simulation benchmark that includes a dual-arm Franka "two-arm lifting" method. It is strong on boxes and paintings and fails on thin or deformable objects. It is simulation-only as far as could be determined. — [arXiv](https://arxiv.org/abs/2608.14049)

**Model-based dual-arm box grabbing with flat or soft pads (non-RL, real robots)**
- **[S] Bombile & Billard — "Dual-arm control for coordinated fast grabbing and tossing of an object," IEEE RA-M (EPFL preprint; venue [S]).** Two KUKA iiwa arms with flat end-effectors grab a box using a modulated dynamical system plus a wrench-optimization QP. — [EPFL PDF](https://infoscience.epfl.ch/record/294383/files/Michael_Bombile_Dual-arm_fast_grabbing_and_tossing_of_an_object_RAM_final.pdf?ln=fr)
- **[S] "Dual-arm Box Grabbing with Impact-aware MPC utilizing soft deformable End-effector Pads" (HAL hal-03602884, ~2022, I.AM. project).** Two Panda arms (same family as FR3) with soft pads and constrained MPC. The full text was blocked by bot protection. — [HAL](https://hal.archives-ouvertes.fr/hal-03602884)
- **[S] Hermus et al., "Impact-Aware Dual-Arm Manipulation: A Framework for Fast Package Handling," IEEE RA-M 2025.** Reports 29% lower task time and 35% lower energy use than classical grabbing. The I.AM. open datasets include dual-Franka box grabbing recordings. — [MBZUAI repo](https://irep.mbzuai.ac.ae/items/e3073cf9-541d-4754-83c5-f688ddf818b6), [4TU data](https://data.4tu.nl/datasets/ab6142db-660e-4737-b7ea-b515bd072cd6/1)
- **[S] Koutras, Ntoliou, Doulgeri — "Towards Passivity Based Nonprehensile Bimanual Manipulation of Large Objects," IEEE-RAS Humanoids 2023, doi 10.1109/Humanoids57100.2023.10375178.** Only the citation was seen; the content is not verified. — [publications page](https://sites.google.com/view/leonidaskoutras/publications)

**Seminal / classical**
- **[S] Ohmura & Kuniyoshi — "Humanoid robot which can lift a 30kg box by whole body contact and tactile feedback," IROS 2007, pp. 1136-1141, doi 10.1109/IROS.2007.4399592.** A whole-body tactile skin lets a humanoid lift a 30 kg box. — [researchr](https://researchr.org/publication/OhmuraK07), [unpaywall](https://unpaywall.org/10.1109%2FIROS.2007.4399592)
- **[S] Smith, Karayiannidis, Nalpantidis, Gratal, Qi, Dimarogonas, Kragic — "Dual arm manipulation—A survey," Robotics and Autonomous Systems 2012.** — [KTH PDF](https://people.kth.se/~dimos/DualSurvey.pdf)
- **[S] Caccavale, Chiacchio, Marino, Villani — "Six-DOF Impedance Control of Dual-Arm Cooperative Manipulators," IEEE/ASME T-Mech 2008.** Separates external (object-environment) impedance from internal (squeeze) impedance. Earlier work: Bonitz & Hsia, "Internal force-based impedance control for cooperating manipulators," 1996. — [search summary](https://next.sciencegate.app/document/10.1109/robot.2000.845171)

### Inferences
- The RL box-lift literature (Lin 2025, SYMDEX, Digit) treats the box as light (Lin's DR mass is up to 0.1 kg, if that range applied to the box) and reports a single success rate. Our 0.5 kg, 41 cm-wide box with flat plates on 7-DoF arms is a heavier and more torque-demanding case than these.
- The most hardware-similar line, dual Panda/iiwa box grabbing in the I.AM. project, is model-based and focused on impact and speed, not on RL or on slip-induced rotation.

### Gaps
- Exact real-robot trial counts for SYMDEX box-lift, the Digit box work, Embracing Bulky Objects, and the MDPI 2024 RL lift were not retrieved.
- Whether the MDPI 2024 paper and the Koutras 2023 paper analyse slip or rotation could not be checked: the MDPI full text returned 403, and only the Koutras citation was found.
- No paper was found that combines RL, two flat plates on industrial arms, a box lift, and real-robot deployment with slip/rotation metrics.

## Q2. Which classical or model-based works analyse friction, squeeze force and object rotation (pivoting) in dual-arm squeeze grasps? Does any analyse grasp height vs CoM causing rotation under slip?

### Takeaway
The physics is established: the moment of gravity about the axis joining the two contacts must be held by torsional friction at the contact patches, which is bounded by about μ·f_n·R_eff. Wong et al. 2026 apply exactly this to dual-arm box lifting with an ellipsoidal limit surface and online CoM estimation. The pivoting literature shows that a CoM offset from the contact line makes gravity rotate the object once grip force drops. No paper was found that studies grasp height vs CoM, or the sequence of translational slip followed by rotation, in a dual-arm squeeze.

### Cited Findings
- **[V] Wong, Yang, Lee, Campolo (NTU) — "Dual-Arm Box Manipulation via Online Inertial Estimation and Convex Wrench Optimization," arXiv 2605.22021 (v2 Oct 2026; submitted to Robotics and Autonomous Systems).**
  - Setup: two 7-DoF Kinova arms with wrist ATI Mini40 F/T sensors and friction-enhanced flat handles (μ=0.4, 100x70 mm patch). The box has a 300x200 mm footprint and weighs 1.7 kg plus a 0.5 kg offset mass that shifts the CoM.
  - Method: after lift-off (15 mm), mass is estimated by least squares and in-plane CoM from moment balance. An ellipsoidal limit surface couples tangential force with torsional moment; the torsional limit is μ·f_n·R_eff, with R_eff derived for a uniform rectangular patch. A 10% safety margin is applied, and an SOCP allocates the contact wrenches. Execution uses Cartesian impedance (K = 1000 N/m, 10 Nm/rad) plus PID force correction.
  - Results: on the real robot, 2 CoM configurations × 3 runs = 6 trials. Mass is overestimated by 3.8-8.7% and CoM error is 3.3-4.5% of object size. Peak tilt (≤0.76°) and hold error (0.12-0.30 mm) are reported only in simulation, and real rotation is not quantified. Baselines are Bombile & Billard's wrench optimization and Bi-Touch.
  - Relevance: this is the closest analytic counterpart to our rotation problem. It handles CoM offset through torsional friction about the contact normal, but it does not analyse grasp height.
  - [arXiv HTML](https://arxiv.org/html/2605.22021)
- **[S] "A two-phase gripper to reorient and grasp" (MIT Open Access; authors, venue and year not verified — [M] likely Chavan-Dafle, Rodriguez, Mason et al., IEEE CASE 2015).** The grip force during pivoting has a lower bound of roughly object weight divided by μ. Any offset between the CoG and the contact points lets gravity pivot the object about the fingertips. A simple torsional-friction approximation is discussed. — [MIT DSpace](https://dspace.mit.edu/bitstream/handle/1721.1/107782/A%20two-phase%20gripper.pdf?sequence=1&isAllowed=y)
- **[S] Toskov et al. — "In-Hand Gravitational Pivoting Using Tactile Sensing," CoRL 2022 (PMLR v205).** Loosening the grip lets gravity torque rotate the object. When an object is held away from its CoM, the gravity-induced torque far exceeds the downward force at the contacts, which causes rotational slip. — [PMLR](https://proceedings.mlr.press/v205/toskov23a/toskov23a.pdf), [arXiv](https://arxiv.org/pdf/2210.05068)
- **[S] "Rotating Objects via In-Hand Pivoting using Vision, Force and Touch," arXiv 2303.10865.** Deliberately induces rotational slip, tracks the angle with vision, and regulates grip force. — [arXiv](https://arxiv.org/pdf/2303.10865)
- **[S] "Motion and Force Planning for Manipulating Heavy Objects by Pivoting," arXiv 2012.06022.** Dual-arm planning of pivoting a heavy object, with grasp-force synthesis under friction-cone and torque limits. — [arXiv](https://arxiv.org/pdf/2012.06022)
- **[S] "Reorienting Objects in 3D Space Using Pivoting," arXiv 1912.02752.** — [arXiv](https://arxiv.org/pdf/1912.02752)
- **[S] DG16M, arXiv 2503.08358.** A dual-arm grasp dataset whose force-closure optimization resists an external wrench at the CoM under friction cones. — [arXiv](https://arxiv.org/html/2503.08358v1)
- **[M] Classical background, not re-verified:**
  - Soft-finger contact and limit surfaces: Goyal, Ruina & Papadopoulos 1991; Howe & Cutkosky 1996, "Practical force-motion models for sliding manipulation," IJRR.
  - Internal/manipulating force decomposition: Yoshikawa & Nagai 1991.
  - Erhart & Hirche, cooperative-manipulation interaction dynamics, T-RO 2016.

### Inferences
- In our geometry, gravity's moment about the grasp axis is about m·g·d, where d is the horizontal offset between the CoM and the line joining the plate centres. Grasp height changes d whenever the box tilts, so slip that changes the contact location (our 3-8 cm) directly changes d. This would produce the "slip then rotate" cascade, because torsional capacity μ·f_n·R_eff is fixed for a given squeeze while the demand grows. This is an inference from the Wong et al. and pivoting models, not a published result for our case.
- Wong et al. model only the in-plane (r_x, r_y) CoM. A grasp height above or below the CoM matters once the box tilts, an effect their formulation (zero bending moments, small tilts) does not study.

### Gaps
- No paper was found that relates grasp height relative to the CoM to rotation magnitude under slip in a two-plate squeeze.
- No measured torsional-friction data was found for rigid metal or plastic plates on cardboard. The effective R_eff for a 140x140 mm plate on a compliant box face is likewise unknown.

## Q3. What evaluation metrics and real-robot trial counts do these papers report?

### Takeaway
Real-robot evaluations are small: 6 runs in Wong et al., and "10 trials per object" in Lin et al. They are dominated by binary success rate. Continuous metrics such as tilt angle, slip distance and hold error appear mainly in simulation, so a quantitative real-robot slip/rotation evaluation would be distinctive.

### Cited Findings
- Lin et al. 2025: box-lift 80% success, averaged over 10 trials per test object; handover 52.5%. — [arXiv](https://arxiv.org/html/2502.20396v1/)
- Wong et al. 2026: 6 real runs (2 CoM configurations × 3); the real metrics are mass and CoM estimation error. Peak tilt angle and hold error are reported in simulation only. — [arXiv](https://arxiv.org/html/2605.22021)
- Soft robotic torso (2509.23556): 88% zero-shot hardware success. — [arXiv](https://arxiv.org/pdf/2509.23556)
- Hermus et al. 2025: task time -29% and energy -35% versus classical grabbing. — [MBZUAI](https://irep.mbzuai.ac.ae/items/e3073cf9-541d-4754-83c5-f688ddf818b6)
- Bi-Touch 2023: real tests on unseen objects under perturbations, with a qualitative squeeze failure. — [arXiv](https://arxiv.org/abs/2307.06423v1)
- SYMDEX: zero-shot real deployment of box-lift is claimed. A third-party blog reports >80% simulation success (unverified). — [PMLR](https://proceedings.mlr.press/v305/li25d.html), [blog](https://davidlxu.github.io/posts/2026/05/symdex-paper-notes/)

### Inferences
- A protocol that reports per-trial slip distance (cm), peak rotation about the grasp axis (deg), squeeze force, and success over N≥20 trials per condition would exceed the reporting depth of every RL work found here.

### Gaps
- Trial counts for SYMDEX, the Digit box work, the Embracing paper, HULK and the MDPI 2024 paper were not retrieved.

## Q4. What do these papers NOT cover that our setup could?

### Takeaway
Our setup could contribute a real-robot characterization of slip followed by rotation for a rigid-plate dual-arm squeeze. It would include the effect of grasp height relative to the CoM, low-friction and heavy/bulky box conditions, and a check of whether simulation ranks policies or grasp heights the same way the real robot does. None of the papers found addresses these.

### Cited Findings (evidence of absence)
- RL box-lift papers report success rate only and use light objects or randomized generic friction (0.5-1.5 in Lin et al.). — [Lin et al.](https://arxiv.org/html/2502.20396v1/)
- The model-based paper with the closest physics (Wong et al.) quantifies tilt only in simulation, uses a high-friction (μ=0.4) handle surface, and does not vary grasp height. — [Wong et al.](https://arxiv.org/html/2605.22021)
- Bi-Touch reports a squeezing failure caused by the sim2real gap but does not analyse it. — [Bi-Touch](https://arxiv.org/abs/2307.06423v1)
- Pivoting works study deliberate gravitational rotation with fingers, not unintended rotation in a two-plate box lift. — [Toskov et al.](https://arxiv.org/pdf/2210.05068)

### Inferences
Gaps our setup could fill:
1. **Slip-then-rotate failure mechanics.** Measure the translational slip and the resulting rotation, and relate them to the torsional friction limit and to grasp height relative to the CoM.
2. **Low-friction rigid flat plates.** Most works add high-friction pads (Wong μ=0.4), soft pads (I.AM. MPC), or tactile skins (Bi-Touch, Ohmura).
3. **Sim2real ranking validity.** Test whether the Isaac Lab ordering of policies or grasp heights holds on the real robot. No paper found reports a rank correlation between simulated and real outcomes for bimanual lifting.
4. **Larger and heavier bulky box on FR3/Panda-class arms with PPO.** Existing RL box-lift work uses humanoids (GR1, Digit, H1-2) or simulation-only Franka setups (FlatLab).
5. **Grasp height as a policy and reward variable.** A CoM-aware grasp-placement reward, or an observation of estimated CoM (Wong-style online estimation), has not been combined with RL.

### Gaps
- The search was limited to about 15 tool calls. Venues like IEEE Xplore were not searched exhaustively, so a niche ICRA/IROS paper on dual-arm box slip or rotation may have been missed. Suggested follow-ups:
  - Koutras et al. 2023 (Humanoids) full text
  - MDPI Robotics 2024 full text
  - The I.AM. soft-pad MPC paper (dual Panda)
  - Search terms: "bimanual hugging", "torsional friction dual-arm", "box tilt dual-arm lift".
