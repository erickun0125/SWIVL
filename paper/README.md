# SWIVL Paper Workspace

This directory contains the technical documentation for the SWIVL paper method section, generated from the actual implementation code.

## Files

### 1. `method.tex`
Complete Method section (Section 3) of the paper, including:
- **3.1 Problem Formulation**: SE(2) setting, frames, wrenches, screw representation
- **3.2 SE(2) Task Space Impedance Control**: Target dynamics, model matching, control law
- **3.3 Screw-Decomposed Impedance Control**: Subspace decomposition, directional compliance
- **3.4 Reference Motion Field Interpretation**: Motion field formulation, stability, wrench modulation
- **3.5 Impedance Variable Learning via RL**: State/action spaces, reward, PPO training
- **3.6 Hierarchical Control Architecture**: 3-level hierarchy (HL planner → Trajectory → LL controller)
- **3.7 Implementation Details**: Environment, configuration system, training infrastructure

### 2. `code_to_paper_mapping.md`
Detailed mapping between mathematical formulations in the paper and actual code implementation:
- Equation numbers → Code locations
- Algorithm steps → Function implementations
- Configuration parameters → YAML files
- Key file summary with class/function references

This document enables:
- **Reproducibility**: Readers can find the exact code implementing each equation
- **Validation**: Reviewers can verify claims against implementation
- **Understanding**: Bridges theory and practice

### 3. `algorithms.tex`
LaTeX algorithm pseudocode for the paper or supplementary material:
- **Algorithm 1**: SE(2) Task Space Impedance Control
- **Algorithm 2**: Screw-Decomposed Impedance Control
- **Algorithm 3**: Hierarchical Control Loop (complete pipeline)
- **Algorithm 4**: Impedance Variable Learning (PPO Training)
- **Algorithm 5**: Minimum Jerk Trajectory Generation
- **Algorithm 6**: Screw Axis Extraction from Object Geometry

These can be directly included in the paper using:
```latex
\input{paper/algorithms.tex}
```

## How the Paper Maps to Code

### Key Mathematical Objects

| Paper Notation | Code Representation | Location |
|----------------|---------------------|----------|
| $T_{si} \in \text{SE}(2)$ | `np.array([x, y, theta])` | Throughout |
| ${}^i V_i$ (body twist) | `body_twist` | `se2_math.py` |
| ${}^s V_i$ (spatial twist) | `spatial_twist` | `se2_math.py` |
| $e_i = \log(T^{-1}T^d)$ | `_compute_pose_error()` | `se2_impedance_controller.py:180` |
| $\Lambda_b$ (task inertia) | `compute_task_space_inertia()` | `se2_dynamics.py:150` |
| $B_i$ (screw axis) | `get_joint_axis_screws()` | `object_manager.py` |
| $(D_d, K_d)$ | `ImpedanceGains` | `task_space_impedance.py:50` |

### Key Equations

| Equation | Implementation | File:Line |
|----------|----------------|-----------|
| Eq. 1 (Pose error) | `log(T_inv @ T_des)` | `se2_impedance_controller.py:180` |
| Eq. 2 (Model matching) | `M_d = Lambda_b` | `se2_impedance_controller.py:150` |
| Eq. 3 (Impedance wrench) | `compute_wrench()` | `se2_impedance_controller.py:200-220` |
| Eq. 4-5 (Screw decomposition) | `_decompose_twist()` | `se2_screw_decomposed_impedance.py:250-270` |
| Eq. 7 (Screw wrench) | `compute_control()` | `se2_screw_decomposed_impedance.py:300-330` |
| Eq. 9 (RL state) | `_get_rl_observation()` | `impedance_learning_env.py:430-440` |
| Eq. 12 (Reward) | `_compute_reward()` | `impedance_learning_env.py:475-483` |

### Architecture Components

```
Hierarchical Control (Section 3.6)
├── Level 1: High-Level Planner
│   ├── Flow Matching: src/hl_planners/flow_matching.py
│   ├── Diffusion Policy: src/hl_planners/diffusion_policy.py
│   └── ACT: src/hl_planners/act.py
│
├── Level 2: Trajectory Generator
│   └── Minimum Jerk: src/trajectory_generator.py (Algorithm 5)
│
└── Level 3: Low-Level Controller
    ├── RL Policy: src/rl_policy/ppo_impedance_policy.py
    ├── SE(2) Impedance: src/ll_controllers/se2_impedance_controller.py
    └── Screw Impedance: src/ll_controllers/se2_screw_decomposed_impedance.py
```

## Training and Evaluation

All experiments are configured via `scripts/configs/rl_config.yaml` and executed through:

### High-Level Policy Training (Imitation Learning)
```bash
python scripts/training/train_hl_policy.py --policy flow_matching
```

### Low-Level Policy Training (RL)
```bash
# SE(2) Impedance Controller
python scripts/training/train_ll_policy.py \
    --hl_policy flow_matching \
    --controller se2_impedance

# Screw-Decomposed Controller
python scripts/training/train_ll_policy.py \
    --hl_policy diffusion \
    --controller screw_decomposed
```

### Evaluation
```bash
python scripts/evaluation/evaluate_hierarchical.py \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --num_episodes 50
```

## Reproducibility

Every equation, algorithm, and claim in the Method section can be traced back to specific code:

1. **Mathematical formulations** → `method.tex`
2. **Algorithm pseudocode** → `algorithms.tex`
3. **Code implementation** → `code_to_paper_mapping.md`
4. **Experimental setup** → `scripts/configs/rl_config.yaml`

This ensures complete transparency and reproducibility of all results.

## Usage in Paper

### Main Paper
Include the method section:
```latex
\input{paper/method.tex}
```

Include selected algorithms (e.g., Algorithm 3):
```latex
\input{paper/algorithms.tex}
```

### Supplementary Material
Include the complete code mapping for reviewers:
```latex
\section{Implementation Details}
% Convert code_to_paper_mapping.md to LaTeX
```

## Key Contributions Validated by Code

1. **SE(2) Task Space Impedance Control**:
   - Implemented in `se2_impedance_controller.py`
   - Model matching for passivity (line 150)
   - Full dynamics compensation (lines 200-220)

2. **Screw Decomposition**:
   - Implemented in `se2_screw_decomposed_impedance.py`
   - Configuration-invariant screw representation
   - Directional compliance (lines 300-330)

3. **Reference Motion Field**:
   - Trajectory generation: `trajectory_generator.py`
   - Minimum jerk interpolation (Algorithm 5)
   - Smooth desired acceleration for feedforward

4. **Wrench-Aware RL Policy**:
   - Environment: `impedance_learning_env.py`
   - 30D state including wrench feedback
   - Controller-agnostic learning (both SE(2) and screw)

5. **Hierarchical Architecture**:
   - Fully modular 3-level design
   - Planner-agnostic interface
   - All combinations tested (HL policy × controller type)

## Notes for Authors

When writing the paper:
- Use notation from `method.tex` consistently
- Reference algorithms by number from `algorithms.tex`
- Include code snippets from `code_to_paper_mapping.md` in supplementary
- All hyperparameters in `rl_config.yaml` should match the paper

When addressing reviewer questions:
- Point to specific files/lines in `code_to_paper_mapping.md`
- All claims are verifiable in the implementation
- Experiments are reproducible via provided scripts

## Contact

For questions about the implementation or paper, please open an issue in the repository.
